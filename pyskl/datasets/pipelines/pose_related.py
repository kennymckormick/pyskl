# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from scipy.stats import mode as get_mode

from ..builder import PIPELINES
from .compose import Compose
from .formatting import Rename

EPS = 1e-4


@PIPELINES.register_module()
class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class PreNormalize2D:
    """Normalize the range of keypoint values. """

    def __init__(self, img_shape=(1080, 1920)):
        self.img_shape = img_shape

    def __call__(self, results):
        h, w = results.get('img_shape', self.img_shape)
        results['keypoint'][..., 0] = (results['keypoint'][..., 0] - (w / 2)) / (w / 2)
        results['keypoint'][..., 1] = (results['keypoint'][..., 1] - (h / 2)) / (h / 2)
        return results


@PIPELINES.register_module()
class RandomRot:

    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
        results['keypoint'] = np.einsum('ab,mtvb->mtva', rot_mat, skeleton)

        return results


@PIPELINES.register_module()
class RandomScale:

    def __init__(self, scale=0.2):
        assert isinstance(scale, tuple) or isinstance(scale, float)
        self.scale = scale

    def __call__(self, results):
        skeleton = results['keypoint']
        scale = self.scale
        if isinstance(scale, float):
            scale = (scale, ) * skeleton.shape[-1]
        assert len(scale) == skeleton.shape[-1]
        scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
        results['keypoint'] = skeleton * scale
        return results


@PIPELINES.register_module()
class RandomGaussianNoise:

    def __init__(self, sigma=0.01, base='frame', shared=False):
        assert isinstance(sigma, float)
        self.sigma = sigma
        self.base = base
        self.shared = shared
        assert self.base in ['frame', 'video']
        if self.base == 'frame':
            assert not self.shared

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape
        skeleton = skeleton.reshape(-1, V, C)
        ske_min, ske_max = skeleton.min(axis=1), skeleton.max(axis=1)
        # MT * C
        flag = ((ske_min ** 2).sum(axis=1) > EPS)
        # MT
        if self.base == 'frame':
            norm = np.linalg.norm(ske_max - ske_min, axis=1) * flag
            # MT
        elif self.base == 'video':
            assert np.sum(flag)
            ske_min, ske_max = ske_min[flag].min(axis=0), ske_max[flag].max(axis=0)
            # C
            norm = np.linalg.norm(ske_max - ske_min)
            norm = np.array([norm] * (M * T)) * flag
        # MT * V
        if self.shared:
            noise = np.random.randn(V) * self.sigma
            noise = np.stack([noise] * (M * T))
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            random_vec = np.concatenate([random_vec] * (M * T), axis=-1)
        else:
            noise = np.random.randn(M * T, V) * self.sigma
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, M * T * V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            # C * MTV
        random_vec = random_vec * noise.reshape(-1)
        # C * MTV
        random_vec = (random_vec.T).reshape(M, T, V, C)
        results['keypoint'] = skeleton.reshape(M, T, V, C) + random_vec
        return results


@PIPELINES.register_module()
class PreNormalize3D:
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z). Codes adapted from https://github.com/lshiwjx/2s-AGCN. """

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __init__(self, zaxis=[0, 1], xaxis=[8, 4], align_spine=True, align_center=True):
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def __call__(self, results):
        skeleton = results['keypoint']
        total_frames = results.get('total_frames', skeleton.shape[1])

        M, T, V, C = skeleton.shape
        assert T == total_frames
        if skeleton.sum() == 0:
            return results

        index0 = [i for i in range(T) if not np.all(np.isclose(skeleton[0, i], 0))]

        assert M in [1, 2]
        if M == 2:
            index1 = [i for i in range(T) if not np.all(np.isclose(skeleton[1, i], 0))]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        results['keypoint'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results


@PIPELINES.register_module()
class JointToBone:

    def __init__(self, dataset='nturgb+d', target='keypoint'):
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'coco']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                          (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                          (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)]
        elif self.dataset == 'openpose':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                          (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                          (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))

    def __call__(self, results):

        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'coco']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


@PIPELINES.register_module()
class ToMotion:

    def __init__(self, dataset='nturgb+d', source='keypoint', target='motion'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, :T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ['openpose', 'coco']:
            score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, :T - 1, :, 2] = score

        results[self.target] = motion

        return results


@PIPELINES.register_module()
class MergeSkeFeat:
    def __init__(self, feat_list=['keypoint'], target='keypoint', axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis. """

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results


@PIPELINES.register_module()
class GenSkeFeat:
    def __init__(self, dataset='nturgb+d', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(Rename({'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)


@PIPELINES.register_module()
class PadTo:

    def __init__(self, length, mode='loop'):
        self.length = length
        assert mode in ['loop', 'zero']
        self.mode = mode

    def __call__(self, results):
        total_frames = results['total_frames']
        assert total_frames <= self.length
        inds = np.arange(self.length)
        inds = np.mod(inds, total_frames)

        keypoint = results['keypoint'][:, inds].copy()
        if self.mode == 'zero':
            keypoint[:, total_frames:] = 0
        results['keypoint'] = keypoint
        results['total_frames'] = self.length
        return results


@PIPELINES.register_module()
class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class DecompressPose:
    """Load Compressed Pose

    In compressed pose annotations, each item contains the following keys:
    Original keys: 'label', 'frame_dir', 'img_shape', 'original_shape', 'total_frames'
    New keys: 'frame_inds', 'keypoint', 'anno_inds'.
    This operation: 'frame_inds', 'keypoint', 'total_frames', 'anno_inds'
         -> 'keypoint', 'keypoint_score', 'total_frames'

    Args:
        squeeze (bool): Whether to remove frames with no human pose. Default: True.
        max_person (int): The max number of persons in a frame, we keep skeletons with scores from high to low.
            Default: 10.
    """

    def __init__(self,
                 squeeze=True,
                 max_person=10):

        self.squeeze = squeeze
        self.max_person = max_person

    def __call__(self, results):

        required_keys = ['total_frames', 'frame_inds', 'keypoint']
        for k in required_keys:
            assert k in results

        total_frames = results['total_frames']
        frame_inds = results.pop('frame_inds')
        keypoint = results['keypoint']

        if 'anno_inds' in results:
            frame_inds = frame_inds[results['anno_inds']]
            keypoint = keypoint[results['anno_inds']]

        assert np.all(np.diff(frame_inds) >= 0), 'frame_inds should be monotonical increasing'

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        results['total_frames'] = total_frames

        num_joints = keypoint.shape[1]
        num_person = get_mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_joints, 2], dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_joints], dtype=np.float16)
        # 32768 is enough
        nperson_per_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, keypoint):
            person_ind = nperson_per_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            nperson_per_frame[frame_ind] += 1

        if num_person > self.max_person:
            for i in range(total_frames):
                nperson = nperson_per_frame[i]
                val = new_kpscore[:nperson, i]
                score_sum = val.sum(-1)

                inds = sorted(range(nperson), key=lambda x: -score_sum[x])
                new_kpscore[:nperson, i] = new_kpscore[inds, i]
                new_kp[:nperson, i] = new_kp[inds, i]
            num_person = self.max_person
            results['num_person'] = num_person

        results['keypoint'] = new_kp[:num_person]
        results['keypoint_score'] = new_kpscore[:num_person]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(squeeze={self.squeeze}, max_person={self.max_person})')
