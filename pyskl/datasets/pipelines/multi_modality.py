import numpy as np
from torch.nn.modules.utils import _pair

from ..builder import PIPELINES
from .loading import DecordDecode, DecordInit
from .pose_related import PoseDecode
from .sampling import UniformSampleFrames

EPS = 1e-4


@PIPELINES.register_module()
class MMPad:

    def __init__(self, hw_ratio=None, padding=0.):
        if isinstance(hw_ratio, float):
            hw_ratio = (hw_ratio, hw_ratio)
        self.hw_ratio = hw_ratio
        self.padding = padding

    # New shape is larger than old shape
    def _pad_kps(self, keypoint, old_shape, new_shape):
        offset_y = int((new_shape[0] - old_shape[0]) / 2)
        offset_x = int((new_shape[1] - old_shape[1]) / 2)
        offset = np.array([offset_x, offset_y], dtype=np.float32)
        keypoint[..., :2] += offset
        return keypoint

    def _pad_imgs(self, imgs, old_shape, new_shape):
        diff_y, diff_x = new_shape[0] - old_shape[0], new_shape[1] - old_shape[1]
        return [
            np.pad(
                img, ((diff_y // 2, diff_y - diff_y // 2),
                      (diff_x // 2, diff_x - diff_x // 2), (0, 0)),
                'constant',
                constant_values=127) for img in imgs
        ]

    def __call__(self, results):
        h, w = results['img_shape']
        h, w = h * (1 + self.padding), w * (1 + self.padding)

        if self.hw_ratio is not None:
            h = max(self.hw_ratio[0] * w, h)
            w = max(1 / self.hw_ratio[1] * h, w)
        h, w = int(h + 0.5), int(w + 0.5)

        if 'keypoint' in results:
            results['keypoint'] = self._pad_kps(results['keypoint'], results['img_shape'], (h, w))

        if 'imgs' in results:
            results['imgs'] = self._pad_imgs(results['imgs'], results['img_shape'], (h, w))

        results['img_shape'] = (h, w)
        return results


@PIPELINES.register_module()
class MMUniformSampleFrames(UniformSampleFrames):

    def __call__(self, results):
        num_frames = results['total_frames']
        modalities = []
        for modality, clip_len in self.clip_len.items():
            if results['test_mode']:
                inds = self._get_test_clips(num_frames, clip_len)
            else:
                inds = self._get_train_clips(num_frames, clip_len)
            inds = np.mod(inds, num_frames)
            results[f'{modality}_inds'] = inds.astype(np.int)
            modalities.append(modality)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        if not isinstance(results['modality'], list):
            # should override
            results['modality'] = modalities
        return results


@PIPELINES.register_module()
class MMDecode(DecordInit, DecordDecode, PoseDecode):
    def __init__(self, io_backend='disk', **kwargs):
        DecordInit.__init__(self, io_backend=io_backend, **kwargs)
        DecordDecode.__init__(self)
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        for mod in results['modality']:
            if results[f'{mod}_inds'].ndim != 1:
                results[f'{mod}_inds'] = np.squeeze(results[f'{mod}_inds'])
            frame_inds = results[f'{mod}_inds']
            if mod == 'RGB':
                if 'filename' not in results:
                    results['filename'] = results['frame_dir'] + '.mp4'
                video_reader = self._get_videoreader(results['filename'])
                imgs = self._decord_load_frames(video_reader, frame_inds)
                del video_reader
                results['imgs'] = imgs
            elif mod == 'Pose':
                assert 'keypoint' in results
                if 'keypoint_score' not in results:
                    keypoint_score = [
                        np.ones(keypoint.shape[:-1], dtype=np.float32)
                        for keypoint in results['keypoint']
                    ]
                    results['keypoint_score'] = keypoint_score
                results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)
                results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)
            else:
                raise NotImplementedError(f'MMDecode: Modality {mod} not supported')

        # We need to scale human keypoints to the new image size
        if 'imgs' in results:
            real_img_shape = results['imgs'][0].shape[:2]
            if real_img_shape != results['img_shape']:
                oh, ow = results['img_shape']
                nh, nw = real_img_shape

                assert results['keypoint'].shape[-1] in [2, 3]
                results['keypoint'][..., 0] *= (nw / ow)
                results['keypoint'][..., 1] *= (nh / oh)

                results['img_shape'] = real_img_shape
                results['original_shape'] = real_img_shape

        return results


@PIPELINES.register_module()
class MMCompact:

    def __init__(self, padding=0.25, threshold=10, hw_ratio=1, allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def _get_box(self, keypoint, img_shape):
        # will return x1, y1, x2, y2
        h, w = img_shape

        kp_x = keypoint[..., 0]
        kp_y = keypoint[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
        return (min_x, min_y, max_x, max_y)

    def _compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [
                np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0))) for img in imgs
            ]
        imgs = [img[min_y: max_y, min_x: max_x] for img in imgs]
        return imgs

    def __call__(self, results):
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']
        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        min_x, min_y, max_x, max_y = self._get_box(kp, img_shape)

        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape
        results['imgs'] = self._compact_images(results['imgs'], img_shape, (min_x, min_y, max_x, max_y))
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str
