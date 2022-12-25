import numpy as np

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
    def _pad_kps(self, kps, old_shape, new_shape):
        offset_y = int((new_shape[0] - old_shape[0]) / 2)
        offset_x = int((new_shape[1] - old_shape[1]) / 2)
        offset = np.array([offset_x, offset_y], dtype=np.float32)
        return [kp + offset for kp in kps]

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
        if 'kp' in results:
            results['kp'] = self._pad_kps(results['kp'], results['img_shape'],
                                          (h, w))
        # img_shape should be: if not identical to results['img_shape'],
        # at least propotional, just a patch here
        if 'imgs' in results:
            real_img_shape = results['imgs'][0].shape[:2]
            real_h, real_w = real_img_shape
            real_h_ratio = results['img_shape'][0] / real_h
            real_w_ratio = results['img_shape'][1] / real_w
            # almost identical
            # assert np.abs(real_h_ratio - real_w_ratio) < 2e-2

            if real_h == results['img_shape'][0]:
                results['imgs'] = self._pad_imgs(results['imgs'],
                                                 results['img_shape'], (h, w))
            else:
                results['imgs'] = self._pad_imgs(results['imgs'],
                                                 (real_h, real_w),
                                                 (int(h / real_h_ratio + 0.5),
                                                  int(w / real_w_ratio + 0.5)))
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
                keypoint = results['keypoint'].copy()

                assert keypoint.shape[-1] == 2
                non_zero = (np.linalg.norm(keypoint, axis=-1) > EPS) * (keypoint > EPS)

                keypoint[..., 0] *= (nw / ow)
                keypoint[..., 1] *= (nh / oh)

                results['keypoint'] = keypoint * non_zero[..., None] + results['keypoint'] * (1 - non_zero)[..., None]
                results['img_shape'] = real_img_shape
                results['original_shape'] = real_img_shape

        return results
