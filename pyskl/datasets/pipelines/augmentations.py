# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import random
import warnings
from collections.abc import Sequence
from torch.nn.modules.utils import _pair

from ..builder import PIPELINES


def _combine_quadruple(a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])


@PIPELINES.register_module()
class PoseCompact:
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required keys in results are "img_shape", "keypoint", add or modified keys
    are "img_shape", "keypoint", "crop_quadruple".

    Args:
        padding (float): The padding size. Default: 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Default: 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Default: None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Default: True.

    Returns:
        type: Description of returned object.
    """

    def __init__(self,
                 padding=0.25,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def __call__(self, results):
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

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

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


@PIPELINES.register_module()
class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs".

    Args:
        size (int): The output size of the images.
    """

    def __init__(self, size):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        # Process entity boxes
        if 'gt_bboxes' in results:
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(size={self.size})'
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        if 'gt_bboxes' in results:
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range})')
        return repr_str


@PIPELINES.register_module()
class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "resize_size".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if 'imgs' in results:
            results['imgs'] = self._resize_imgs(results['imgs'], new_w, new_h)
        if 'keypoint' in results:
            results['keypoint'] = self._resize_kps(results['keypoint'], self.scale_factor)

        if 'gt_bboxes' in results:
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'], self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "flip_direction".
    The Flip augmentation should be placed after any cropping / reshaping
    augmentations, to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    def _flip_imgs(self, imgs, modality):
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'keypoint' in results:
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'], modality)
            if 'keypoint' in results:
                kp = results['keypoint']
                kpscore = results.get('keypoint_score', None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results['keypoint'] = kp
                if 'keypoint_score' in results:
                    results['keypoint_score'] = kpscore

        if 'gt_bboxes' in results and flip:
            assert self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map})')
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB' or 'RGB' in modality:
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(RandomCrop):
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "img_shape".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        if 'gt_bboxes' in results:
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop:
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side. Required keys are "imgs", "img_shape", added or modified keys
    are "imgs", "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('ThreeCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop:
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally. Required keys are "imgs",
    "img_shape", added or modified keys are "imgs", "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('TenCrop cannot process bounding boxes')

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str
