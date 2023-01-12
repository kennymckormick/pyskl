# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from collections import defaultdict

from ..builder import PIPELINES
from .sampling import UniformSampleFrames

EPS = 1e-4


@PIPELINES.register_module()
class KineticsSSampling(UniformSampleFrames):

    def __init__(self,
                 num_clips=1,
                 clip_olen=60,
                 clip_len=30,
                 step=30,
                 seed=255,
                 num_skeletons=20,
                 squeeze=True,
                 iou_thre=0,
                 track_method='bbox'):

        super().__init__(num_clips=num_clips, clip_len=clip_len, seed=seed)
        nske = num_skeletons
        if isinstance(nske, int):
            assert nske >= 1
            nske = (nske, nske)
        else:
            assert len(nske) == 2 and 0 < nske[0] <= nske[1]
        self.min_skeletons, self.max_skeletons = nske
        self.clip_olen = clip_olen
        self.step = step
        assert track_method in ['bbox', 'score', 'bbox&score', 'bbox_propagate', 'bbox_still']
        self.track_method = track_method

        from mmdet.core import BboxOverlaps2D
        self.iou_calc = BboxOverlaps2D()
        assert iou_thre in list(range(6))
        self.iou_thre = iou_thre
        assert iou_thre >= 0
        self.squeeze = squeeze

    @staticmethod
    def auto_box(kpts, thre=0.3, expansion=1.25, default_shape=(320, 426)):
        # It can return None if the box is too small
        assert len(kpts.shape) == 3 and kpts.shape[-1] == 3
        score = kpts[..., 2]
        flag = score >= thre
        boxes = []
        img_h, img_w = default_shape
        for i, kpt in enumerate(kpts):
            remain = kpt[flag[i]]
            if remain.shape[0] < 2:
                boxes.append([0, 0, img_w, img_h])
            else:
                min_x, max_x = np.min(remain[:, 0]), np.max(remain[:, 0])
                min_y, max_y = np.min(remain[:, 1]), np.max(remain[:, 1])
                if max_x - min_x < 10 or max_y - min_y < 10:
                    boxes.append([0, 0, img_w, img_h])
                else:
                    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
                    lx, ly = (max_x - min_x) / 2 * expansion, (max_y - min_y) / 2 * expansion
                    min_x, max_x = cx - lx, cx + lx
                    min_y, max_y = cy - ly, cy + ly
                    boxes.append([min_x, min_y, max_x, max_y])
        return np.array(boxes).astype(np.float32)

    def sample_seq_meta(self, ind2info, total_frames, test_mode):
        if total_frames <= self.clip_olen:
            max_idx, max_val = -1, 0
            for i in ind2info:
                if len(ind2info[i]) > max_val:
                    max_val = len(ind2info[i])
                    max_idx = i
            return [(0, total_frames, max_idx, i) for i in range(max_val)]
        if test_mode:
            offset = 0
            np.random.seed(self.seed)
        else:
            offset = np.random.randint(0, min(self.step, total_frames - self.clip_olen))
        windows = []

        clip_olen = self.clip_olen
        for start in range(offset, total_frames - self.step, self.step):
            break_flag = False
            if start + clip_olen >= total_frames:
                start = total_frames - clip_olen
                break_flag = True

            center_frame = start + clip_olen // 2 if test_mode else np.random.randint(start, start + clip_olen)
            num_boxes = len(ind2info[center_frame])
            windows.extend([(start, start + clip_olen, center_frame, i) for i in range(num_boxes)])
            if break_flag:
                break
        return windows

    def track_pre_computed(self, ind2info, start, end, center_idx, box_idx, keypoint, prevnext):
        inst = ind2info[center_idx][box_idx]
        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        prev, nxt = prevnext

        for i in range(center_idx - 1, start - 1, -1):
            p = prev[inst]
            if p == 65535:
                break
            kpt_ret[i - start] = keypoint[p]
            inst = p
        inst = original_inst
        for i in range(center_idx + 1, end):
            n = nxt[inst]
            if n == 65535:
                break
            kpt_ret[i - start] = keypoint[n]
            inst = n
        return kpt_ret

    def track_by_score(self, ind2info, start, end, center_idx, box_idx, keypoint, score_rank):
        inst = ind2info[center_idx][box_idx]
        rank = None
        for k in score_rank[center_idx]:
            if score_rank[center_idx][k] == inst:
                rank = k
        assert rank is not None

        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)

        for i in range(start, end):
            if rank in score_rank[i]:
                cinst = score_rank[i][rank]
                kpt_ret[i - start] = keypoint[cinst]
        return kpt_ret

    def track_by_ious(self, ind2info, start, end, center_idx, box_idx, keypoint, ious):
        # We track by bounding box
        inst = ind2info[center_idx][box_idx]
        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        cur_box_id = box_idx

        for i in range(center_idx - 1, start - 1, -1):
            iou = ious[i].T
            cur_box_id = np.argmax(iou[cur_box_id])
            kpt_ret[i - start] = keypoint[ind2info[i][cur_box_id]]

        inst, cur_box_id = original_inst, box_idx

        for i in range(center_idx + 1, end):
            iou = ious[i - 1]
            cur_box_id = np.argmax(iou[cur_box_id])
            kpt_ret[i - start] = keypoint[ind2info[i][cur_box_id]]
        return kpt_ret

    def track_by_bbox(self, ind2info, start, end, center_idx, box_idx, keypoint, bbox):
        # We track by bounding box
        inst = ind2info[center_idx][box_idx]
        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        cur_box = bbox[inst]

        if self.track_method == 'bbox_propagate':
            for i in range(center_idx - 1, start - 1, -1):
                bboxes = torch.tensor([bbox[x] for x in ind2info[i]])
                cur_box_t = torch.tensor(cur_box)[None]
                ious = self.iou_calc(cur_box_t, bboxes)[0]
                idx = torch.argmax(ious)
                if ious[idx] < self.iou_thre / 10:
                    break
                kpt_ret[i - start] = keypoint[ind2info[i][idx]]
                cur_box = bbox[ind2info[i][idx]]
            inst = original_inst
            cur_box = bbox[inst]
            for i in range(center_idx + 1, end):
                bboxes = torch.tensor([bbox[x] for x in ind2info[i]])
                cur_box_t = torch.tensor(cur_box)[None]
                ious = self.iou_calc(cur_box_t, bboxes)[0]
                idx = torch.argmax(ious)
                if ious[idx] < self.iou_thre / 10:
                    break
                kpt_ret[i - start] = keypoint[ind2info[i][idx]]
                cur_box = bbox[ind2info[i][idx]]
        elif self.track_method == 'bbox_still':
            st_inst, ed_inst = min(ind2info[start]), max(ind2info[end - 1])
            bboxes = torch.tensor(bbox[st_inst: ed_inst + 1])
            cur_box_t = torch.tensor(cur_box)[None]
            ious = self.iou_calc(cur_box_t, bboxes)[0]
            for t in range(start, end):
                if t == center_idx:
                    continue
                box_st, box_ed = min(ind2info[t]), max(ind2info[t])
                box_idx_t = torch.argmax(ious[box_st - st_inst: box_ed - st_inst + 1]) + box_st
                if torch.max(ious[box_st - st_inst: box_ed - st_inst + 1]) >= self.iou_thre / 10:
                    kpt_ret[t - start] = keypoint[box_idx_t]
        return kpt_ret

    def track_by_bns(self, ind2info, start, end, center_idx, box_idx, keypoint, prevnext, score_rank):
        inst = ind2info[center_idx][box_idx]
        rank = None
        for k in score_rank[center_idx]:
            if score_rank[center_idx][k] == inst:
                rank = k
        assert rank is not None

        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        prev, nxt = prevnext

        for i in range(center_idx - 1, start - 1, -1):
            p = prev[inst]
            if p == 65535:
                if rank in score_rank[i]:
                    p = score_rank[i][rank]
                else:
                    break
            kpt_ret[i - start] = keypoint[p]
            inst = p

        inst = original_inst
        for i in range(center_idx + 1, end):
            n = nxt[inst]
            if n == 65535:
                if rank in score_rank[i]:
                    n = score_rank[i][rank]
                else:
                    break
            kpt_ret[i - start] = keypoint[n]
            inst = n
        return kpt_ret

    def _get_score_rank(self, keypoint, ind2info):
        scores = keypoint[..., 2].sum(axis=-1)
        score_rank = dict()
        for find in ind2info:
            score_rank[find] = dict()
            seg = ind2info[find]
            s, e = min(seg), max(seg) + 1
            score_sub = scores[s: e]
            order_sub = (-score_sub).argsort()
            rank_sub = order_sub.argsort()
            for i in range(e - s):
                idx = s + i
                score_rank[find][rank_sub[i]] = idx
        return score_rank

    def __call__(self, results):
        keypoint = results['keypoint']
        ske_frame_inds = results['ske_frame_inds']
        total_frames = results['total_frames']
        test_mode = results['test_mode']

        def mapinds(ske_frame_inds):
            uni = np.unique(ske_frame_inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in ske_frame_inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            ske_frame_inds = mapinds(ske_frame_inds)
            total_frames = np.max(ske_frame_inds) + 1
            results['ske_frame_inds'], results['total_frames'] = ske_frame_inds, total_frames

        kpt_shape = keypoint[0].shape

        assert keypoint.shape[0] == ske_frame_inds.shape[0]

        h, w = results['img_shape']
        bbox = results.get('bbox', self.auto_box(keypoint, default_shape=results['img_shape']))
        bbox = bbox.astype(np.float32)
        bbox[:, 0::2] /= w
        bbox[:, 1::2] /= h
        bbox = np.clip(bbox, 0, 1)

        ind2info = defaultdict(list)
        for i, find in enumerate(ske_frame_inds):
            ind2info[find].append(i)

        seq_meta = self.sample_seq_meta(ind2info, total_frames, test_mode)

        kpt_ret, stinfos = [], []
        ious = results.pop('ious', None)

        if ious is not None:
            assert len(ious) == len(set(ske_frame_inds)) - 1

        if 'score' in self.track_method:
            score_rank = results.pop('score_rank', self._get_score_rank(keypoint, ind2info))

        prevnext = results.pop('iouthr_pn', None)
        if prevnext is not None:
            assert prevnext.shape == (6, 2, keypoint.shape[0])
            prevnext = prevnext[self.iou_thre]

        # In train mode, we use max_skeletons
        if not test_mode and len(seq_meta) > self.max_skeletons:
            assert self.num_clips == 1
            indices = self._get_train_clips(len(seq_meta), self.max_skeletons)
            seq_meta = [seq_meta[i] for i in indices]

        if self.track_method == 'bbox' and ious is None and prevnext is None:
            self.track_method = 'bbox_propagate'
        if self.track_method in ['bbox_propagate', 'bbox_still']:
            prevnext, ious = None, None

        for item in seq_meta:
            start, end, center_idx, box_idx = item
            cur_ske_idx = ind2info[center_idx][box_idx]
            cur_kpt, cur_box = keypoint[cur_ske_idx], bbox[cur_ske_idx]
            stinfo = np.array(list(cur_box) + [center_idx / total_frames, np.mean(cur_kpt[:, 2])], dtype=np.float32)
            stinfos.append(stinfo)
            if self.track_method in ['bbox_propagate', 'bbox_still']:
                kpt = self.track_by_bbox(ind2info, start, end, center_idx, box_idx, keypoint, bbox)
            elif self.track_method == 'score':
                kpt = self.track_by_score(ind2info, start, end, center_idx, box_idx, keypoint, score_rank)
            elif self.track_method == 'bbox':
                if prevnext is not None:
                    kpt = self.track_pre_computed(ind2info, start, end, center_idx, box_idx, keypoint, prevnext)
                elif ious is not None:
                    kpt = self.track_by_ious(ind2info, start, end, center_idx, box_idx, keypoint, ious)
            elif self.track_method == 'bns':
                kpt = self.track_by_bns(ind2info, start, end, center_idx, box_idx, keypoint, prevnext, score_rank)

            if test_mode:
                indices = self._get_test_clips(end - start, self.clip_len)
            else:
                indices = self._get_train_clips(end - start, self.clip_len)

            indices = np.mod(indices, end - start)
            kpt = kpt[indices].reshape((self.num_clips, self.clip_len, *kpt_shape))
            kpt_ret.append(kpt)

        # Aug, Skeletons, Clip_len, V, C
        kpt_ret = np.stack(kpt_ret, axis=1)
        # Skeletons, 6
        stinfo_old = np.stack(stinfos)

        min_ske, max_ske, all_skeletons = self.min_skeletons, self.max_skeletons, kpt_ret.shape[1]
        num_ske = np.clip(all_skeletons, min_ske, max_ske) if test_mode else max_ske
        keypoint = np.zeros((self.num_clips, num_ske) + kpt_ret.shape[2:], dtype=np.float32)
        stinfo = np.zeros((self.num_clips, num_ske, 6), dtype=np.float32)

        if test_mode:
            if all_skeletons < num_ske:
                keypoint[:, :all_skeletons] = kpt_ret
                stinfo[:, :all_skeletons] = stinfo_old
            elif all_skeletons > num_ske:
                stinfo_old = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                indices = self._get_test_clips(all_skeletons, num_ske)
                indices = indices.reshape((self.num_clips, num_ske))
                for i in range(self.num_clips):
                    keypoint[i] = kpt_ret[i, indices[i]]
                    stinfo[i] = stinfo_old[i, indices[i]]
            else:
                stinfo = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                keypoint = kpt_ret
        else:
            # only use max_ske
            if all_skeletons > num_ske:
                stinfo_old = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                indices = self._get_train_clips(all_skeletons, num_ske)
                indices = indices.reshape((self.num_clips, num_ske))
                for i in range(self.num_clips):
                    keypoint[i] = kpt_ret[i, indices[i]]
                    stinfo[i] = stinfo_old[i, indices[i]]
            else:
                keypoint[:, :all_skeletons] = kpt_ret
                stinfo[:, :all_skeletons] = stinfo_old

        results['keypoint'] = keypoint
        results['stinfo'] = stinfo
        results['name'] = results['frame_dir']
        return results


@PIPELINES.register_module()
class AVASSampling(UniformSampleFrames):

    def __init__(self,
                 num_clips=1,
                 clip_olen=60,
                 clip_len=30,
                 seed=255,
                 num_skeletons=20,
                 rel_offset=0.75):
        super().__init__(num_clips=num_clips, clip_len=clip_len, seed=seed)
        self.clip_olen = clip_olen
        nske = num_skeletons
        if isinstance(nske, int):
            assert nske >= 1
            nske = (nske, nske)
        else:
            assert 0 < nske[0] <= nske[1]
        self.min_skeletons, self.max_skeletons = nske
        self.rel_offset = rel_offset

    def __call__(self, results):
        assert 'data' in results
        data = results.pop('data')
        test_mode = results['test_mode']

        kpts, stinfos, labels, names = [], [], [], []

        for i, frame in enumerate(data):
            for ske in frame:
                keypoint, name, label = ske['keypoint'], ske['name'], ske['label']
                box = ske['box']
                box_score = ske.get('box_score', 1.0)

                if len(keypoint.shape) == 4:
                    assert keypoint.shape[0] == 1
                    keypoint = keypoint[0]
                # keypoint.shape == (T, V, C)
                stinfo = np.array(list(box) + [i / len(data), box_score], dtype=np.float32)
                stinfo = np.clip(stinfo, 0, 1)
                stinfos.append(stinfo)
                names.append(name)
                labels.append(label)

                total_frames = keypoint.shape[0]
                if total_frames > self.clip_olen:
                    offset = (total_frames - self.clip_olen) // 2
                    if not test_mode:
                        offset *= (1 - self.rel_offset) + np.random.rand() * self.rel_offset * 2
                        offset = int(offset)
                    keypoint = keypoint[offset:offset + self.clip_olen]
                    total_frames = self.clip_olen

                assert total_frames >= self.clip_len
                if test_mode:
                    frame_inds = self._get_test_clips(total_frames, self.clip_len)
                else:
                    frame_inds = self._get_train_clips(total_frames, self.clip_len)
                assert len(frame_inds) == self.clip_len * self.num_clips
                keypoint = keypoint[frame_inds].reshape((self.num_clips, self.clip_len) + (keypoint.shape[1:]))
                # keypoint.shape == (num_clips, T, V, C)
                kpts.append(keypoint)

        kpts = np.stack(kpts, axis=1)  # kpts.shape == (num_clips, num_skeletons, T, V, C)
        labels = np.stack(labels)  # labels.shape == (num_skeletons, num_classes)
        stinfo_old = np.stack(stinfos)  # stinfo_old.shape = (num_skeletons, 6)
        names = np.stack(names)

        min_ske, max_ske, all_skeletons = self.min_skeletons, self.max_skeletons, kpts.shape[1]
        num_ske = np.clip(all_skeletons, min_ske, max_ske) if test_mode else max_ske
        keypoint = np.zeros((self.num_clips, num_ske) + kpts.shape[2:], dtype=np.float32)
        label = np.zeros((self.num_clips, num_ske, labels.shape[-1]), dtype=np.float32)
        stinfo = np.zeros((self.num_clips, num_ske, 6), dtype=np.float32)

        if test_mode:
            if all_skeletons > num_ske:
                indices = self._get_test_clips(all_skeletons, num_ske).reshape((self.num_clips, num_ske))
                new_names = []
                for i in range(self.num_clips):
                    keypoint[i] = kpts[i, indices[i]]
                    label[i] = labels[indices[i]]
                    stinfo[i] = stinfo_old[indices[i]]
                    new_names.append(names[indices[i]])
                names = np.stack(new_names)
            elif all_skeletons < num_ske:
                keypoint[:, :all_skeletons] = kpts
                label[:, :all_skeletons] = labels
                stinfo[:, :all_skeletons] = stinfo_old
                names = np.concatenate([names, ['NA'] * (num_ske - all_skeletons)])
                names = np.stack([names] * self.num_clips)
            else:
                keypoint = kpts
                label = np.tile(labels[None], (self.num_clips, 1, 1))
                stinfo = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                names = np.stack([names] * self.num_clips)
        else:
            if all_skeletons > num_ske:
                indices = self._get_train_clips(all_skeletons, num_ske)
                indices = indices.reshape((self.num_clips, num_ske))
                new_names = []
                for i in range(self.num_clips):
                    keypoint[i] = kpts[i, indices[i]]
                    label[i] = labels[indices[i]]
                    stinfo[i] = stinfo_old[indices[i]]
                    new_names.append(names[indices[i]])
                names = np.stack(new_names)
            else:
                keypoint[:, :all_skeletons] = kpts
                label[:, :all_skeletons] = labels
                stinfo[:, :all_skeletons] = stinfo_old
                names = np.concatenate([names, ['NA'] * (num_ske - all_skeletons)])
                names = np.stack([names] * self.num_clips)

        results['keypoint'] = keypoint
        results['label'] = label
        results['stinfo'] = stinfo
        results['name'] = names
        return results
