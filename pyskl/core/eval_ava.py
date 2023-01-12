# Copyright (c) OpenMMLab. All rights reserved.
# This piece of code is directly adapted from ActivityNet official repo
# https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py.
# Some unused codes are removed.
import multiprocessing
import numpy as np
import time
from collections import defaultdict

from ..smp import mrlines
from .ava_utils import metrics, np_box


def print_time(message, start):
    """Print processing time."""
    print('==> %g seconds to %s' % (time.time() - start, message), flush=True)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{int(timestamp):04d}'


def read_csv(csv_file, class_whitelist=None):
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)

    if isinstance(csv_file, list):
        lines = csv_file
    else:
        lines = mrlines(csv_file)
    lines = [x for x in lines if not x.startswith('NA,')]
    lines = [x.split(',') for x in lines]

    for row in lines:
        assert len(row) in [7, 8], 'Wrong number of columns: ' + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue

        score = 1.0
        if len(row) == 8:
            score = float(row[7])

        entries[image_key].append((score, action_id, y1, x1, y2, x2))

    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        boxes[image_key] = [x[2:] for x in entry]
        labels[image_key] = [x[1] for x in entry]
        scores[image_key] = [x[0] for x in entry]

    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g.
        "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    excluded = set()
    lines = mrlines(exclusions_file)
    for row in lines:
        row = row.split(',')
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the
        object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


def get_overlaps_and_scores_box_mode(detected_boxes, detected_scores,
                                     groundtruth_boxes):

    detected_boxlist = np_box.BoxList(detected_boxes)
    detected_boxlist.add_field('scores', detected_scores)
    gt_non_group_of_boxlist = np_box.BoxList(groundtruth_boxes)

    iou = np_box.iou(detected_boxlist.get(), gt_non_group_of_boxlist.get())
    scores = detected_boxlist.get_field('scores')
    num_boxes = detected_boxlist.num_boxes()
    return iou, scores, num_boxes


def tpfp_single(tup, threshold=0.5):
    gt_bboxes, gt_labels, bboxes, labels, scores = tup
    ret_scores, ret_tp_fp_labels = dict(), dict()
    all_labels = list(set(labels))
    for label in all_labels:
        gt_bbox = np.array(
            [x for x, y in zip(gt_bboxes, gt_labels) if y == label],
            dtype=np.float32).reshape(-1, 4)
        bbox = np.array([x for x, y in zip(bboxes, labels) if y == label],
                        dtype=np.float32).reshape(-1, 4)
        score = np.array([x for x, y in zip(scores, labels) if y == label],
                         dtype=np.float32).reshape(-1)
        iou, score, num_boxes = get_overlaps_and_scores_box_mode(
            bbox, score, gt_bbox)
        if gt_bbox.size == 0:
            ret_scores[label] = score
            ret_tp_fp_labels[label] = np.zeros(num_boxes, dtype=bool)
            continue
        tp_fp_labels = np.zeros(num_boxes, dtype=bool)
        if iou.shape[1] > 0:
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(num_boxes):
                gt_id = max_overlap_gt_ids[i]
                if iou[i, gt_id] >= threshold:
                    if not is_gt_box_detected[gt_id]:
                        tp_fp_labels[i] = True
                        is_gt_box_detected[gt_id] = True
        ret_scores[label], ret_tp_fp_labels[label] = score, tp_fp_labels
    return ret_scores, ret_tp_fp_labels


# Seems there is at most 100 detections for each image
def eval_ava(result_file,
             label_file='data/ava/label_map.txt',
             ann_file='data/ava/ava_val_v2.2.csv',
             exclude_file='data/ava/ava_val_excluded_timestamps_v2.2.csv',
             ignore_empty_frames=True):
    """Perform ava evaluation."""

    start = time.time()
    categories, class_whitelist = read_labelmap(open(label_file))

    # loading gt, do not need gt score
    gt_bboxes, gt_labels, _ = read_csv(open(ann_file), class_whitelist)
    print_time('Reading GT results', start)

    if exclude_file is not None:
        excluded_keys = read_exclusions(open(exclude_file))
    else:
        excluded_keys = list()

    start = time.time()
    boxes, labels, scores = read_csv(open(result_file), class_whitelist)
    print_time('Reading Detection results', start)

    start = time.time()
    all_gt_labels = np.concatenate(list(gt_labels.values()))
    gt_count = {k: np.sum(all_gt_labels == k) for k in class_whitelist}

    pool = multiprocessing.Pool(32)
    if ignore_empty_frames:
        tups = [(gt_bboxes[k], gt_labels[k], boxes[k], labels[k], scores[k])
                for k in gt_bboxes if k not in excluded_keys]
    else:
        tups = [(gt_bboxes.get(k, np.zeros((0, 4), dtype=np.float32)),
                 gt_labels.get(k, []), boxes[k], labels[k], scores[k])
                for k in boxes if k not in excluded_keys]
    rets = pool.map(tpfp_single, tups)

    print_time('Calculating TP/FP', start)

    start = time.time()
    scores, tpfps = defaultdict(list), defaultdict(list)
    for score, tpfp in rets:
        for k in score:
            scores[k].append(score[k])
            tpfps[k].append(tpfp[k])

    cls_AP = []
    for k in scores:
        scores[k] = np.concatenate(scores[k])
        tpfps[k] = np.concatenate(tpfps[k])
        precision, recall = metrics.compute_precision_recall(
            scores[k], tpfps[k], gt_count[k])
        ap = metrics.compute_average_precision(precision, recall)
        class_name = [x['name'] for x in categories if x['id'] == k]
        assert len(class_name) == 1
        class_name = class_name[0]
        cls_AP.append((k, class_name, ap))

    print_time('Run Evaluator', start)

    print('Per-class results: ', flush=True)
    for k, class_name, ap in cls_AP:
        print(f'Index: {k}, Action: {class_name}: AP: {ap:.4f};', flush=True)

    overall = np.nanmean([x[2] for x in cls_AP])
    person_movement = np.nanmean([x[2] for x in cls_AP if x[0] <= 14])
    object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
    person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])

    print('Overall Results: ', flush=True)
    print(f'Overall mAP: {overall:.4f}', flush=True)
    print(f'Person Movement mAP: {person_movement:.4f}', flush=True)
    print(f'Object Manipulation mAP: {object_manipulation:.4f}', flush=True)
    print(f'Person Interaction mAP: {person_interaction:.4f}', flush=True)

    results = {}
    results['overall'] = overall
    results['person_movement'] = person_movement
    results['object_manipulation'] = object_manipulation
    results['person_interaction'] = person_interaction
    for k, class_name, ap in cls_AP:
        results[class_name] = ap

    return results
