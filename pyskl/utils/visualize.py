import copy as cp
import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib.animation import FuncAnimation
from mmcv import load
from tqdm import tqdm

from pyskl.smp import h2r

skeleton_map = dict(
    coco=[
        (0, 1, 'f'), (0, 2, 'f'), (1, 3, 'f'), (2, 4, 'f'), (0, 5, 't'), (0, 6, 't'), (5, 7, 'ru'), (6, 8, 'lu'),
        (7, 9, 'ru'), (8, 10, 'lu'), (5, 11, 't'), (6, 12, 't'), (11, 13, 'ld'), (12, 14, 'rd'), (13, 15, 'ld'),
        (14, 16, 'rd')
    ],
    onehand=[
        (0, 1, 't'), (1, 2, 'f'), (2, 3, 'f'), (3, 4, 'f'), (0, 5, 't'), (5, 6, 'lu'), (6, 7, 'lu'), (7, 8, 'lu'),
        (0, 9, 't'), (9, 10, 'ru'), (10, 11, 'ru'), (11, 12, 'ru'), (0, 13, 't'), (13, 14, 'ld'), (14, 15, 'ld'),
        (15, 16, 'ld'), (0, 17, 't'), (17, 18, 'rd'), (18, 19, 'rd'), (19, 20, 'rd')
    ],
    interhand=[
        (20, 0, 't'), (0, 1, 'f'), (1, 2, 'f'), (2, 3, 'f'), (20, 4, 't'), (4, 5, 'lu'), (5, 6, 'lu'), (6, 7, 'lu'),
        (20, 8, 't'), (8, 9, 'ru'), (9, 10, 'ru'), (10, 11, 'ru'), (20, 12, 't'), (12, 13, 'ld'), (13, 14, 'ld'),
        (14, 15, 'ld'), (20, 16, 't'), (16, 17, 'rd'), (17, 18, 'rd'), (18, 19, 'rd')
    ])


def load_frames(vid):
    vid = cv2.VideoCapture(vid)
    images = []
    success, image = vid.read()
    while success:
        images.append(np.ascontiguousarray(image[..., ::-1]))
        success, image = vid.read()
    return images


def Vis3DPose(item, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), with_grid=False):
    kp = item['keypoint'].copy()
    colors = ('#3498db', '#000000', '#e74c3c')  # l, m, r

    assert layout == 'nturgb+d'
    if layout == 'nturgb+d':
        num_joint = 25
        kinematic_tree = [
            [1, 2, 21, 3, 4],
            [21, 9, 10, 11, 12, 25], [12, 24],
            [21, 5, 6, 7, 8, 23], [8, 22],
            [1, 17, 18, 19, 20],
            [1, 13, 14, 15, 16]
        ]
        kinematic_tree = [[x - 1 for x in lst] for lst in kinematic_tree]
        colors = ['black', 'blue', 'blue', 'red', 'red', 'darkblue', 'darkred']

    assert len(kp.shape) == 4 and kp.shape[3] == 3 and kp.shape[2] == num_joint
    x, y, z = kp[..., 0], kp[..., 1], kp[..., 2]
    min_x, max_x = min(x[x != 0]), max(x[x != 0])
    min_y, max_y = min(y[y != 0]), max(y[y != 0])
    min_z, max_z = min(z[z != 0]), max(z[z != 0])

    max_axis = max(max_x - min_x, max_y - min_y, max_z - min_z)
    mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2

    min_x, max_x = mid_x - max_axis / 2, mid_x + max_axis / 2
    min_y, max_y = mid_y - max_axis / 2, mid_y + max_axis / 2
    min_z, max_z = mid_z - max_axis / 2, mid_z + max_axis / 2

    fig = plt.figure(figsize=fig_size)
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([min_x, max_x])
    ax.set_ylim3d([min_y, max_y])
    ax.set_zlim3d([min_z, max_z])
    ax.view_init(*angle)
    fig.suptitle(item.get('frame_dir', 'demo'), fontsize=20)
    save_path = item.get('frame_dir', 'tmp').split('/')[-1] + '.mp4'

    def update(t):
        ax.lines = []
        ax.view_init(*angle)
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 2.0
            for j in range(kp.shape[0]):
                ax.plot3D(kp[j, t, chain, 0], kp[j, t, chain, 1], kp[j, t, chain, 2], linewidth=linewidth, color=color)
        if not with_grid:
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=kp.shape[1], interval=0, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
    video = mpy.VideoFileClip(save_path)
    return video


def Vis2DPose(item, thre=0.2, out_shape=(540, 960), layout='coco', fps=24, video=None):
    tx = cp.deepcopy(item)
    item = tx

    if isinstance(item, str):
        item = load(item)

    kp = item['keypoint']
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        kp = np.concatenate([kp, kpscore[..., None]], -1)

    total_frames = None
    if len(kp.shape) == 4:
        total_frames = item.get('total_frames', kp.shape[1])
        assert total_frames == kp.shape[1]
    else:
        assert len(kp.shape) == 3
        assert 'frame_inds' in item
        frame_inds = item['frame_inds']
        min_ind, max_ind = min(frame_inds), max(frame_inds)
        assert min_ind >= 1
        total_frames = max_ind

    if video is None:
        assert out_shape is not None
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 for i in range(total_frames)]
    else:
        frames = load_frames(video)
        if out_shape is None:
            out_shape = frames[0].shape[:2]

        frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]

    assert kp.shape[-1] == 3
    img_shape = item.get('img_shape', out_shape)
    kp[..., 0] *= out_shape[1] / img_shape[1]
    kp[..., 1] *= out_shape[0] / img_shape[0]

    # Note that the shape of kp can be M, T, V, C or X, V, C.
    if len(kp.shape) == 4:
        kps = [kp[:, i] for i in range(total_frames)]
    else:
        kps = []
        for i in range(total_frames):
            kps.append(kp[frame_inds == (i + 1)])

    edges = skeleton_map[layout]

    color_map = {
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))}

    for i in tqdm(range(total_frames)):
        kp = kps[i]
        for m in range(kp.shape[0]):
            ske = kp[m]
            for e in edges:
                st, ed, co = e
                co_tup = color_map[co]
                j1, j2 = ske[st], ske[ed]
                j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                conf = min(j1[2], j2[2])
                if conf > thre:
                    color = [x + (y - x) * (conf - thre) / 0.8 for x, y in zip(co_tup[0], co_tup[1])]
                    color = tuple([int(x) for x in color])
                    frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
    return mpy.ImageSequenceClip(frames, fps=fps)


def VisLayout(item, out_shape=(540, 960), fps=12, video=None):
    plate = 'f15152-3a2e39-1e555c-f4d8cd-edb183'.split('-')
    plate = [h2r(x) for x in plate]

    if video is None:
        assert out_shape is not None
        total_frames = max(item['frame_inds'])
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 for i in range(total_frames)]
    else:
        frames = load_frames(video)
        if out_shape is None:
            out_shape = frames[0].shape[:2]
        else:
            frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]
    frames = [np.ascontiguousarray(f, np.uint8) for f in frames]

    if isinstance(item, str):
        item = load(item)

    bbox = item['bbox']
    total_box = bbox.shape[0]
    assert len(bbox.shape) == 2

    frame_inds = item['frame_inds']
    min_inds = min(frame_inds)
    assert min_inds == 1
    total_frames = max(frame_inds)

    category = [None for i in range(total_box)]
    if 'category' in item:
        category = item['category']
    num_category = len(set(category))
    assert num_category <= len(plate), (set(category), len(plate))

    h, w = item.get('img_shape', out_shape)
    outh, outw = out_shape
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * outw / w, 0, w - 1)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * outh / h, 0, h - 1)
    bbox = bbox.astype(int)

    color_map = dict()
    if 'hand' in category:
        color_map['hand'] = 0
    # return frames

    for i in range(total_frames):
        box = [bbox[j] for j in range(total_box) if frame_inds[j] == i + 1]
        cate = [category[j] for j in range(total_box) if frame_inds[j] == i + 1]
        for m in range(len(box)):
            x1, y1, x2, y2 = box[m]
            c = cate[m]
            if c is None:
                color = plate[0]
            elif c in color_map:
                color = plate[color_map[c]]
            else:
                color_idx = len(color_map)
                color_map[c] = color_idx
                color = plate[color_map[c]]

            frames[i] = cv2.rectangle(frames[i], (x1, y1), (x2, y2), color, thickness=2)

    return mpy.ImageSequenceClip(frames, fps=fps)
