import cv2
import decord
import io
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
from mmcv import load
from tqdm import tqdm


class Vis3DPose:

    def __init__(self, item, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), dpi=80):
        kp = item['keypoint']
        self.kp = kp
        assert self.kp.shape[-1] == 3
        self.layout = layout
        self.fps = fps
        self.angle = angle  # For 3D data only
        self.colors = ('#3498db', '#000000', '#e74c3c')  # l, m, r
        self.fig_size = fig_size
        self.dpi = dpi

        assert layout == 'nturgb+d'
        if self.layout == 'nturgb+d':
            self.num_joint = 25
            self.links = np.array([
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)], dtype=np.int) - 1
            self.left = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23], dtype=np.int) - 1
            self.right = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25], dtype=np.int) - 1
            self.num_link = len(self.links)
        self.limb_tag = [1] * self.num_link

        for i, link in enumerate(self.links):
            if link[0] in self.left or link[1] in self.left:
                self.limb_tag[i] = 0
            elif link[0] in self.right or link[1] in self.right:
                self.limb_tag[i] = 2

        assert len(kp.shape) == 4 and kp.shape[3] == 3 and kp.shape[2] == self.num_joint
        x, y, z = kp[..., 0], kp[..., 1], kp[..., 2]

        min_x, max_x = min(x[x != 0]), max(x[x != 0])
        min_y, max_y = min(y[y != 0]), max(y[y != 0])
        min_z, max_z = min(z[z != 0]), max(z[z != 0])
        max_axis = max(max_x - min_x, max_y - min_y, max_z - min_z)
        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
        self.min_x, self.max_x = mid_x - max_axis / 2, mid_x + max_axis / 2
        self.min_y, self.max_y = mid_y - max_axis / 2, mid_y + max_axis / 2
        self.min_z, self.max_z = mid_z - max_axis / 2, mid_z + max_axis / 2

        self.images = []

    def get_img(self, dpi=80):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        return cv2.imdecode(img, -1)

    def vis(self):
        self.images = []
        plt.figure(figsize=self.fig_size)
        for t in range(self.kp.shape[1]):
            ax = plt.gca(projection='3d')
            ax.set_xlim3d([self.min_x, self.max_x])
            ax.set_ylim3d([self.min_y, self.max_y])
            ax.set_zlim3d([self.min_z, self.max_z])
            ax.view_init(*self.angle)
            ax.set_aspect('auto')
            for i in range(self.num_link):
                for m in range(self.kp.shape[0]):
                    link = self.links[i]
                    color = self.colors[self.limb_tag[i]]
                    j1, j2 = self.kp[m, t, link[0]], self.kp[m, t, link[1]]
                    if not ((np.allclose(j1, 0) or np.allclose(j2, 0)) and link[0] != 1 and link[1] != 1):
                        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], lw=1, c=color)
            self.images.append(self.get_img(dpi=self.dpi))
            ax.cla()
        return mpy.ImageSequenceClip(self.images, fps=self.fps)


def Vis2DPose(item, thre=0.2, out_shape=(540, 960), layout='coco', fps=24, video=None):
    if isinstance(item, str):
        item = load(item)

    assert layout == 'coco'

    kp = item['keypoint']
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        kp = np.concatenate([kp, kpscore[..., None]], -1)

    assert kp.shape[-1] == 3
    img_shape = item.get('img_shape', out_shape)
    kp[..., 0] *= out_shape[1] / img_shape[1]
    kp[..., 1] *= out_shape[0] / img_shape[0]

    total_frames = item.get('total_frames', kp.shape[1])
    assert total_frames == kp.shape[1]

    if video is None:
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 for i in range(total_frames)]
    else:
        vid = decord.VideoReader(video)
        frames = [x.asnumpy() for x in vid]
        frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]
        if len(frames) != total_frames:
            frames = [frames[int(i / total_frames * len(frames))] for i in range(total_frames)]

    if layout == 'coco':
        edges = [
            (0, 1, 'f'), (0, 2, 'f'), (1, 3, 'f'), (2, 4, 'f'), (0, 5, 't'), (0, 6, 't'),
            (5, 7, 'ru'), (6, 8, 'lu'), (7, 9, 'ru'), (8, 10, 'lu'), (5, 11, 't'), (6, 12, 't'),
            (11, 13, 'ld'), (12, 14, 'rd'), (13, 15, 'ld'), (14, 16, 'rd')
        ]
    color_map = {
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))}

    for i in tqdm(range(total_frames)):
        for m in range(kp.shape[0]):
            ske = kp[m, i]
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
