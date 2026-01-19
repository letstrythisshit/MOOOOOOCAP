from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class Pose:
    keypoints: np.ndarray  # shape (num_kpts, 3) -> x, y, score
    score: float


LIMB_SEQ = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 16),
    (5, 17),
]

PAF_IDX = [
    (12, 13),
    (20, 21),
    (14, 15),
    (16, 17),
    (22, 23),
    (24, 25),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (28, 29),
    (30, 31),
    (34, 35),
    (32, 33),
    (36, 37),
    (18, 19),
    (26, 27),
]


def _extract_keypoints(heatmaps: np.ndarray, threshold: float) -> List[List[Tuple[int, int, float]]]:
    all_keypoints: List[List[Tuple[int, int, float]]] = []
    for kpt_idx in range(heatmaps.shape[0]):
        heatmap = heatmaps[kpt_idx]
        max_f = maximum_filter(heatmap, size=3)
        peaks = (heatmap == max_f) & (heatmap > threshold)
        ys, xs = np.where(peaks)
        keypoints = [(int(x), int(y), float(heatmap[y, x])) for x, y in zip(xs, ys)]
        all_keypoints.append(keypoints)
    return all_keypoints


def _score_pairs(
    cand_a: List[Tuple[int, int, float]],
    cand_b: List[Tuple[int, int, float]],
    paf_x: np.ndarray,
    paf_y: np.ndarray,
    paf_score_th: float,
    num_inter: int,
) -> List[Tuple[int, int, float]]:
    if not cand_a or not cand_b:
        return []

    connections: List[Tuple[int, int, float]] = []
    for i, a in enumerate(cand_a):
        for j, b in enumerate(cand_b):
            ax, ay, _ = a
            bx, by, _ = b
            dx = bx - ax
            dy = by - ay
            norm = np.hypot(dx, dy)
            if norm < 1e-6:
                continue
            vx = dx / norm
            vy = dy / norm
            xs = np.linspace(ax, bx, num=num_inter).astype(int)
            ys = np.linspace(ay, by, num=num_inter).astype(int)
            paf_scores = paf_x[ys, xs] * vx + paf_y[ys, xs] * vy
            if (paf_scores > paf_score_th).sum() / num_inter > 0.7:
                score = paf_scores.mean() + cand_a[i][2] + cand_b[j][2]
                connections.append((i, j, float(score)))
    connections.sort(key=lambda x: x[2], reverse=True)
    return connections


def decode_poses(
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    threshold: float = 0.1,
    paf_score_th: float = 0.05,
    num_inter: int = 10,
) -> List[Pose]:
    keypoints = _extract_keypoints(heatmaps, threshold)
    connections_all: List[List[Tuple[int, int, float]]] = []
    for limb_idx, (a, b) in enumerate(LIMB_SEQ):
        paf_x = pafs[PAF_IDX[limb_idx][0]]
        paf_y = pafs[PAF_IDX[limb_idx][1]]
        connections = _score_pairs(keypoints[a], keypoints[b], paf_x, paf_y, paf_score_th, num_inter)
        connections_all.append(connections)

    subset = -1 * np.ones((0, heatmaps.shape[0] + 2))
    for limb_idx, (a, b) in enumerate(LIMB_SEQ):
        connections = connections_all[limb_idx]
        if not connections:
            continue
        for i, j, score in connections:
            kp_a = keypoints[a][i]
            kp_b = keypoints[b][j]
            found = np.where((subset[:, a] == i) | (subset[:, b] == j))[0]
            if len(found) == 0:
                row = -1 * np.ones(heatmaps.shape[0] + 2)
                row[a] = i
                row[b] = j
                row[-1] = 2
                row[-2] = kp_a[2] + kp_b[2] + score
                subset = np.vstack([subset, row])
            elif len(found) == 1:
                row = subset[found[0]]
                if row[b] == -1:
                    row[b] = j
                    row[-1] += 1
                    row[-2] += kp_b[2] + score
            else:
                first, second = found[:2]
                row1 = subset[first]
                row2 = subset[second]
                if np.any((row1 >= 0) & (row2 >= 0)):
                    continue
                subset[first][:-2] = np.where(row1[:-2] >= 0, row1[:-2], row2[:-2])
                subset[first][-2:] += row2[-2:]
                subset = np.delete(subset, second, axis=0)

    poses: List[Pose] = []
    for row in subset:
        if row[-1] < 4 or row[-2] / row[-1] < 0.2:
            continue
        kpts = np.zeros((heatmaps.shape[0], 3), dtype=np.float32)
        for idx in range(heatmaps.shape[0]):
            if row[idx] < 0:
                continue
            x, y, score = keypoints[idx][int(row[idx])]
            kpts[idx] = (x, y, score)
        poses.append(Pose(kpts, float(row[-2])))
    return poses
