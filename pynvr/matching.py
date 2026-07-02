import numpy as np

def iou(b1, b2):
    """
    Compute IoU between two boxes in tlbr format.
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def iou_distance(tracks, detections):
    """
    Compute IoU distance matrix between track boxes and detection boxes.
    """
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost[i, j] = 1 - iou(t.tlbr, d.tlbr)
    return cost


def linear_assignment(cost_matrix, thresh):
    """
    Hungarian matching with threshold.
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches, unmatched_a, unmatched_b = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append((r, c))

    for i in range(cost_matrix.shape[0]):
        if i not in row_ind:
            unmatched_a.append(i)

    for j in range(cost_matrix.shape[1]):
        if j not in col_ind:
            unmatched_b.append(j)

    return matches, unmatched_a, unmatched_b
