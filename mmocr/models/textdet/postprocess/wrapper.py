import cv2
import numpy as np
import pyclipper
import torch
from numpy.linalg import norm
from shapely.geometry import Polygon
from skimage.morphology import skeletonize

from mmocr.core import points2boundary


def filter_instance(area, confidence, min_area, min_confidence):
    return bool(area < min_area or confidence < min_confidence)


def decode(
        decoding_type='pan',  # 'pan' or 'pse'
        **kwargs):
    if decoding_type == 'pan':
        return pan_decode(**kwargs)
    if decoding_type == 'pse':
        return pse_decode(**kwargs)
    if decoding_type == 'db':
        return db_decode(**kwargs)
    if decoding_type == 'textsnake':
        return textsnake_decode(**kwargs)

    raise NotImplementedError


def pan_decode(preds,
               text_repr_type='poly',
               min_text_confidence=0.5,
               min_kernel_confidence=0.5,
               min_text_avg_confidence=0.85,
               min_kernel_area=0,
               min_text_area=16):
    """Convert scores to quadrangles via post processing in PANet. This is
    partially adapted from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        preds (tensor): The head output tensor of size 6xHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_kernel_area (int): The minimal text kernel area.
        min_text_area (int): The minimal text instance region area.
    Returns:
        boundaries: (list[list[float]]): The instance boundary and its
            instance confidence list.
    """
    from .pan import assign_pixels, estimate_text_confidence, get_pixel_num
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()

    text_score = preds[0].astype(np.float32)
    text = preds[0] > min_text_confidence
    kernel = (preds[1] > min_kernel_confidence) * text
    embeddings = preds[2:].transpose((1, 2, 0))  # (h, w, 4)

    region_num, labels = cv2.connectedComponents(
        kernel.astype(np.uint8), connectivity=4)
    valid_kernel_inx = []
    region_pixel_num = get_pixel_num(labels, region_num)

    # from inx 1. 0: meaningless.
    for region_idx in range(1, region_num):
        if region_pixel_num[region_idx] < min_kernel_area:
            continue
        valid_kernel_inx.append(region_idx)

    # assign pixels to valid kernels
    assignment = assign_pixels(
        text.astype(np.uint8), embeddings, labels, region_num, 0.8)
    assignment = assignment.reshape(text.shape)

    boundaries = []

    # compute text avg confidence

    text_points = estimate_text_confidence(assignment, text_score, region_num)
    for text_inx, text_point in text_points.items():
        if text_inx not in valid_kernel_inx:
            continue
        text_confidence = text_point[0]
        text_point = text_point[2:]
        text_point = np.array(text_point, dtype=int).reshape(-1, 2)
        area = text_point.shape[0]

        if filter_instance(area, text_confidence, min_text_area,
                           min_text_avg_confidence):
            continue
        vertices_confidence = points2boundary(text_point, text_repr_type,
                                              text_confidence)
        if vertices_confidence is not None:
            boundaries.append(vertices_confidence)

    return boundaries


def pse_decode(preds,
               text_repr_type='poly',
               min_kernel_confidence=0.5,
               min_text_avg_confidence=0.85,
               min_kernel_area=0,
               min_text_area=16):
    """Decoding predictions of PSENet to instances. This is partially adapted
    from https://github.com/whai362/PSENet.

    Args:
        preds (tensor): The head output tensor of size nxHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_kernel_area (int): The minimal text kernel area.
        min_text_area (int): The minimal text instance region area.
    Returns:
        boundaries: (list[list[float]]): The instance boundary and its
            instance confidence list.
    """
    preds = torch.sigmoid(preds)  # text confidence

    score = preds[0, :, :]
    masks = preds > min_kernel_confidence
    text_mask = masks[0, :, :]
    kernel_masks = masks[0:, :, :] * text_mask

    score = score.data.cpu().numpy().astype(np.float32)  # to numpy

    kernel_masks = kernel_masks.data.cpu().numpy().astype(np.uint8)  # to numpy
    from .pse import pse

    region_num, labels = cv2.connectedComponents(
        kernel_masks[-1], connectivity=4)

    # labels = pse(kernel_masks, min_kernel_area)
    labels = pse(kernel_masks, min_kernel_area, labels, region_num)
    labels = np.array(labels)
    label_num = np.max(labels) + 1
    boundaries = []
    for i in range(1, label_num):
        points = np.array(np.where(labels == i)).transpose((1, 0))[:, ::-1]
        area = points.shape[0]
        score_instance = np.mean(score[labels == i])
        if filter_instance(area, score_instance, min_text_area,
                           min_text_avg_confidence):
            continue

        vertices_confidence = points2boundary(points, text_repr_type,
                                              score_instance)
        if vertices_confidence is not None:
            boundaries.append(vertices_confidence)

    return boundaries


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def db_decode(preds,
              text_repr_type='poly',
              mask_thr=0.3,
              min_text_score=0.3,
              min_text_width=5,
              unclip_ratio=1.5,
              max_candidates=1000):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        preds (Tensor): The head output tensor of size nxHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.

    Returns:
        boundaries: (list[list[float]]): The predicted text boundaries.
    """
    prob_map = preds[0, :, :]
    text_mask = prob_map > mask_thr

    score_map = prob_map.data.cpu().numpy().astype(np.float32)
    text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

    contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boundaries = []
    for i, poly in enumerate(contours):
        if i > max_candidates:
            break
        epsilon = 0.01 * cv2.arcLength(poly, True)
        approx = cv2.approxPolyDP(poly, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(score_map, points)
        if score < min_text_score:
            continue
        poly = unclip(points, unclip_ratio=unclip_ratio)
        if len(poly) == 0 or isinstance(poly[0], list):
            continue
        poly = poly.reshape(-1, 2)
        poly = points2boundary(poly, text_repr_type, score, min_text_width)
        if poly is not None:
            boundaries.append(poly)
    return boundaries


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask


def centralize(points_yx,
               normal_sin,
               normal_cos,
               radius,
               contour_mask,
               step_ratio=0.03):

    h, w = contour_mask.shape
    top_yx = bot_yx = points_yx
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool)
    step = step_ratio * radius * np.hstack([normal_sin, normal_cos])
    while np.any(step_flags):
        next_yx = np.array(top_yx + step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (0 <= next_y) & (next_y < h) & (0 < next_x) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                       np.clip(next_x, 0, w - 1)]
        top_yx = top_yx + step_flags.reshape((-1, 1)) * step
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool)
    while np.any(step_flags):
        next_yx = np.array(bot_yx - step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (0 <= next_y) & (next_y < h) & (0 < next_x) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                       np.clip(next_x, 0, w - 1)]
        bot_yx = bot_yx - step_flags.reshape((-1, 1)) * step
    centers = np.array((top_yx + bot_yx) * 0.5, dtype=np.int32)
    return centers


def merge_disks(disks, disk_overlap_thr):
    xy = disks[:, 0:2]
    radius = disks[:, 2]
    scores = disks[:, 3]
    order = scores.argsort()[::-1]

    merged_disks = []
    while order.size > 0:
        if order.size == 1:
            merged_disks.append(disks[order])
            break
        else:
            i = order[0]
            d = norm(xy[i] - xy[order[1:]], axis=1)
            ri = radius[i]
            r = radius[order[1:]]
            d_thr = (ri + r) * disk_overlap_thr

            merge_inds = np.where(d <= d_thr)[0] + 1
            if merge_inds.size > 0:
                merge_order = np.hstack([i, order[merge_inds]])
                merged_disks.append(np.mean(disks[merge_order], axis=0))
            else:
                merged_disks.append(disks[i])

            inds = np.where(d > d_thr)[0] + 1
            order = order[inds]
    merged_disks = np.vstack(merged_disks)

    return merged_disks


def textsnake_decode(preds,
                     text_repr_type='poly',
                     min_text_region_confidence=0.6,
                     min_center_region_confidence=0.2,
                     min_center_area=30,
                     disk_overlap_thr=0.03,
                     radius_shrink_ratio=1.03):
    """Decoding predictions of TextSnake to instances. This was partially
    adapted from https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        preds (tensor): The head output tensor of size 6xHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_region_confidence (float): The confidence threshold of text
            region in TextSnake.
        min_center_region_confidence (float): The confidence threshold of text
            center region in TextSnake.
        min_center_area (int): The minimal text center region area.
        disk_overlap_thr (float): The radius overlap threshold for merging
            disks.
        radius_shrink_ratio (float): The shrink ratio of ordered disks radii.

    Returns:
        boundaries (list[list[float]]): The instance boundary and its
            instance confidence list.
    """
    assert text_repr_type == 'poly'
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()

    pred_text_score = preds[0]
    pred_text_mask = pred_text_score > min_text_region_confidence
    pred_center_score = preds[1] * pred_text_score
    pred_center_mask = pred_center_score > min_center_region_confidence
    pred_sin = preds[2]
    pred_cos = preds[3]
    pred_radius = preds[4]
    mask_sz = pred_text_mask.shape

    scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
    pred_sin = pred_sin * scale
    pred_cos = pred_cos * scale

    pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
    center_contours, _ = cv2.findContours(pred_center_mask, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

    boundaries = []
    for contour in center_contours:
        if cv2.contourArea(contour) < min_center_area:
            continue
        instance_center_mask = np.zeros(mask_sz, dtype=np.uint8)
        cv2.drawContours(instance_center_mask, [contour], -1, 1, -1)
        skeleton = skeletonize(instance_center_mask)
        skeleton_yx = np.argwhere(skeleton > 0)
        y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
        cos = pred_cos[y, x].reshape((-1, 1))
        sin = pred_sin[y, x].reshape((-1, 1))
        radius = pred_radius[y, x].reshape((-1, 1))

        center_line_yx = centralize(skeleton_yx, cos, -sin, radius,
                                    instance_center_mask)
        y, x = center_line_yx[:, 0], center_line_yx[:, 1]
        radius = (pred_radius[y, x] * radius_shrink_ratio).reshape((-1, 1))
        score = pred_center_score[y, x].reshape((-1, 1))
        instance_disks = np.hstack([np.fliplr(center_line_yx), radius, score])
        instance_disks = merge_disks(instance_disks, disk_overlap_thr)

        instance_mask = np.zeros(mask_sz, dtype=np.uint8)
        for x, y, radius, score in instance_disks:
            if radius > 0:
                cv2.circle(instance_mask, (int(x), int(y)), int(radius), 1, -1)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        score = np.sum(instance_mask * pred_text_score) / (
            np.sum(instance_mask) + 1e-8)
        if len(contours) > 0:
            boundary = contours[0].flatten().tolist()
            boundaries.append(boundary + [score])

    return boundaries
