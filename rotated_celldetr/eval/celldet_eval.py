# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import cv2

import scipy
from scipy.optimize import linear_sum_assignment

from ..util.distributed import is_dist_avail_and_initialized, all_gather

class BaseCellMetric(object):
    def __init__(self, num_classes : int, 
                       thresholds : Union[int, List[int]], 
                       class_names : Optional[List[str]] = None,
                       *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
        self.class_names = class_names if class_names is not None else [str(i) for i in range(1, num_classes+1)]
        
        # predictions and targets
        self.preds = []
        self.targets = []

    def synchronize_between_processes(self):
        # synchronize the preds and targets
        dist.barrier()
        all_preds   = all_gather(self.preds)
        all_targets = all_gather(self.targets)
        # List of List of Dicts of Tensors to List of Dicts of Tensors
        self.preds   = list(itertools.chain(*all_preds))
        self.targets = list(itertools.chain(*all_targets))
    
    def reset(self):
        self.preds = []
        self.targets = []        

    def update(self, preds: List[Dict[str, torch.Tensor]], target: List[Dict[str, torch.Tensor]]) -> None:
        self.preds.extend(preds)
        self.targets.extend(target)

    def compute(self):
        # synchronize between processes
        self.synchronize_between_processes()
        # get the prediction and target values
        values = self._get_values()
        # compute the metric
        return self._compute(*values)
    
    def _get_values(self):
        raise NotImplementedError
    
    def _compute(self):
        raise NotImplementedError
    
    def calculate_rotated_iou(self, pred_moments, true_moments):
        pred_bboxes = [extract_bounding_box_from_moments(moment) for moment in pred_moments]
        true_bboxes = [extract_bounding_box_from_moments(moment) for moment in true_moments]
        ious = []
        for pred_bbox, true_bbox in zip(pred_bboxes, true_bboxes):
            iou = rotated_iou(pred_bbox, true_bbox)
            if not np.isnan(iou):
                ious.append(iou)
        if len(ious) == 0:
            return 0.0
        return np.mean(ious)
    
def extract_bounding_box_from_moments(moment):
    centroid_x, centroid_y, mu11, mu20, mu02 = moment
    theta_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta_rad)
    #Nan check
    a = np.sqrt(np.maximum(0, 2 * (mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4 * mu11**2))))
    b = np.sqrt(np.maximum(0, 2 * (mu20 + mu02 - np.sqrt((mu20 - mu02)**2 + 4 * mu11**2))))
    return ((float(centroid_x), float(centroid_y)), (float(a), float(b)), float(theta_deg))

def rotated_iou(bbox1, bbox2):
    rect1 = ((bbox1[0][0], bbox1[0][1]), (bbox1[1][0], bbox1[1][1]), bbox1[2])
    rect2 = ((bbox2[0][0], bbox2[0][1]), (bbox2[1][0], bbox2[1][1]), bbox2[2])
    intersection_area = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    if intersection_area is None or len(intersection_area) == 0:
        return 0.0
    intersection_area = cv2.contourArea(intersection_area)
    area1 = bbox1[1][0] * bbox1[1][1]
    area2 = bbox2[1][0] * bbox2[1][1]
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

class CellDetectionMetric(BaseCellMetric):
    def __init__(self, num_classes : int, 
                       thresholds : Union[int, List[int]], 
                       class_names : Optional[List[str]] = None,
                       *args, **kwargs):
        super().__init__(num_classes, thresholds, class_names, *args, **kwargs)

    def _get_values(self):
        # obtain predictions and targets
        true_moments  = [t["moments"].cpu().numpy() for t in self.targets]
        true_labels = [t["labels"].cpu().numpy() for t in self.targets]
        pred_moments  = [p["moments"].cpu().numpy() for p in self.preds]
        pred_labels = [p["labels"].cpu().numpy() for p in self.preds]
        pred_scores = [p["scores"].cpu().numpy() for p in self.preds]

        return true_moments, true_labels, pred_moments, pred_labels, pred_scores

    def _compute(self, true_moments, true_labels, pred_moments, pred_labels, pred_scores):
        # metrics
        all_metrics = dict()
        # compute metrics at different thresholds
        for threshold in self.thresholds:
            # detection scores
            paired_all = []  # unique matched index pair
            unpaired_true_all = (
                []
            )  # the index must exist in `true_inst_type_all` and unique
            unpaired_pred_all = (
                []
            )  # the index must exist in `pred_inst_type_all` and unique
            true_inst_type_all = []  # each index is 1 independent data point
            pred_inst_type_all = []  # each index is 1 independent data point
            true_inst_moment = []
            pred_inst_moment = []

            # for detections scores
            true_idx_offset = 0
            pred_idx_offset = 0

            # for each image
            for i in range(len(true_moments)):
                # get the mask accordint to the threshold
                mask = pred_scores[i] >= threshold

                # get the true and pred centroids and labels
                true_moments_i  = true_moments[i]
                true_labels_i = true_labels[i]
                pred_moments_i = pred_moments[i][mask]
                pred_labels_i = pred_labels[i][mask]

                # no predictions / no ground truth
                if true_moments_i.shape[0] == 0:
                    true_moments_i = np.array([[0, 0, 0, 0, 0]])
                    true_labels_i = np.array([0])
                if pred_moments_i.shape[0] == 0:
                    pred_moments_i = np.array([[0, 0, 0, 0, 0]])
                    pred_labels_i = np.array([0])

                # pairing
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_moments_i[:, :2], pred_moments_i[:, :2], 12)  # only use centroids for pairing
                
                # accumulating
                true_idx_offset = (
                    true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
                )
                true_inst_type_all.append(true_labels_i)
                pred_inst_type_all.append(pred_labels_i)
                true_inst_moment.append(true_moments_i)
                pred_inst_moment.append(pred_moments_i)

                # increment the pairing index statistic
                if paired.shape[0] != 0:  # ! sanity
                    paired[:, 0] += true_idx_offset
                    paired[:, 1] += pred_idx_offset
                    paired_all.append(paired)

                unpaired_true += true_idx_offset
                unpaired_pred += pred_idx_offset
                unpaired_true_all.append(unpaired_true)
                unpaired_pred_all.append(unpaired_pred)
            
            paired_all = np.concatenate(paired_all, axis=0) if len(paired_all) != 0 else np.empty((0,2), dtype=np.int64)
            unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
            unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
            true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
            pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)
            true_inst_moment = np.concatenate(true_inst_moment, axis=0)
            pred_inst_moment = np.concatenate(pred_inst_moment, axis=0)
            paired_true_type = true_inst_type_all[paired_all[:, 0]]
            paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
            unpaired_true_type = true_inst_type_all[unpaired_true_all]
            unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

            # compute the detection scores
            f1_d, prec_d, rec_d = cell_detection_scores(
                paired_true=paired_true_type,
                paired_pred=paired_pred_type,
                unpaired_true=unpaired_true_type,
                unpaired_pred=unpaired_pred_type)
            nuclei_metrics = {
                "detection": {
                    "f1": f1_d,
                    "prec": prec_d,
                    "rec": rec_d,
                },
            }
            
            iou_scores = self.calculate_rotated_iou(
                [pred_inst_moment[j] for j in paired_all[:, 1]], 
                [true_inst_moment[j] for j in paired_all[:, 0]]
            )
            nuclei_metrics["iou"] = iou_scores
            
            # compute the classification scores
            if self.num_classes > 1: # if num_classes is 1, only detection scenario
                for nuc_type in range(1, self.num_classes+1):
                    f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                    paired_true_type,
                    paired_pred_type,
                    unpaired_true_type,
                    unpaired_pred_type,
                    nuc_type,
                    )
                    nuclei_metrics[ self.class_names[nuc_type-1] ] = {
                        "f1": f1_cell,
                        "prec": prec_cell,
                        "rec": rec_cell,
                    }
            
            all_metrics["th"+str(threshold).replace(".","")] = nuclei_metrics
        # print free GPU memory
        return all_metrics

def pair_coordinates(setA: np.ndarray, setB: np.ndarray, radius: float):
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)

    return pairing, unpairedA, unpairedB

    
def cell_detection_scores(
    paired_true, paired_pred, unpaired_true, unpaired_pred, w: List = [1, 1]
):
    tp_d = paired_pred.shape[0]
    fp_d = unpaired_pred.shape[0]
    fn_d = unpaired_true.shape[0]

    # tp_tn_dt = (paired_pred == paired_true).sum()
    # fp_fn_dt = (paired_pred != paired_true).sum()
    prec_d = tp_d / (tp_d + fp_d)
    rec_d = tp_d / (tp_d + fn_d)

    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    return f1_d, prec_d, rec_d

def cell_type_detection_scores(
    paired_true,
    paired_pred,
    unpaired_true,
    unpaired_pred,
    type_id,
    w: List = [2, 2, 1, 1],
    exhaustive: bool = True,
    eps = 1e-6):
    type_samples = (paired_true == type_id) | (paired_pred == type_id)

    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()  #
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d + eps)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d + eps)

    f1_type = (2 * (tp_dt + tn_dt)) / (
        2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d + eps
    )
    return f1_type, prec_type, rec_type