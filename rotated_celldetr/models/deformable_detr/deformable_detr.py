# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from ...util import box_ops
from ...util import moment_ops
from ...util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, interpolate, inverse_sigmoid)
from ...util.distributed import get_world_size, is_dist_avail_and_initialized
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
import copy

from ...util.constants import MOMENT_MIN_VALUES, MOMENT_MAX_VALUES


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.moment_embed = MLP(hidden_dim, hidden_dim, 5, 3) #Modified to learn image moments!
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # Initialize moment prediction heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.moment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.moment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and moment_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.moment_embed = _get_clones(self.moment_embed, num_pred)
            nn.init.constant_(self.moment_embed[0].layers[-1].bias.data[2:], 0.0) #-2.0 (original implementation) or 0.0 for initial moments? //TODO
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.moment_embed = self.moment_embed
        else:
            nn.init.constant_(self.moment_embed.layers[-1].bias.data[2:], 0.0) # -2.0 Should this change? Investigate further //TODO
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.moment_embed = nn.ModuleList([self.moment_embed for _ in range(num_pred)])
            self.transformer.decoder.moment_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for moment_embed in self.moment_embed:
                nn.init.constant_(moment_embed.layers[-1].bias.data[2:], 0.0) #Should this change? Investigate further //TODO

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # Handle multi-scale feature maps. 
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        # Forward pass through the transformer
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_moments = []
        # Iterative refinement
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            # Predict moment deltas
            tmp = self.moment_embed[lvl](hs[lvl])
            if reference.shape[-1] == 5:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_moment = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_moments.append(outputs_moment)
            
        outputs_class = torch.stack(outputs_classes)
        outputs_moment = torch.stack(outputs_moments)

        out = {'pred_logits': outputs_class[-1], 'pred_moments': outputs_moment[-1]} #MOD!
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_moment)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_moments': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_moment):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_moments': b}
                for a, b in zip(outputs_class[:-1], outputs_moment[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_moments(self, outputs, targets, indices, num_boxes):
        """ Compute the losses related to the moments: regression (L1) and KL divergence.
        """
        assert 'pred_moments' in outputs
        losses = {}
        losses['loss_moments_l1'] = self.loss_moments_regression(outputs, targets, indices, num_boxes)
        losses['loss_moments_kl'] = self.loss_moments_kl(outputs, targets, indices, num_boxes)
        return losses
    
    def loss_moments_regression(self, outputs, targets, indices, num_boxes):
        """Compute the regression losses for the moments.
        targets dicts must contain the key "moments" containing a tensor of dim [nb_target_boxes, 5]
        with each row being [cx, cy, mu11, mu20, mu02].
        """
        assert 'pred_moments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_moments = outputs['pred_moments'][idx]
        target_moments = torch.cat([t['moments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Compute L1 loss for the moments
        loss_moments_l1 = F.l1_loss(src_moments, target_moments, reduction='none')
        loss_moments = loss_moments_l1.sum() / num_boxes
        
        return loss_moments
    
    def loss_moments_kl(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the moments, specifically using KL divergence
        targets dicts must contain the key "moments" containing a tensor of dim [nb_target_boxes, 5]
        with each row being [cx, cy, mu11, mu20, mu02].
        """
        assert 'pred_moments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_moments = outputs['pred_moments'][idx]
        
        #TODO
        if src_moments.shape[0] == 0:
            print("WARNING! No boxes in the batch")
            return torch.tensor(0.0, device=src_moments.device)
        
        target_moments = torch.cat([t['moments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        src_moments = moment_ops.denormalize_moments(src_moments)
        target_moments = moment_ops.denormalize_moments(target_moments)

        # Split the moments into their respective components
        mu_src = src_moments[:, :2]  # Extracting [cx, cy]
        mu_tgt = target_moments[:, :2]
        
        # Constructing the covariance matrices from the moments
        Sigma_src = moment_ops.moments_to_cov(src_moments[:, 2:])
        Sigma_tgt = moment_ops.moments_to_cov(target_moments[:, 2:])

        # Compute KL divergence for each pair of predicted and target distribution
        kl_divergences = [moment_ops.kl_divergence(mu_src[i], Sigma_src[i], mu_tgt[i], Sigma_tgt[i]) 
                                    for i in range(mu_src.shape[0])]
        
        kl_divergences = torch.stack(kl_divergences)
        loss_kl = kl_divergences.sum() / num_boxes

        return loss_kl

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'moments': self.loss_moments,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, method='topk', min_values=None, max_values=None):
        super().__init__()
        if method is None:
            method = 'topk'
        assert method in ['topk', 'label_topk', 'label']
        self.method = method
        self.fn = dict(topk=postprocess_topk,
                       label_topk=postprocess_label_topk,
                       label=postprocess_label)[method]
        
        self.min_values = min_values
        self.max_values = max_values

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_moments = outputs['pred_logits'], outputs['pred_moments']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
       
        moments, labels, scores = self.fn(outputs)

        # and from relative [0, 1] to absolute [0, height] coordinates
        # rescale centroids (cx, cy)
        img_h, img_w = target_sizes.unbind(1)
        
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(moments.device)
        moments[..., :2] = moments[..., :2] * scale_fct[:, None, :2]

        min_values = torch.tensor(MOMENT_MIN_VALUES, dtype=torch.float32, device=moments.device) 
        max_values = torch.tensor(MOMENT_MAX_VALUES, dtype=torch.float32, device=moments.device)

        # Rescale central moments (mu11, mu20, mu02)
        moments[..., 2:] = moments[..., 2:] * (max_values - min_values) + min_values

        results = [{'scores': s, 'labels': l, 'moments': m} for s, l, m in zip(scores, labels, moments)]

        return results
    
def postprocess_topk(outputs):
    """
    This is the default procedure for postprocessing the output of DETR.
    We have modified it so that k is one third of the number of queries, rather than 100.
    """
    out_logits = outputs['pred_logits']
    out_moments = outputs['pred_moments']
    prob = out_logits.sigmoid()
    k    = int(out_logits.size(-2) // 3)
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k, dim=1)
    scores = topk_values
    topk_moments = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    moments = torch.gather(out_moments, 1, topk_moments.view(-1, k, 1).repeat(1, 1, out_moments.shape[-1]))
    return moments, labels, scores

def postprocess_label_topk(outputs):
    out_logits = outputs['pred_logits']
    out_moments = outputs['pred_moments']
    prob = out_logits.sigmoid()
    
    # get score and label of each query
    scores, labels  = prob.max(-1)
    # select top-k queries
    k    = int(out_logits.size(-2) // 3)
    topk_scores, topk_indices = torch.topk(scores, k, dim=1)
    
    # gather
    labels = torch.gather(labels, 1, topk_indices)
    scores = torch.gather(scores, 1, topk_indices)
    moments = torch.gather(out_moments, 1, topk_indices.view(-1, k, 1).repeat(1,1,5))
    
    # convert format
    #boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    return moments, labels, scores

def postprocess_label(outputs):
    out_logits = outputs['pred_logits']
    out_moments = outputs['pred_moments']
    prob = out_logits.sigmoid()
    
    # get score and label of each query
    scores, labels  = prob.max(-1)
    
    return out_moments, labels, scores


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x