import torch
from ..backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .deformable_detr import DeformableDETR, PostProcess, SetCriterion
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .matcher import build_matcher

def build_deformable_detr(cfg, backbone):
    # the num_classes of the model refers to max_obj_id (i.e. 5 for pannuke) +1
    num_classes = cfg.model.num_classes + 1
    transformer = build_deforamble_transformer(cfg)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.model.num_queries,
        num_feature_levels=cfg.model.num_feature_levels,
        aux_loss=cfg.model.aux_loss,
        with_box_refine=cfg.model.with_box_refine,
        two_stage=cfg.model.two_stage,
    )

    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    assert cfg.matcher.name == 'HungarianMatcher', "Currently only HungarianMatcher is supported"
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.loss.class_coef, 
                   'loss_moments_l1': cfg.loss.moments_coef,
                   'loss_moments_kl': cfg.loss.kl_coef}
    
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #weight_dict["loss_mask"] = args.mask_loss_coef
        #weight_dict["loss_dice"] = args.dice_loss_coef
    
    # TODO this is a hack
    if cfg.model.aux_loss:
        aux_weight_dict = {}
        for i in range(cfg.model.transformer.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'moments', 'cardinality']
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #losses += ["masks"]
    
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=cfg.loss.focal_alpha)
    postprocessors = {'moments': PostProcess(cfg.model.postprocess)}
    
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #postprocessors['segm'] = PostProcessSegm()
        #if args.dataset_file == "coco_panoptic":
        #    is_thing_map = {i: i <= 90 for i in range(201)}
        #    postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

def modify_checkpoint_for_moments(checkpoint):
    """
    Adjusts checkpoint parameters for the moments-based bbox_embed layers.
    """
    new_checkpoint = {}
    for key, value in checkpoint.items():
        # Adjust the bbox_embed layers that have changed due to moments addition
        if 'bbox_embed' in key and ('layers.2.weight' in key or 'layers.2.bias' in key):
            new_value = value  # Start with the original value
            if 'weight' in key:
                # Adjust weight dimensions for moments ([5, 256] for example)
                new_shape = (5, value.size(1))  # Ensure 2D shape for weights
                new_value = torch.empty(new_shape)
                torch.nn.init.xavier_uniform_(new_value)  # Re-initialize
            elif 'bias' in key:
                # Adjust bias to match moments dimension (5,)
                new_shape = (5,)
                new_value = torch.zeros(new_shape)  # Bias can be initialized to zeros
            new_checkpoint[key] = new_value
        else:
            # Keep unchanged parameters as is
            new_checkpoint[key] = value
    return new_checkpoint

def load_sd_deformable_detr(model, checkpoint):
    # get model, if checkpoint is a dict
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    # remove class_embed from checkpoint
    checkpoint = {k: v for k, v in checkpoint.items() if 'class_embed' not in k}

    # Modify the checkpoint to accommodate moment-based bbox_embed layers
    modified_checkpoint = modify_checkpoint_for_moments(checkpoint)

    print(f"\t Loading deformable detr (adapted for moments) with {len(modified_checkpoint)} keys...")
    # load state dict
    missing_keys, unexpected_keys = model.load_state_dict(modified_checkpoint, strict=False)
    print(f"\t # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(modified_checkpoint.keys())}")
    print(f"\t # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")