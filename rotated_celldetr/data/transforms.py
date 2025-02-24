import copy
import numpy as np

from skimage import color

import torch
import torch.nn as nn
import cv2

import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import _get_fill, _setup_fill_arg

from ..util.box_ops import normalize_box, denormalize_box
from ..util.moment_ops import normalize_moments, denormalize_moments
from rotated_celldetr.util.constants import MOMENT_MAX_VALUES, VARIANCE_THRESHOLD

def build_transforms(cfg, is_train=True):
    transforms = [v2.ToImage()]

    # sanity check
    transforms.append(v2.SanitizeBoundingBoxes())
    transforms.append(v2.ClampBoundingBoxes())

    # augmentation when training
    if is_train:
        # augmentations
        transforms.append(build_augmentations(cfg))

        # sanity check after augmentations
        transforms.append(v2.SanitizeBoundingBoxes())
        transforms.append(v2.ClampBoundingBoxes())
        
    # convert image and bbox format
    transforms.append(v2.ConvertBoundingBoxFormat(format='CXCYWH'))
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    
    # image normalization
    mean = cfg.transforms.normalize.mean
    std = cfg.transforms.normalize.std
    transforms.append(v2.Normalize(mean=mean, std=std))
    # bounding box normalization
    transforms.append(NormalizeBoundingBoxes())

    transforms.append(MaskToMoments())

    # moments sanity check
    transforms.append(FilterMoments(variance_threshold=VARIANCE_THRESHOLD))

    # clamp moments: 0 - max values
    transforms.append(ClampMoments(max_values=MOMENT_MAX_VALUES))

    # min-max normalization based on clamped values
    transforms.append(NormalizeMoments())

    return v2.Compose(transforms)

def build_augmentations(cfg):
    augs = list()
    assert 'augmentations' in cfg.transforms
    for aug in cfg.transforms.augmentations:
        assert 'name' in aug, "Augmentation must have a name."

        # build the augmentation, don't send name and p params as kwargs
        a = AugmentationFactory.build(aug.name,
                                **{k : v for k, v in aug.items() \
                                        if k not in ['name','p']})
        # if p in kwargs, random apply the transform
        if 'p' in aug:
            # TODO: we should use v2.RandomApply but it's crashing with 2 inputs idk why
            a = RandomApply([a], p=aug.p)
        augs.append(a)
    return v2.Compose(augs)

class AugmentationFactory:
    def build(name, **kwargs):
        if name == "hflip":
            t = v2.RandomHorizontalFlip(p=1.0)
        elif name == "vflip":
            t = v2.RandomVerticalFlip(p=1.0)
        elif name == "rotate90":
            t = RandomRotation90(**kwargs)
        elif name == "cjitter":
            t = v2.ColorJitter(**kwargs)
        elif name == "elastic":
            t = v2.ElasticTransform(**kwargs)
        elif name == "blur":
            t = v2.GaussianBlur(**kwargs)
        elif name == "resizedcrop":
            t = v2.RandomResizedCrop(**kwargs, antialias=True)
        elif name == "resize":
            t = v2.Resize(**kwargs, antialias=True)
        elif name == "randomcrop":
            t = v2.RandomCrop(**kwargs)
        elif name == "hedjitter":
            t = HEDJitter(**kwargs)
        else:
            raise ValueError(f'Unknown augmentation: {name}')
        return t

class NormalizeBoundingBoxes(nn.Module):
    def forward(self, image, target):
        h, w = target['boxes'].canvas_size
        # boxes to float
        boxes = copy.deepcopy(target['boxes'].data).float()
        # update target
        target['boxes'].data = normalize_box(boxes, (h,w))
        return image, target

class DenormalizeBoundingBoxes(nn.Module):
    def forward(self, image, target):
        h, w = target['boxes'].canvas_size
        # boxes to float
        boxes = copy.deepcopy(target['boxes'].data)
        # update target
        target['boxes'].data = denormalize_box(boxes, (h,w))
        return image, target

class RandomRotation90(v2.Transform):
    f"""
        Extend RandomRotation by only allowing rotations of 90, 180 and -90 (270) degrees.
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._fill = _setup_fill_arg(0)

    def _get_params(self, flat_inputs):
        #angle = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        # randomly select 90, -90 or 180 (as tensor)
        angles = torch.tensor([0, 90, -90, 180])
        angle  = angles[torch.randperm(4)[0]].item()
        return dict(angle=angle)

    def _transform(self, inpt, params):
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(
            F.rotate,
            inpt,
            **params,
            interpolation=v2.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=fill,
        )

class HEDJitter(nn.Module):
    def __init__(self, alpha = (0.95,1.05),
                       beta  = (-0.05, 0.05)):
        super().__init__()
        if not isinstance(alpha, tuple):
            alpha = (1.0-alpha, 1.0+alpha)
        if not isinstance(beta, tuple):
            beta = (-beta, beta)
        self.alpha = alpha
        self.beta = beta

        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                        [0.07, 0.99, 0.11],
                                        [0.27, 0.57, 0.78]], 
                                        dtype=torch.float32, 
                                        requires_grad=False)
        self.hed_from_rgb = torch.inverse(self.rgb_from_hed)
    
    def forward(self, image, target):
        # get alpha and beta for H, E and D channels
        alpha_H = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        alpha_E = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        #alpha_D = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        beta_H = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()
        beta_E = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()
        #beta_D = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()

        # convert to float32
        orig_dtype = image.dtype
        image = F.convert_image_dtype(image, torch.float32)

        # 
        image = color.rgb2hed(image.permute(1,2,0).numpy())
        image[...,0] = image[...,0] * alpha_H + beta_H
        image[...,1] = image[...,1] * alpha_E + beta_E
        #image[...,2] = image[...,2] * alpha_D + beta_D

        # convert back to rgb tensor
        image = torch.tensor(color.hed2rgb(image)).permute(2,0,1)
        image = F.convert_image_dtype(image, orig_dtype)

        return image, target
    
class RandomApply(v2.Transform):
    def __init__(self, transforms, p = 0.5) -> None:
        super().__init__()

        if not isinstance(transforms, (list, nn.ModuleList)):
            raise TypeError("Argument transforms should be a sequence of callables or a `nn.ModuleList`")
        self.transforms = transforms

        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    def _extract_params_for_v1_transform(self):
        return {"transforms": self.transforms, "p": self.p}

    def forward(self, *inputs):
        needs_unpacking = len(inputs) > 1

        if torch.rand(1) >= self.p:
            return inputs if needs_unpacking else inputs[0]

        for transform in self.transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)
    

class MaskToMoments(torch.nn.Module):
    def forward(self, image, target):
        if 'masks' in target:
            masks = target['masks']
            moments_list = []
            for mask in masks:
                moments = self._moments(mask)
                # Ensure that moments is always a list of floats
                moments_list.append(moments if isinstance(moments, list) else moments.tolist())
            # Safely create a tensor from a uniformly formatted list
            target['moments'] = torch.tensor(moments_list, dtype=torch.float32) if moments_list else torch.zeros(0, 5)
            del target['masks']
        return image, target

    def _moments(self, mask):
        mask = mask.detach().cpu().numpy()
        moments = cv2.moments(mask, binaryImage=True)
        if moments['m00'] == 0: 
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']

        return [cx, cy, mu11, mu20, mu02]

class NormalizeMoments(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, target):
        h, w = image.shape[-2], image.shape[-1]
        if 'moments' in target:
            moments = copy.deepcopy(target['moments']).float()
            target['moments'] = normalize_moments(moments, w, h)
        return image, target

class DenormalizeMoments(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, target):
        h, w = image.shape[-2], image.shape[-1]
        if 'moments' in target:
            moments = copy.deepcopy(target['moments'])
            target['moments'] = denormalize_moments(moments, w, h)
        return image, target

class FilterMoments:
    def __init__(self, variance_threshold=1):
        self.variance_threshold = variance_threshold

    def __call__(self, image, target):
        moments = target['moments']
        mu20 = moments[:, 3]
        mu02 = moments[:, 4]

        # Create a mask for valid moments based on variance threshold
        valid_mask = (mu20 >= self.variance_threshold) & (mu02 >= self.variance_threshold)

        # Filter moments
        target['moments'] = moments[valid_mask]
        target['labels'] = target['labels'][valid_mask]
        #target['masks'] = target['masks'][valid_mask]
        target['boxes'] = target['boxes'][valid_mask]

        return image, target

class ClampMoments:
    def __init__(self, max_values):
        self.max_values = max_values

    def __call__(self, image, target):
        moments = target['moments']
        cx = moments[:, 0]
        cy = moments[:, 1]
        mu11 = moments[:, 2]
        mu20 = moments[:, 3]
        mu02 = moments[:, 4]

        # Clamp moments
        mu11 = torch.clamp(mu11, -self.max_values[0], self.max_values[0])
        mu20 = torch.clamp(mu20, 0, self.max_values[1])
        mu02 = torch.clamp(mu02, 0, self.max_values[2])

        target['moments'] = torch.stack([cx, cy, mu11, mu20, mu02], dim=-1)
        return image, target