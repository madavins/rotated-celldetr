# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for moment manipulation, normalization and KL divergence.
"""
import torch
from torchvision.ops.boxes import box_area
import numpy as np
from math import sin, cos, radians
from celldetr.util.constants import MOMENT_MIN_VALUES, MOMENT_MAX_VALUES


def box_moments_to_xyxy(x):
    x_c, y_c, mu11, mu20, mu02 = x.unbind(-1)
    
    # Angle of the oriented bounding box
    theta_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta_rad)
    
    # Estimate semi-major and semi-minor axes lengths
    a = np.sqrt(2 * (mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4*mu11**2)))
    b = np.sqrt(2 * (mu20 + mu20 - np.sqrt((mu20 - mu02)**2 + 4*mu11**2)))
    
    corners = [(-a, -b), (-a, b), (a, b), (a, -b)]
    
    rotated_corners = [rotate_point((0, 0), corner, radians(theta_deg)) for corner in corners]
    bbox_corners = [(x + x_c, y + y_c) for x, y in rotated_corners]
    
    return torch.stack(bbox_corners, dim=-1)
    
def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Parameters:
    - origin: A tuple (ox, oy) representing the coordinates of the origin.
    - point: A tuple (px, py) representing the coordinates of the point to be rotated.
    - angle: The rotation angle in radians.

    Returns:
    - A tuple representing the rotated point's coordinates.
    """
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)

    return qx, qy

def normalize_moments(moments, w=256, h=256):
    cx, cy, mu11, mu20, mu02 = moments.unbind(-1)
    cx /= w
    cy /= h
    moments_vector = torch.stack([mu11, mu20, mu02], dim=-1)
    min_values = torch.tensor(MOMENT_MIN_VALUES, device=moments.device)
    max_values = torch.tensor(MOMENT_MAX_VALUES, device=moments.device)
    normalized_moments = (moments_vector - min_values) / (max_values - min_values)
    return torch.stack([cx, cy] + list(normalized_moments.unbind(-1)), dim=-1)

def denormalize_moments(moments, w=256, h=256):
    cx, cy, mu11, mu20, mu02 = moments.unbind(-1)
    cx = cx * w
    cy = cy * h
    moments_vector = torch.stack([mu11, mu20, mu02], dim=-1)
    min_values = torch.tensor(MOMENT_MIN_VALUES, device=moments.device)
    max_values = torch.tensor(MOMENT_MAX_VALUES, device=moments.device)
    denormalized_moments = moments_vector * (max_values - min_values) + min_values
    return torch.stack([cx, cy] + list(denormalized_moments.unbind(-1)), dim=-1)

def moments_to_cov(moments):
    """Converts moment values to covariance matrices."""
    mu11, mu20, mu02 = moments[:, 0], moments[:, 1], moments[:, 2]
    cov_matrices = torch.zeros((moments.shape[0], 2, 2), device=moments.device)
    cov_matrices[:, 0, 0] = mu20
    cov_matrices[:, 1, 1] = mu02
    cov_matrices[:, 0, 1] = cov_matrices[:, 1, 0] = mu11
    return cov_matrices

def kl_divergence(mu0, Sigma0, mu1, Sigma1):
    """
    Compute the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    - mu0: Mean vector of the first Gaussian distribution. Predicted moments.
    - Sigma0: Covariance matrix of the first Gaussian distribution. Predicted moments.
    - mu1: Mean vector of the second Gaussian distribution. Target moments.
    - Sigma1: Covariance matrix of the second Gaussian distribution. Target moments.
    
    Returns:
    - KL divergence between the two distributions.
    """

    epsilon = 1e-4  # Small value to add to the diagonal of Sigma1
    Sigma1 = Sigma1 + torch.eye(2, device=Sigma1.device) * epsilon 
    Sigma0 = Sigma0 + torch.eye(2, device=Sigma0.device) * epsilon 
    Sigma1_inv = torch.linalg.inv(Sigma1)
    term1 = torch.trace(Sigma1_inv @ Sigma0)
    diff = (mu1 - mu0).unsqueeze(-1) / 256 # Rescale to avoid numerical issues
    term2 = diff.T @ Sigma1_inv @ diff
    term3 = torch.log(torch.linalg.det(Sigma1) / torch.linalg.det(Sigma0))

    kl_div = 0.5 * (term1 + term2 + term3 - 2)
    
    return kl_div
    
def kl_divergence_batched(mu0, Sigma0, mu1, Sigma1):
    """
    Compute the batched KL divergence between sets of Gaussian distributions.
    
    Parameters:
    - mu0: Tensor of shape [n, 2] - Mean vectors of n source Gaussians.
    - Sigma0: Tensor of shape [n, 2, 2] - Covariance matrices of source Gaussians.
    - mu1: Tensor of shape [m, 2] - Mean vectors of m target Gaussians.
    - Sigma1: Tensor of shape [m, 2, 2] - Covariance matrices of target Gaussians.
    - epsilon: Small regularization term to ensure covariance matrices are invertible.
    
    Returns:
    - A matrix of shape [n, m] containing KL divergences between each pair (i, j).
    """
    epsilon=1e-4
    n, d = mu0.shape # d=2 
    m, _ = mu1.shape

    # Regularize the covariance matrices to ensure they are invertible
    Sigma1 = Sigma1 + torch.eye(d, device=Sigma1.device) * epsilon
    Sigma0 = Sigma0 + torch.eye(d, device=Sigma0.device) * epsilon

    # Invert Sigma1 once since it's for targets and calculate determinant
    Sigma1_inv = torch.linalg.inv(Sigma1)  # Shape [m, 2, 2]
    det_Sigma1 = torch.linalg.det(Sigma1)  # Shape [m]

    #Repeat the tensor of sigma_1s n times adapting it's dimensions to make
    # a 1 to 1 comparison with the tensor of sigma_0s.
    Sigma1_inv_expanded = Sigma1_inv.unsqueeze(0).expand(n, m, d, d) 

    #TERM 1: trace(Sigma_1_inv * Sigma_0)
    #Tensor of sigma_0s is also expanded to adapt its dimensionality.
    product = torch.matmul(Sigma1_inv_expanded, Sigma0.unsqueeze(1).expand(n, m, d, d)) #[n, m, 2, 2]
    term1 = torch.diagonal(product, dim1=-2, dim2=-1).sum(-1) #[n, m]

    #TERM 2: Mahalanobis distance
    diff = (mu1.unsqueeze(0) - mu0.unsqueeze(1)) / 256 #[n, m, 2]
    intermediate = torch.matmul(diff.unsqueeze(-2), Sigma1_inv_expanded) #[n, m, 1, 2]
    term2 = torch.matmul(intermediate, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1) #[n, m, 1, 1] -> [n, m]

    #TERM 3: ln(det(sigma_1) / det(sigma_0))
    det_Sigma0 = torch.linalg.det(Sigma0)
    term3 = torch.log(det_Sigma1.unsqueeze(0) / det_Sigma0.unsqueeze(1))

    #Final computation -> number of dimensions of the gaussian distribution is 
    # substracted to make the divergence invariant to the number of dimensions
    kl_matrix = 0.5 * (term1 + term2 + term3 - d)
    return kl_matrix
