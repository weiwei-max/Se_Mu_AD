import torch
from typing import Dict, List
import torch.nn as nn


class AlignedMTLUB:
    def __init__(self, task_names: List[str]):
        self.task_names = task_names

    def compute_aligned_feature_gradient(self, feature_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            feature_grads: Dict from task name to gradient w.r.t. shared representation H
                           Each value has shape [B, D] or [D]
        Returns:
            aligned_grads: Dict of aligned gradients with same keys
        """
        # Filter out tasks with invalid gradients
        feature_grads = {
            k: v for k, v in feature_grads.items()
            if v is not None and isinstance(v, torch.Tensor) and not torch.isnan(v).any()
        }

        if len(feature_grads) == 0:
            print("[AlignedMTLUB] No valid feature gradients found. Skipping alignment.")
            return {}

        tasks = list(feature_grads.keys())
        Z = torch.stack([feature_grads[t].reshape(-1) for t in tasks], dim=1)  # Shape: [D, T]
        device = Z.device

        if torch.isnan(Z).any():
            raise ValueError("[AlignedMTLUB] NaN detected in feature gradients before alignment")

        # SVD-based gradient alignment
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)  # [D, T] -> [D, D], [D], [T, T]
        sigma_min = S[-1]
        Z_aligned = sigma_min * torch.matmul(U, Vh)  # [D, T]

        # Equal task weights
        w = torch.ones(len(tasks), device=device) / len(tasks)
        z_hat = torch.matmul(Z_aligned, w)  # [D]

        # Return aligned grads per task
        aligned_grads = {}
        for idx, t in enumerate(tasks):
            aligned_grads[t] = Z_aligned[:, idx].view_as(feature_grads[t]).clone()

        return aligned_grads

    def apply_to_shared_feature(self, shared_feature: torch.Tensor, aligned_grads: Dict[str, torch.Tensor]):
        """
        Inject the sum of aligned gradients into shared_feature.
        Call this before optimizer.step().
        """
        total_grad = sum(aligned_grads.values())
        if shared_feature.grad is not None:
            shared_feature.grad.zero_()
        shared_feature.backward(total_grad)
