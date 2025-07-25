import torch
from typing import Dict, List
import torch.nn as nn


class AlignedMTLOptimizer:
    def __init__(self, model: nn.Module, task_names: List[str]):
        self.model = model
        self.task_names = task_names
        # self.shared_params = [p for p in model.shared_encoder.parameters() if p.requires_grad]
        self.shared_params = [
          p for name, p in model.named_parameters()
          if any(key in name for key in ['backbone', 'encoder', 'fusion']) and p.requires_grad
        ]

    def _get_task_gradient(self) -> torch.Tensor:
        grads = []
        for p in self.shared_params:
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))
        return torch.cat(grads)

    def compute_aligned_gradient(self, loss_dict: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
      device = self.shared_params[0].device
      task_grads = []

    # 统一一次 forward 的计算图（无需多次 backward）
      for task in self.task_names:
        grads = torch.autograd.grad(
            loss_dict[task],
            self.shared_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )
        grad_vector = torch.cat([
            g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
            for g, p in zip(grads, self.shared_params)
        ])
        task_grads.append(grad_vector.detach())

      # G = torch.stack(task_grads, dim=1)
      MAX_PARAM = 500_000
      G = torch.stack([g[:MAX_PARAM] for g in task_grads], dim=1)

      U, S, Vh = torch.linalg.svd(G, full_matrices=False)
      sigma_min = S[-1]
      G_aligned = sigma_min * torch.matmul(U, Vh)

      w = torch.ones(len(self.task_names), device=device) / len(self.task_names)
      g_hat = torch.matmul(G_aligned, w)

    # 设置梯度
      start = 0
      for p in self.shared_params:
          numel = p.numel()
          p.grad = g_hat[start:start + numel].view_as(p).clone()
          start += numel

    # 清空非共享梯度（可选）
      for n, p in self.model.named_parameters():
        if p not in self.shared_params and p.grad is not None:
            p.grad = None
