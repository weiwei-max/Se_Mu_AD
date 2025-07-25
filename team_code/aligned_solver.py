import torch

class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        """
        grads: Tensor of shape [1, T, D] where
               T = number of tasks,
               D = dimension of shared representation
        """
        assert len(grads.shape) == 3, f"Expected shape [1, T, D], got {grads.shape}"

        with torch.no_grad():
            G = grads[0]  # [T, D]
            GtG = G.T @ G  # [D, D] covariance matrix in task space

            # Eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(GtG)  # ascending order

            # Filter near-zero eigenvalues for stability
            eps = torch.finfo(eigvals.dtype).eps
            valid = eigvals > eps
            eigvals = eigvals[valid]
            eigvecs = eigvecs[:, valid]

            if len(eigvals) == 0:
                raise ValueError("All eigenvalues are below numerical precision.")

            # Choose scaling strategy
            if scale_mode == 'min':
                σ = eigvals.min()
            elif scale_mode == 'median':
                σ = eigvals.median()
            elif scale_mode == 'rmse':
                σ = eigvals.mean()
            else:
                raise ValueError("Invalid scale_mode. Choose from ['min', 'median', 'rmse'].")

            # Construct transformation matrix B
            S_inv = torch.diag(1.0 / torch.sqrt(eigvals))
            B = torch.sqrt(σ) * eigvecs @ S_inv @ eigvecs.T  # [D, D]

            # Apply B to each task gradient
            aligned_grads = grads @ B.T.unsqueeze(0)  # [1, T, D] × [D, D]^T

            return aligned_grads, B, eigvals
