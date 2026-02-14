import torch
from torchmetrics import Metric


class HorizontalPCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("pccs", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:   shape [B, C] - predicted regression outputs
        targets: shape [B, C] - true target values
        """
        if preds.ndim != 2 or targets.ndim != 2:
            raise ValueError("Expected preds and targets to be of shape [B, C]")

        # Centered vectors
        preds_centered = preds - preds.mean(dim=1, keepdim=True)
        targets_centered = targets - targets.mean(dim=1, keepdim=True)

        # Numerator and denominator
        numerators = (preds_centered * targets_centered).sum(dim=1)
        denominators = preds_centered.norm(dim=1) * targets_centered.norm(dim=1)
        pccs = numerators / denominators

        # Handle NaNs if norm is zero
        pccs = torch.nan_to_num(pccs, nan=0.0)

        # Store
        self.pccs.append(pccs)

    def compute(self):
        """
        Returns mean PCC across all samples
        """
        all_pccs = torch.cat(self.pccs, dim=0)
        return all_pccs.mean()

    def reset(self):
        self.pccs = []


class HorizontalSCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("coeffs", default=[], dist_reduce_fx="cat")

    def _rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ranks row-wise for a 2D tensor x ∈ ℝ^{B×C}
        """
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
        return ranks

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if preds.ndim != 2 or targets.ndim != 2:
            raise ValueError("Expected preds and targets to be of shape [B, C]")

        pred_ranks = self._rank(preds)
        target_ranks = self._rank(targets)

        pred_ranks = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
        target_ranks = target_ranks - target_ranks.mean(dim=1, keepdim=True)

        numerators = (pred_ranks * target_ranks).sum(dim=1)
        denominators = pred_ranks.norm(dim=1) * target_ranks.norm(dim=1)
        coeffs = numerators / denominators

        coeffs = torch.nan_to_num(coeffs, nan=0.0)
        self.coeffs.append(coeffs)

    def compute(self):
        return torch.cat(self.coeffs, dim=0).mean()

    def reset(self):
        self.coeffs = []
