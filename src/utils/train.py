# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
#         super().__init__()
#         self.alpha = alpha  # tensor di pesi per classe, es: [1.0, 2.5, 0.8, ...]
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # inputs: [B, C], raw logits
#         # targets: [B], class indices

#         log_probs = F.log_softmax(inputs, dim=1)
#         probs = torch.exp(log_probs)  # shape: [B, C]
#         targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])

#         # focal weight: (1 - p_t)^gamma
#         pt = (probs * targets_one_hot).sum(dim=1)
#         focal_weight = (1 - pt) ** self.gamma

#         # ce loss: -log(p_t)
#         ce_loss = F.nll_loss(log_probs, targets, reduction="none", weight=self.alpha)

#         loss = focal_weight * ce_loss

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss  # no reduction
