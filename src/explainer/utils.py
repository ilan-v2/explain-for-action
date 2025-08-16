import torch
import torch.nn as nn
import torch.nn.functional as F

class LTXLoss(nn.Module):
    def __init__(self,
                 lambda_mask: float = 30.0,
                 lambda_inv: float = 0.0,
                 lambda_smooth: float = 0.0):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_inv = lambda_inv
        self.lambda_smooth = lambda_smooth
        self.pred_loss = nn.CrossEntropyLoss()

    def forward(self,
                logits_m: torch.Tensor,   # [batch_size, n_classes]
                logits_inv: torch.Tensor, # [batch_size, n_classes]
                mask: torch.Tensor,       # [batch_size, 1, img_size, img_size]
                target: torch.Tensor      # [batch_size]
            ) -> torch.Tensor:

        # 1. Prediction loss
        pred = self.pred_loss(logits_m, target)

        # 2. Mask sparsity (L1)
        mask_l1 = mask.mean()

        # 3. Inversion: encourage misclassification (maximize entropy)
        inv = torch.tensor(0.0, device=mask.device)
        if self.lambda_inv > 0:
            probs = F.softmax(logits_inv, dim=1)
            # entropy across classes
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            inv = ent

        # 4. Smoothness: TV via conv for compactness
        tv_h = (mask[:,:,1:,:] - mask[:,:,:-1,:]).abs().mean()
        tv_w = (mask[:,:,:,1:] - mask[:,:,:,:-1]).abs().mean()
        smooth = (tv_h + tv_w) if self.lambda_smooth > 0 else 0.0

        # 5. Weighted sum
        return (pred
                + self.lambda_mask * mask_l1
                + self.lambda_inv * inv
                + self.lambda_smooth * smooth)