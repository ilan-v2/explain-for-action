import torch 
from lightning.pytorch import LightningModule
from torcheval.metrics import MulticlassAccuracy
from explainer.cnn_explainer import CNNExplainer

class LTXLoss(torch.nn.Module):
    def __init__(self,
                 lambda_mask: float = 30.0,
                 lambda_inv: float = 0.0,
                 lambda_smooth: float = 0.0):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_inv = lambda_inv
        self.lambda_smooth = lambda_smooth
        self.pred_loss_fn = torch.nn.CrossEntropyLoss()

        self.pred_loss = 0
        self.mask_loss = 0
        self.inv_loss = 0
        self.smooth_loss = 0

    @staticmethod
    def _calc_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.nn.functional.softmax(logits, dim=1).clamp_min(1e-8)
        H = -(probs * probs.log()).sum(dim=1).mean()  # entropy
        return -H

    def forward(self,
                logits_m: torch.Tensor,
                logits_inv: torch.Tensor,
                mask: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        
        # 1. Prediction loss
        self.pred_loss = self.pred_loss_fn(logits_m, target)

        # 2. Mask sparsity (L1)
        self.mask_loss = mask.abs().mean()

        # 3. Inversion: encourage misclassification (maximize entropy)
        # take negative pos loss (BCE)
        self.inv_loss = self._calc_entropy(logits_inv)

        # 4. Smoothness: TV via conv for compactness
        # WIP
        self.smooth_loss = torch.tensor(0.0, device=mask.device)

        # 5. Weighted sum
        return (self.pred_loss
                + self.lambda_mask * self.mask_loss
                + self.lambda_inv * self.inv_loss
                + self.lambda_smooth * self.smooth_loss
                )
    
class LTX(LightningModule):
    """
    LightningModule that encapsulates:
    1. The frozen explained model (e.g. a pretrained CNN or ViT).
    2. The trainable explainer (CNNExplainer).
    3. The counterfactual LTX loss.
    Handles both pretraining (on a dataset) and can be re-used for per-instance finetuning.
    """
    num_classes = 7  # Number of classes in the FER dataset TODO: make it dynamic
    def __init__(
        self, 
        explained_model: torch.nn.Module,
        explainer : CNNExplainer,
        activation_function: str = "sigmoid",
        img_size: int = 224,
        img_mean: list = [0.5, 0.5, 0.5],
        img_std: list = [1,1, 1],
        lr: float = 2e-3,
        lambda_mask: float = 30.0,
        lambda_inv: float = 0.0,
        lambda_smooth: float = 0.0,
    ):
        super().__init__()
        # freeze the explained model
        self.explained = explained_model.eval()
        for p in self.explained.parameters():
            p.requires_grad = False

        # save hyperparameters
        self.save_hyperparameters(
            "activation_function",
            "img_size",
            "lr",
            "lambda_mask",
            "lambda_inv",
            "lambda_smooth",
        )

        self.explainer = explainer
        explainer.activation_function = activation_function
        
        self.img_mean = img_mean
        self.img_std = img_std

        self.loss_fn = LTXLoss(
            lambda_mask=lambda_mask,
            lambda_inv=lambda_inv,
            lambda_smooth=lambda_smooth,
        )

    def forward(self, x):
        # returns (upsampled_mask, raw_mask)
        return self.explainer(x)


    def log_metrics(self, logits_m, logits_inv, target, prefix: str = "train"):
        on_step = True if prefix == "train" else False
        self.log(f"{prefix}/pred_loss", self.loss_fn.pred_loss, on_step=on_step, on_epoch=True)
        self.log(f"{prefix}/mask_loss", self.loss_fn.mask_loss, on_step=on_step, on_epoch=True)
        self.log(f"{prefix}/inv_loss", self.loss_fn.inv_loss, on_step=on_step, on_epoch=True)

        # self.log(f"{prefix}/inv_loss", self.loss_fn.inv_loss, on_step=False, on_epoch=True)
        # self.log(f"{prefix}/smooth_loss", self.loss_fn.smooth_loss, on_step=False, on_epoch=True)

        # explained accuracy
        pred_acc = self._calc_accuracy(self.num_classes, logits_m, target)
        self.log(f"{prefix}/pred_accuracy", pred_acc, on_step=False, on_epoch=True)
        inv_acc = self._calc_accuracy(self.num_classes, logits_inv, target)
        self.log(f"{prefix}/inv_accuracy", inv_acc, on_step=False, on_epoch=True)

    @staticmethod
    def _calc_accuracy(num_classes, logits, target):
        acc = MulticlassAccuracy(num_classes=num_classes)
        acc.update(logits, target)
        return acc.compute()

    def _eval(self,x,y,mode):
        # 1. get mask
        up_mask, raw_mask = self.explainer(x)

        # 2. form masked and inverted inputs
        xm = x * up_mask  # upsampled mask
        x_inv = x * (1 - up_mask)

        # 3. feed through explained model
        logits_m = self.explained(xm)
        logits_inv = self.explained(x_inv)

        # 4. compute LTX loss
        loss = self.loss_fn(logits_m, logits_inv, raw_mask, y)
        
        self.log(f"{mode}/loss", loss, prog_bar=True)
        self.log_metrics(logits_m, logits_inv, y, prefix=mode)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: [B,3,H,W], y: [B]
        loss = self._eval(x, y, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._eval(x, y, mode="val")
        return loss

    def configure_optimizers(self):
        params = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.AdamW(params, lr=self.hparams.lr)
