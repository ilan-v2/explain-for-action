import torch
from lightning.pytorch import LightningModule

class FERClassifier(LightningModule):
    def __init__(self, backbone, num_classes=7):
        super(FERClassifier, self).__init__()
        self.backbone = backbone 
        # freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(backbone.config.hidden_size, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        features = x.last_hidden_state[:, 0, :]  # Use the CLS token
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', (logits.argmax(dim=1) == y).float().mean(), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # log accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-6, weight_decay=0.02)