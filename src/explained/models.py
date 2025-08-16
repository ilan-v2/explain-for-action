import torch
from lightning.pytorch import LightningModule

class FERClassifier(LightningModule):
    def __init__(
            self, 
            backbone, 
            backbone_type = 'cnn', 
            num_classes=7, 
            lr=3e-6, 
            weight_decay=0.02,
            freeze_backbone=True
        ):
        super(FERClassifier, self).__init__()
        self.backbone = backbone
        self.save_hyperparameters('lr', 'weight_decay', 'freeze_backbone')
        self.num_classes = num_classes
        # freeze the backbone
        if self.hparams.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Check if backbone is ViT or CNN
        if backbone_type == 'vit':
            # ViT-style backbone
            feature_dim = backbone.config.hidden_size
            self.encoder = self._extract_vit_features
        
        elif backbone_type == 'cnn':
            # CNN-style backbone 
            # assuming pytorch implementation
            self.encoder = self._extract_cnn_features
            feature_dim = self.backbone.fc.in_features
        
        else:
            raise ValueError("Unsupported backbone type")
        
        self.classifier = torch.nn.Linear(feature_dim, num_classes)
        
    def _extract_vit_features(self, x):
        x = self.backbone(x, output_hidden_states=True)
        return x.last_hidden_state[:, 0, :]  # CLS token

    def _extract_cnn_features(self, x):
        # remove the classifier head
        backbone_children = list(self.backbone.children())
        feature_extractor = torch.nn.Sequential(*backbone_children[:-1])
        feats = feature_extractor(x) # output shape: [batch_size, 512, 1, 1]
        # flatten the output to [batch_size, 512]
        return feats.view(feats.size(0), -1)  
    
    def forward(self, x):
        if self.hparams.freeze_backbone:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', (logits.argmax(dim=1) == y).float().mean(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)