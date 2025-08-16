from dataset.fer_dataset import FERDataset
from explained.models import FERClassifier
from paths import TRAIN, TEST, CHECKPOINTS, LOGS

import os
from explainer.cnn_explainer import CNNExplainer
from cf_explainer import LTX

from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import models

from torchvision.transforms.v2 import(
    CenterCrop,  
    Compose,  
    Normalize,  
    RandomRotation,  
    RandomResizedCrop,  
    RandomHorizontalFlip, 
    RandomAdjustSharpness,  
    Resize,  
    ToImage
)

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from transformers import AutoModel, AutoImageProcessor, AutoModelForImageClassification
from transformers import get_linear_schedule_with_warmup

from functools import partial
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# TODO: put in config or some other place
img_size = 224
batch_size = 32


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18", use_fast=True)

train_transforms = Compose(
    [
        Resize((img_size, img_size)),
        # RandomRotation(90),
        # RandomAdjustSharpness(2),   
        # RandomHorizontalFlip(0.5),
    ]
)

test_transforms = Compose(
    [
        Resize((img_size, img_size))
    ]
)

train = FERDataset(root_dir='data/fer-2013/train', transform=train_transforms)
test = FERDataset(root_dir='data/fer-2013/test', transform=test_transforms)

def collate_fn(batch, processor):
    imgs = [item['pixel_values'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    out = processor(images=imgs, return_tensors="pt")
    return out["pixel_values"], labels


train_loader = DataLoader(
    train, 
    batch_size=batch_size, 
    num_workers=4,
    shuffle=True, 
    persistent_workers=True,
    collate_fn=partial(collate_fn, processor=processor)
)
test_loader = DataLoader(
    test, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,
    persistent_workers=True,
    collate_fn=partial(collate_fn, processor=processor)
)


EXPLAINED_TYPE = 'cnn'  # or 'vit' for ViT-based explainers

explain_backbone = models.resnet18(weights='IMAGENET1K_V1')
checkpoints_path = os.path.join(CHECKPOINTS,EXPLAINED_TYPE)
explained_model = FERClassifier.load_from_checkpoint(
    checkpoint_path=os.path.join(checkpoints_path, "best-checkpoint.ckpt"), 
    backbone=explain_backbone, 
    backbone_type=EXPLAINED_TYPE
)

explainer = CNNExplainer(
    cnn_model=copy.deepcopy(explained_model.backbone),
    activation_function='sigmoid',
    img_size=img_size
)

ltx = LTX(
    explained_model=explained_model,
    explainer=explainer,
    activation_function='sigmoid',
    img_size=img_size,
    img_mean=processor.image_mean,
    img_std=processor.image_std,
    lr=2e-3,
    lambda_inv=1,
    lambda_mask=1
)

checkpoints_path = os.path.join(CHECKPOINTS, 'LTX' , EXPLAINED_TYPE)
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoints_path,
    filename='best-checkpoint',
    monitor='val/loss',
    mode='min',
    save_top_k=1,
    enable_version_counter=False
)

logs_path = os.path.join(LOGS, 'LTX', EXPLAINED_TYPE)

early_stopping_callback = EarlyStopping(
    monitor='val/loss',
    patience=10,  # Stop training if no improvement for 3 epochs
    mode='min',  # We want to minimize the validation loss
    verbose=False,
    check_finite=True
)

logger = TensorBoardLogger(
    logs_path
)

trainer = Trainer(
    max_epochs=20,
    accelerator='gpu',
    enable_checkpointing=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
    logger=logger
)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    trainer.fit(
        model=ltx,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        # ckpt_path=os.path.join(checkpoints_path, 'best-checkpoint.ckpt')
    )