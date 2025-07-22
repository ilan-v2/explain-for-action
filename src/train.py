import logging
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from paths import TRAIN, TEST, CHECKPOINTS, LOGS
import torch

# import logging


from backbone import DinoBackbone
from models import FERClassifier
from dataset.fer_dataset import FERDataset

from torchvision.transforms.v2 import(
    CenterCrop,  
    Compose,  
    Normalize,  
    RandomRotation,  
    RandomResizedCrop,  
    RandomHorizontalFlip, 
    RandomAdjustSharpness,  
    Resize,  
    ToTensor
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = FERDataset(root_dir = TRAIN)
test = FERDataset(root_dir = TEST)

backbone = DinoBackbone()

size = backbone.processor.size['shortest_edge']
train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(degrees=90),
    RandomAdjustSharpness(sharpness_factor=2.0),
    RandomHorizontalFlip(0.5),
])

test_transforms = Compose([
    Resize((size, size)),
])

train.transform = train_transforms
test.transform = test_transforms

train_loader = DataLoader(
    train,
    batch_size=32,
    shuffle=True,
    collate_fn=backbone.collate_fn,
    num_workers=4,
    persistent_workers=True
)

test_loader = DataLoader(
    test,
    batch_size=32,
    shuffle=False,
    collate_fn=backbone.collate_fn,
    num_workers=4,
    persistent_workers=True
)

classifier = FERClassifier(backbone.model)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINTS,
    filename='best-checkpoint',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop training if no improvement for 3 epochs
    mode='min',  # We want to minimize the validation loss
    verbose=False,
    check_finite=True
)

logger = TensorBoardLogger(
    LOGS
)

trainer = Trainer(
    max_epochs=20,
    accelerator='gpu',
    enable_checkpointing=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
    logger=logger
)

if __name__ == "__main__":
    # silence logging from transformers library
    logging.getLogger("transformers").setLevel(logging.ERROR)

    torch.set_float32_matmul_precision('high')
    trainer.fit(
        model=classifier,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )

