from abc import ABC
from transformers import AutoModel, AutoImageProcessor
import torch

dinov2 = AutoModel.from_pretrained('facebook/dinov2-small')
dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small',use_fast=True)

class Backbone(ABC):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def collate_fn(self, batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

class DinoBackbone(Backbone):
    def __init__(self):
        model = dinov2
        processor = dinov2_processor
        super().__init__(model, processor)
    
    def collate_fn(self, batch):
        img = [item['pixel_values'] for item in batch]
        labels = [item['label'] for item in batch]
        processed_img = self.processor(img)
        
        img_tensor = torch.stack(processed_img['pixel_values'])
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return img_tensor, label_tensor