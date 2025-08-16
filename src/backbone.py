from abc import ABC
from transformers import AutoModel, AutoImageProcessor
import torch

class Backbone(ABC):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.processing_mean = None
        self.processing_std = None
    
    def collate_fn(self, batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

class HuggingFaceViT(Backbone):
    def __init__(self, model_str='facebook/dinov2-small'):
        model = AutoModel.from_pretrained(model_str)
        processor = AutoImageProcessor.from_pretrained(model_str, use_fast=True)
        self.processing_mean = processor.image_mean
        self.processing_std = processor.image_std
        super().__init__(model, processor)
    
    
# TODO: not finished yet
class PytorchCNN(Backbone):
    def __init__(self, model):
        self.model = model
        processor = None
        super().__init__(model, processor)
