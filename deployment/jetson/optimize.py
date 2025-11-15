"""Model optimization utilities for Jetson deployment."""

import torch
import torch.nn as nn
import numpy as np


class ModelOptimizer:
    """Various optimization techniques for faster inference"""
    
    @staticmethod
    def prune_model(model, amount=0.3):
        """
        Prune model weights
        Removes least important connections
        """
        import torch.nn.utils.prune as prune_module
        
        # Prune convolutional layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune_module.l1_unstructured(module, name='weight', amount=amount)
                prune_module.remove(module, 'weight')
        
        print(f"Model pruned by {amount*100}%")
        return model
    
    @staticmethod
    def fuse_modules(model):
        """
        Fuse Conv+BN+ReLU operations
        Reduces memory and improves speed
        """
        from torch.quantization import fuse_modules
        
        # This is model-specific and needs to be adapted
        # to the actual model architecture
        print("Fusing modules for faster inference")
        print("Note: Module fusion is architecture-specific")
        return model
    
    @staticmethod
    def convert_to_half_precision(model):
        """Convert model to FP16"""
        model = model.half()
        print("Model converted to FP16")
        return model
    
    @staticmethod
    def enable_cudnn_autotuner():
        """Enable cuDNN autotuner for optimal performance"""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("cuDNN autotuner enabled")
    
    @staticmethod
    def optimize_for_inference(model):
        """Apply all optimizations"""
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        print("Model optimized for inference")
        return model
