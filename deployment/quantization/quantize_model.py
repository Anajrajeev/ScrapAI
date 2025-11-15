"""Model quantization utilities."""

import torch
import torch.quantization
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os
from pathlib import Path


class ModelQuantizer:
    """Quantize model for edge deployment"""
    
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.model_path = model_path
        
    def quantize_dynamic(self, output_path='models/exported/model_quantized_dynamic.pth'):
        """
        Dynamic quantization - fastest, good for models with large linear layers
        Reduces model size by ~75% with minimal accuracy loss
        """
        # Load model
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.model_path)
        model.eval()
        
        # Note: Dynamic quantization for Detectron2 models is more complex
        # This is a simplified version - full implementation would require
        # quantizing individual components (backbone, RPN, ROI heads)
        
        # Apply dynamic quantization to supported layers
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
                dtype=torch.qint8
            )
        except Exception as e:
            print(f"Warning: Full dynamic quantization not supported: {e}")
            print("Using model as-is for now")
            quantized_model = model
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)
        print(f"Dynamic quantized model saved to {output_path}")
        
        # Print size comparison
        if os.path.exists(self.model_path):
            original_size = os.path.getsize(self.model_path) / (1024**2)
            quantized_size = os.path.getsize(output_path) / (1024**2)
            print(f"Original size: {original_size:.2f} MB")
            print(f"Quantized size: {quantized_size:.2f} MB")
            if original_size > 0:
                print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        return quantized_model
    
    def quantize_static(self, calibration_data, output_path='models/exported/model_quantized_static.pth'):
        """
        Static quantization - best accuracy/performance tradeoff
        Requires calibration data
        """
        # Load model
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.model_path)
        model.eval()
        
        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        print("Calibrating model...")
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                if i >= 100:  # Use 100 samples for calibration
                    break
                try:
                    model(data)
                except Exception as e:
                    print(f"Warning: Calibration sample {i} failed: {e}")
                    continue
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Static quantized model saved to {output_path}")
        
        return model
