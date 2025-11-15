"""Model export utilities for deployment."""

import torch
import onnx
from detectron2.export import TracingAdapter
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch.nn as nn
from pathlib import Path


class ModelExporter:
    """Export trained model for deployment"""
    
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
    def export_torchscript(self, output_path='models/exported/model.ts'):
        """Export to TorchScript format"""
        # Build model
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.cfg.MODEL.WEIGHTS)
        model.eval()
        
        # Create tracing adapter
        traceable_model = TracingAdapter(model, self.cfg.INPUT.FORMAT)
        
        # Example input
        height, width = 800, 800
        image = torch.randn(3, height, width).to(self.cfg.MODEL.DEVICE)
        
        # Trace model
        with torch.no_grad():
            traced_model = torch.jit.trace(traceable_model, (image,))
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced_model, output_path)
        print(f"TorchScript model saved to {output_path}")
        
        return traced_model
    
    def export_onnx(self, output_path='models/exported/model.onnx'):
        """Export to ONNX format"""
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.cfg.MODEL.WEIGHTS)
        model.eval()
        
        # Wrapper for ONNX export
        class ONNXWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, image):
                inputs = [{"image": image}]
                outputs = self.model(inputs)
                return outputs[0]["instances"]
        
        wrapped_model = ONNXWrapper(model)
        
        # Example input
        dummy_input = torch.randn(1, 3, 800, 800).to(self.cfg.MODEL.DEVICE)
        
        # Export
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model saved to {output_path}")
        
        return output_path
