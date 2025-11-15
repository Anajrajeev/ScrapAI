"""Optimized inference module."""

import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import time
from pathlib import Path
from typing import Dict

from src.data.dataset import WasteDataset


class OptimizedPredictor:
    """Optimized predictor for fast inference"""
    
    def __init__(self, model_path, config_path=None, device='cuda'):
        self.device = device
        self.cfg = self._setup_config(model_path, config_path)
        self.predictor = DefaultPredictor(self.cfg)
        
        # Warm up
        self._warmup()
    
    def _setup_config(self, model_path, config_path):
        """Setup configuration"""
        cfg = get_cfg()
        
        if config_path:
            cfg.merge_from_file(config_path)
        else:
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )
        
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = self.device
        
        # Inference optimizations
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.TEST.DETECTIONS_PER_IMAGE = 100
        
        return cfg
    
    def _warmup(self, num_iterations=3):
        """Warm up the model"""
        dummy_image = np.zeros((800, 800, 3), dtype=np.uint8)
        for _ in range(num_iterations):
            _ = self.predictor(dummy_image)
    
    def predict(self, image, return_time=False):
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess if needed
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        # Run prediction
        outputs = self.predictor(image)
        
        inference_time = time.time() - start_time
        
        if return_time:
            return outputs, inference_time
        return outputs
    
    def predict_batch(self, images, batch_size=4):
        """Batch inference for multiple images"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = [self.predict(img) for img in batch]
            results.extend(batch_results)
        
        return results
    
    def get_recyclability_score(self, outputs) -> Dict:
        """
        Calculate recyclability score based on detected materials
        
        Scoring algorithm:
        1. Material base scores (0-100):
           - Plastic: 70 (recyclable but less preferred)
           - Metal: 95 (highly recyclable)
           - Cardboard: 85 (easily recyclable)
           - Glass: 90 (highly recyclable)
        
        2. Confidence weighting:
           - Multiply base score by detection confidence
        
        3. Contamination penalty:
           - Mixed materials in same object: -20%
           - Multiple material types: average scores
        
        4. Condition assessment:
           - Based on image quality metrics (future enhancement)
        """
        
        # Base recyclability scores
        recyclability_map = {
            0: 70,   # plastic
            1: 95,   # metal
            2: 85,   # cardboard
            3: 90    # glass
        }
        
        instances = outputs["instances"]
        
        if len(instances) == 0:
            return {
                'overall_score': 0,
                'material_scores': {},
                'confidence': 0,
                'recommendation': 'No waste detected'
            }
        
        pred_classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        # Calculate weighted scores
        material_scores = {}
        total_weighted_score = 0
        total_confidence = 0
        
        for cls, conf in zip(pred_classes, scores):
            material = WasteDataset.CATEGORIES[int(cls)]
            base_score = recyclability_map[int(cls)]
            weighted_score = base_score * conf
            
            if material not in material_scores:
                material_scores[material] = []
            material_scores[material].append({
                'score': weighted_score,
                'confidence': float(conf)
            })
            
            total_weighted_score += weighted_score
            total_confidence += conf
        
        # Overall score
        overall_score = total_weighted_score / len(pred_classes) if len(pred_classes) > 0 else 0
        avg_confidence = total_confidence / len(pred_classes) if len(pred_classes) > 0 else 0
        
        # Mixed material penalty
        if len(set(pred_classes)) > 2:
            overall_score *= 0.9  # 10% penalty for complex mixed waste
        
        # Generate recommendation
        if overall_score >= 85:
            recommendation = "Highly recyclable - Place in recycling bin"
        elif overall_score >= 70:
            recommendation = "Recyclable - Check local guidelines"
        elif overall_score >= 50:
            recommendation = "Limited recyclability - May require special handling"
        else:
            recommendation = "Low recyclability - Consider waste reduction"
        
        return {
            'overall_score': float(overall_score),
            'material_scores': material_scores,
            'confidence': float(avg_confidence),
            'recommendation': recommendation,
            'material_breakdown': {
                material: len(scores)
                for material, scores in material_scores.items()
            }
        }
