"""Visualization utilities for waste detection."""

import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional


class WasteVisualizer:
    """Custom visualization for waste detection"""
    
    def __init__(self, metadata=None):
        if metadata is None:
            try:
                self.metadata = MetadataCatalog.get("waste_train")
            except:
                # Fallback metadata
                from src.data.dataset import WasteDataset
                self.metadata = type('Metadata', (), {
                    'thing_classes': list(WasteDataset.CATEGORIES.values())
                })()
        else:
            self.metadata = metadata
        
    def visualize_predictions(self, image, outputs, show_labels=True, show_masks=True):
        """Visualize predictions with customization"""
        v = Visualizer(
            image[:, :, ::-1],
            metadata=self.metadata,
            scale=1.2,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        instances = outputs["instances"].to("cpu")
        vis = v.draw_instance_predictions(instances)
        
        return vis.get_image()[:, :, ::-1]
    
    def create_comparison_image(self, original, predicted, ground_truth=None):
        """Create side-by-side comparison"""
        images = [original, predicted]
        titles = ['Original', 'Predicted']
        
        if ground_truth is not None:
            images.append(ground_truth)
            titles.append('Ground Truth')
        
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        
        if len(images) == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            if img.ndim == 3 and img.shape[2] == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def draw_recyclability_overlay(self, image, recyclability_info):
        """Draw recyclability information on image"""
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Create info box
        box_height = 150
        box_width = 300
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + box_width, 10 + box_height),
            (0, 0, 0),
            -1
        )
        
        # Add transparency
        alpha = 0.7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Add text
        score = recyclability_info['overall_score']
        confidence = recyclability_info['confidence']
        
        cv2.putText(
            image,
            f"Recyclability: {score:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            image,
            f"Confidence: {confidence:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )
        
        # Color code based on score
        if score >= 85:
            color = (0, 255, 0)  # Green
        elif score >= 70:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(
            image,
            (20, 90),
            (20 + int(score * 2.6), 110),
            color,
            -1
        )
        
        # Add recommendation
        rec = recyclability_info['recommendation']
        y_offset = 130
        for line in rec.split(' - '):
            cv2.putText(
                image,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 20
        
        return image
