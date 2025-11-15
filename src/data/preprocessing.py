"""Data cleaning and preprocessing utilities."""

import cv2
import numpy as np
import json
from pathlib import Path


class DataPreprocessor:
    """Data cleaning and preprocessing utilities"""
    
    @staticmethod
    def clean_dataset(data_dir):
        """Remove corrupted images and invalid annotations"""
        valid_images = []
        corrupted = []
        
        for img_path in Path(data_dir).rglob('*.jpg'):
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    corrupted.append(img_path)
                    continue
                
                # Check minimum size
                if img.shape[0] < 100 or img.shape[1] < 100:
                    corrupted.append(img_path)
                    continue
                    
                valid_images.append(img_path)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                corrupted.append(img_path)
        
        print(f"Valid images: {len(valid_images)}")
        print(f"Corrupted images: {len(corrupted)}")
        
        # Remove corrupted files
        for path in corrupted:
            path.unlink()
        
        return valid_images, corrupted
    
    @staticmethod
    def normalize_annotations(ann_file):
        """Normalize and validate COCO annotations"""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Filter invalid annotations
        valid_anns = []
        for ann in data.get('annotations', []):
            bbox = ann.get('bbox', [])
            # Check bbox validity
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                valid_anns.append(ann)
        
        data['annotations'] = valid_anns
        
        # Save cleaned annotations
        output_file = str(ann_file).replace('.json', '_cleaned.json')
        with open(output_file, 'w') as f:
            json.dump(data, f)
        
        return output_file

