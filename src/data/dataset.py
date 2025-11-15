"""Dataset loading and management utilities for waste segmentation."""

import os
import json
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


class WasteDataset:
    """Custom dataset class for waste segmentation"""
    
    CATEGORIES = {
        0: 'plastic',
        1: 'metal',
        2: 'cardboard',
        3: 'glass'
    }
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.ann_file = os.path.join(data_dir, split, 'annotations.json')
        
    def load_annotations(self):
        """Load COCO format annotations"""
        if not os.path.exists(self.ann_file):
            return []
            
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to annotations mapping
        img_to_anns = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Convert to Detectron2 format
        dataset_dicts = []
        for img_info in coco_data.get('images', []):
            record = {}
            
            filename = os.path.join(self.img_dir, img_info['file_name'])
            record["file_name"] = filename
            record["image_id"] = img_info['id']
            record["height"] = img_info['height']
            record["width"] = img_info['width']
            
            anns = img_to_anns.get(img_info['id'], [])
            objs = []
            for ann in anns:
                obj = {
                    "bbox": ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": ann.get('segmentation', []),
                    "category_id": ann['category_id'],
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            
        return dataset_dicts
    
    @staticmethod
    def register_dataset(data_dir, splits=['train', 'val', 'test']):
        """Register dataset with Detectron2"""
        for split in splits:
            dataset_name = f"waste_{split}"
            DatasetCatalog.register(
                dataset_name,
                lambda s=split, d=data_dir: WasteDataset(d, s).load_annotations()
            )
            MetadataCatalog.get(dataset_name).set(
                thing_classes=list(WasteDataset.CATEGORIES.values())
            )

