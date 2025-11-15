"""Comprehensive model evaluation."""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from sklearn.metrics import confusion_matrix, classification_report
import cv2

from src.data.dataset import WasteDataset


class WasteEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(self.cfg)
        
    def evaluate_coco_metrics(self, dataset_name):
        """Evaluate using COCO metrics (mAP, mAP50, etc.)"""
        evaluator = COCOEvaluator(dataset_name, output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, dataset_name)
        results = inference_on_dataset(self.predictor.model, val_loader, evaluator)
        
        # Extract key metrics
        if 'segm' in results:
            metrics = {
                'mAP': results['segm']['AP'],
                'mAP50': results['segm']['AP50'],
                'mAP75': results['segm']['AP75'],
                'mAP_small': results['segm']['APs'],
                'mAP_medium': results['segm']['APm'],
                'mAP_large': results['segm']['APl'],
            }
        else:
            metrics = results
        
        return metrics, results
    
    def compute_confusion_matrix(self, dataset_name, output_path):
        """Compute confusion matrix"""
        dataset_dicts = DatasetCatalog.get(dataset_name)
        
        y_true = []
        y_pred = []
        
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            outputs = self.predictor(img)
            
            pred_classes = outputs["instances"].pred_classes.cpu().numpy()
            true_classes = [ann["category_id"] for ann in d["annotations"]]
            
            # Match predictions to ground truth (simple IoU matching)
            # This is simplified - production code would use Hungarian matching
            for true_class in true_classes:
                if len(pred_classes) > 0:
                    y_true.append(true_class)
                    y_pred.append(pred_classes[0])
        
        if len(y_true) == 0:
            print("No ground truth annotations found")
            return None, ""
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(WasteDataset.CATEGORIES.values()),
            yticklabels=list(WasteDataset.CATEGORIES.values())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
        plt.close()
        
        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=list(WasteDataset.CATEGORIES.values())
        )
        
        with open(os.path.join(output_path, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        return cm, report
    
    def analyze_predictions(self, dataset_name, num_samples=10, output_path='output'):
        """Visualize sample predictions"""
        dataset_dicts = DatasetCatalog.get(dataset_name)
        
        if len(dataset_dicts) == 0:
            print("No data found in dataset")
            return
        
        samples = np.random.choice(len(dataset_dicts), min(num_samples, len(dataset_dicts)), replace=False)
        os.makedirs(output_path, exist_ok=True)
        
        metadata = MetadataCatalog.get(dataset_name)
        
        for i, idx in enumerate(samples):
            d = dataset_dicts[idx]
            img = cv2.imread(d["file_name"])
            outputs = self.predictor(img)
            
            # Visualize
            v = Visualizer(
                img[:, :, ::-1],
                metadata=metadata,
                scale=1.0
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            cv2.imwrite(
                os.path.join(output_path, f'prediction_{i}.jpg'),
                out.get_image()[:, :, ::-1]
            )
    
    def failure_case_analysis(self, dataset_name, threshold=0.3, output_path='output'):
        """Analyze failure cases (low confidence or wrong predictions)"""
        dataset_dicts = DatasetCatalog.get(dataset_name)
        
        failure_cases = []
        
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            outputs = self.predictor(img)
            
            # Check for low confidence predictions
            scores = outputs["instances"].scores.cpu().numpy()
            if len(scores) > 0 and np.min(scores) < threshold:
                failure_cases.append({
                    'file': d["file_name"],
                    'min_score': float(np.min(scores)),
                    'reason': 'low_confidence'
                })
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save failure cases
        with open(os.path.join(output_path, 'failure_cases.json'), 'w') as f:
            json.dump(failure_cases, f, indent=2)
        
        print(f"Found {len(failure_cases)} failure cases")
        return failure_cases
