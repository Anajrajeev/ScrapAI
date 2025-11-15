#!/usr/bin/env python3
"""Evaluation script for waste detection model."""

import sys
import yaml
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import WasteDataset
from src.model.evaluator import WasteEvaluator
from src.model.trainer import setup_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate waste detection model")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--dataset', default='waste_val', help='Dataset split to evaluate')
    parser.add_argument('--output', default=None, help='Output directory for results')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Register dataset
    data_dir = config.get('dataset', {}).get('data_dir', 'data/processed')
    print(f"Registering dataset from {data_dir}...")
    WasteDataset.register_dataset(data_dir)
    
    # Setup configuration
    num_classes = config.get('model', {}).get('num_classes', 4)
    cfg = setup_config(num_classes=num_classes)
    
    # Set output directory
    if args.output:
        cfg.OUTPUT_DIR = args.output
    else:
        cfg.OUTPUT_DIR = config.get('paths', {}).get('output_dir', 'output')
    
    # Evaluate
    print(f"Evaluating model: {args.model}")
    evaluator = WasteEvaluator(cfg, args.model)
    
    # COCO metrics
    print("\nComputing COCO metrics...")
    try:
        metrics, results = evaluator.evaluate_coco_metrics(args.dataset)
        
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Error computing COCO metrics: {e}")
        print("This may be because the model is not fully trained or the dataset is empty")
    
    # Confusion matrix
    print("\nComputing confusion matrix...")
    try:
        evaluator.compute_confusion_matrix(args.dataset, cfg.OUTPUT_DIR)
        print("Confusion matrix saved")
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")
    
    # Sample predictions
    print("\nGenerating sample predictions...")
    try:
        evaluator.analyze_predictions(args.dataset, num_samples=10, output_path=cfg.OUTPUT_DIR)
        print("Sample predictions saved")
    except Exception as e:
        print(f"Error generating sample predictions: {e}")
    
    # Failure analysis
    print("\nAnalyzing failure cases...")
    try:
        evaluator.failure_case_analysis(args.dataset, output_path=cfg.OUTPUT_DIR)
        print("Failure analysis complete")
    except Exception as e:
        print(f"Error analyzing failure cases: {e}")
    
    print(f"\nEvaluation complete! Results saved to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

