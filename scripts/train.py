#!/usr/bin/env python3
"""Training script for waste detection model."""

import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import WasteDataset
from src.model.trainer import setup_config, train_model


def main():
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
    print("Setting up model configuration...")
    num_classes = config.get('model', {}).get('num_classes', 4)
    output_dir = config.get('paths', {}).get('checkpoint_dir', 'models/checkpoints')
    
    cfg = setup_config(
        num_classes=num_classes,
        output_dir=output_dir
    )
    
    # Override with config file settings if available
    if 'training' in config:
        training_cfg = config['training']
        cfg.SOLVER.IMS_PER_BATCH = training_cfg.get('batch_size', 4)
        cfg.SOLVER.BASE_LR = training_cfg.get('base_lr', 0.001)
        if 'num_epochs' in training_cfg:
            cfg.SOLVER.MAX_ITER = training_cfg['num_epochs'] * 1000  # Rough estimate
        cfg.SOLVER.AMP.ENABLED = training_cfg.get('amp_enabled', True)
    
    # Train
    print("Starting training...")
    trainer = train_model(cfg)
    
    print("Training complete!")
    print(f"Model saved to: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

