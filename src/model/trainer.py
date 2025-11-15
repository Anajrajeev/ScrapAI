"""Model training utilities with Detectron2."""

import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper


class WasteTrainer(DefaultTrainer):
    """Custom trainer with augmentation support"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    
    @classmethod
    def build_train_loader(cls, cfg):
        # Custom data mapper with augmentation
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice"
                ),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.3, horizontal=False, vertical=True),
            ]
        )
        return build_detection_train_loader(cfg, mapper=mapper)


def setup_config(num_classes=4, output_dir='./output'):
    """Setup Detectron2 configuration"""
    cfg = get_cfg()
    
    # Model architecture
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Dataset
    cfg.DATASETS.TRAIN = ("waste_train",)
    cfg.DATASETS.TEST = ("waste_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Training hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (7000, 9000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    # Model settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # Input settings
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Output
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Mixed precision training for speed
    cfg.SOLVER.AMP.ENABLED = True
    
    return cfg


def train_model(cfg):
    """Train the Mask R-CNN model"""
    trainer = WasteTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer


# Training script
if __name__ == "__main__":
    # Register dataset
    from src.data.dataset import WasteDataset
    WasteDataset.register_dataset('data/processed')
    
    # Setup and train
    cfg = setup_config(num_classes=4, output_dir='models/checkpoints')
    trainer = train_model(cfg)
