"""Advanced augmentation pipeline for waste images."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


class WasteAugmentation:
    """Advanced augmentation pipeline for waste images"""
    
    @staticmethod
    def get_training_augmentation():
        """Training augmentation pipeline"""
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.7
            ),
            
            # Color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # Weather effects
            A.RandomShadow(p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            
            # Quality degradation
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids']
        ))
    
    @staticmethod
    def get_validation_augmentation():
        """Validation augmentation (minimal)"""
        return A.Compose([
            # Only normalization for validation
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids']
        ))
    
    @staticmethod
    def apply_augmentation(image, bboxes, masks, category_ids, augmentation):
        """Apply augmentation to image, bboxes, and masks"""
        # Convert masks to list format for albumentations
        transformed = augmentation(
            image=image,
            bboxes=bboxes,
            masks=masks,
            category_ids=category_ids
        )
        
        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['masks'],
            transformed['category_ids']
        )

