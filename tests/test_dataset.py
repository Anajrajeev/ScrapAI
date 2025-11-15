"""Tests for dataset module."""

import pytest
from pathlib import Path
from src.data.dataset import ScrapDataset, create_dataloader


def test_scrap_dataset_initialization():
    """Test dataset initialization."""
    data_path = Path("data/raw")
    annotations_path = Path("data/annotations/annotations.json")
    
    dataset = ScrapDataset(
        data_path=data_path,
        annotations_path=annotations_path,
        split="train"
    )
    
    assert dataset.data_path == data_path
    assert dataset.split == "train"


def test_create_dataloader():
    """Test dataloader creation."""
    data_path = Path("data/raw")
    annotations_path = Path("data/annotations/annotations.json")
    
    dataset = ScrapDataset(
        data_path=data_path,
        annotations_path=annotations_path,
        split="train"
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    assert dataloader.batch_size == 32
    assert dataloader.shuffle == True

