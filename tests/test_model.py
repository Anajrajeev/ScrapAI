"""Tests for model module."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from src.model.trainer import Trainer
from src.model.evaluator import Evaluator


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(784, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_trainer_initialization():
    """Test trainer initialization."""
    model = DummyModel(num_classes=10)
    device = torch.device("cpu")
    
    # Create dummy data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 1, 28, 28),
        torch.randint(0, 10, (100,))
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.randn(50, 1, 28, 28),
        torch.randint(0, 10, (50,))
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    config = {
        "learning_rate": 0.001
    }
    
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    assert trainer.device == device
    assert trainer.config == config


def test_evaluator_initialization():
    """Test evaluator initialization."""
    model = DummyModel(num_classes=10)
    device = torch.device("cpu")
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.randn(50, 1, 28, 28),
        torch.randint(0, 10, (50,))
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    class_names = [f"class_{i}" for i in range(10)]
    
    evaluator = Evaluator(model, test_loader, device, class_names)
    
    assert evaluator.device == device
    assert evaluator.class_names == class_names

