"""API routes for waste detection."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import cv2
import numpy as np
from typing import List, Optional

from api.models import PredictionResponse, BatchPredictionResponse

router = APIRouter()

# Global predictor instance (should be initialized on startup)
predictor = None


def initialize_predictor(model_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """Initialize the predictor model."""
    global predictor
    from src.inference.predictor import OptimizedPredictor
    predictor = OptimizedPredictor(model_path, config_path, device)


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict waste materials in a single uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with detections and recyclability score
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_path = Path(tmp_file.name)
        content = await file.read()
        tmp_path.write_bytes(content)
    
    try:
        # Make prediction
        outputs, inference_time = predictor.predict(tmp_path, return_time=True)
        
        # Get recyclability score
        recyclability = predictor.get_recyclability_score(outputs)
        
        # Format detections
        instances = outputs["instances"]
        detections = []
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
            
            from src.data.dataset import WasteDataset
            
            for i in range(len(instances)):
                detection = {
                    "bbox": boxes[i].tolist(),
                    "class": WasteDataset.CATEGORIES[int(classes[i])],
                    "confidence": float(scores[i])
                }
                if masks is not None:
                    detection["has_mask"] = True
                detections.append(detection)
        
        return {
            "detections": detections,
            "recyclability": recyclability,
            "inference_time": round(inference_time * 1000, 2)  # ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict waste materials for multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with predictions for all images
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    tmp_paths = []
    
    try:
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_path = Path(tmp_file.name)
                content = await file.read()
                tmp_path.write_bytes(content)
                tmp_paths.append(tmp_path)
            
            # Make prediction
            outputs, inference_time = predictor.predict(tmp_path, return_time=True)
            
            # Get recyclability score
            recyclability = predictor.get_recyclability_score(outputs)
            
            # Format detections
            instances = outputs["instances"]
            detections = []
            
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                
                from src.data.dataset import WasteDataset
                
                for i in range(len(instances)):
                    detection = {
                        "bbox": boxes[i].tolist(),
                        "class": WasteDataset.CATEGORIES[int(classes[i])],
                        "confidence": float(scores[i])
                    }
                    detections.append(detection)
            
            results.append({
                "file": file.filename,
                "detections": detections,
                "recyclability": recyclability,
                "inference_time": round(inference_time * 1000, 2)
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        for tmp_path in tmp_paths:
            if tmp_path.exists():
                tmp_path.unlink()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "predictor_loaded": predictor is not None}


@router.get("/categories")
async def get_categories():
    """Get list of waste categories."""
    from src.data.dataset import WasteDataset
    return {
        "categories": list(WasteDataset.CATEGORIES.values()),
        "category_mapping": {v: k for k, v in WasteDataset.CATEGORIES.items()}
    }
