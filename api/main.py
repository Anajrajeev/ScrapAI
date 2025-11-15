"""Main API application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

from api.routes import router, initialize_predictor
from api.models import HealthResponse

app = FastAPI(
    title="ScrapAI API",
    description="API for waste material detection and recyclability assessment",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/model_final.pth")
    config_path = os.getenv("CONFIG_PATH", None)
    device = os.getenv("DEVICE", "cuda")
    
    if os.path.exists(model_path):
        try:
            initialize_predictor(model_path, config_path, device)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
    else:
        print(f"Warning: Model not found at {model_path}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check."""
    return HealthResponse(status="ok", message="ScrapAI API is running")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
