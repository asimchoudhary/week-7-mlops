from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json
import joblib
import numpy as np
import pandas as pd

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer for GCP Cloud Trace
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI app
app = FastAPI(title="🌸 Iris Classifier with Logging & Monitoring")

# Load the actual Iris classification model
model = None

def load_model():
    """Load the Iris classification model"""
    global model
    try:
        model = joblib.load("model.joblib")
        logger.info(json.dumps({
            "event": "model_loaded",
            "status": "success",
            "message": "Iris classification model loaded successfully"
        }))
        return True
    except Exception as e:
        logger.error(json.dumps({
            "event": "model_load_error",
            "error": str(e),
            "status": "failed"
        }))
        return False

def iris_model_inference(features: dict):
    """Perform inference using the loaded Iris model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    # Convert input to DataFrame for model prediction
    input_df = pd.DataFrame([features])
    
    # Perform prediction
    prediction = model.predict(input_df)[0]
    
    # Get prediction probabilities if available
    try:
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))
    except:
        confidence = 0.95  # Default confidence if probabilities not available
    
    return {
        "predicted_class": prediction,
        "confidence": confidence,
        "input_features": features
    }

# Input schema for Iris classification
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float  
    petal_length: float
    petal_width: float


# Simulated flags, normally these would be set by various parts of the code
# e.g. if model load is taking time due to weights being large, 
#  then is_ready would be False until the model is loaded.
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info(json.dumps({
        "event": "startup",
        "message": "Starting model loading process"
    }))
    
    # Load the model
    if load_model():
        app_state["is_ready"] = True
        logger.info(json.dumps({
            "event": "startup_complete",
            "message": "Service is ready to serve predictions"
        }))
    else:
        logger.error(json.dumps({
            "event": "startup_failed",
            "message": "Failed to load model during startup"
        }))
        app_state["is_ready"] = False

@app.get("/")
async def root():
    """Root endpoint with basic service info"""
    return {
        "service": "🌸 Iris Classifier with Logging & Monitoring",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/live_check",
            "readiness": "/ready_check"
        }
    }

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
async def predict(input: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = input.dict()
            result = iris_model_inference(input_data)
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))
            return result

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
