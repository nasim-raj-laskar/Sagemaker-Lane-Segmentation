"""src/api/main.py — FastAPI inference service wrapping the SageMaker endpoint."""

from __future__ import annotations

import base64
import io
import os
import time

import boto3
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "road-seg-endpoint")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(title="Road Segmentation API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


class SegmentResponse(BaseModel):
    mask_png_base64: str
    road_coverage_pct: float
    mask_shape: list[int]
    threshold_used: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "endpoint": ENDPOINT_NAME}


@app.post("/segment", response_model=SegmentResponse)
async def segment(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB guard
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")

    t0 = time.perf_counter()
    try:
        response = _runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType=file.content_type,
            Accept="application/json",
            Body=image_bytes,
        )
    except _runtime.exceptions.ModelError as e:
        raise HTTPException(status_code=502, detail=f"Model error: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Endpoint unavailable: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    import json
    result = json.loads(response["Body"].read())
    return SegmentResponse(latency_ms=round(latency_ms, 1), **result)
