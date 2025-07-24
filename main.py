"""
Combined Vehicle Detection + Face Comparison API v0.1.0
Supports both vehicle detection and face comparison functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ExifTags
import requests
import io
import torch
import numpy as np
import base64

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Detection + Face Comparison API",
    description="Combined API for vehicle detection and face comparison",
    version="0.1.0"
)

# ========================= Vehicle Detection =========================

# Load YOLO model for vehicle detection
vehicle_model = YOLO("yolov8n.pt")
ALLOWED_CLASSES = ["bicycle", "car", "motorcycle", "bus", "truck"]

class ImageRequest(BaseModel):
    image_url: str
    
@app.post("/predict", tags=["Vehicle Detection"])
async def predict_vehicle(req: ImageRequest):
    """Detect vehicles in an image from URL"""
    try:
        # Download image from URL
        response = requests.get(req.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Image could not be downloaded.")

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        results = vehicle_model(image)

        predictions = []
        boxes = results[0].boxes
        
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = vehicle_model.names[class_id]
            if class_name not in ALLOWED_CLASSES:
                continue

            confidence = float(box.conf[0]) * 100
            x1, y1, x2, y2 = box.xyxy[0]
            area = float((x2 - x1) * (y2 - y1))

            predictions.append({
                "type": class_name,
                "confidence": round(confidence, 2),
                "area": round(area, 2)
            })

        if not predictions:
            return JSONResponse(
                status_code=200,
                content={
                    "top_result": "unknown",
                    "message": "No known vehicle type detected.",
                    "all_predictions": [],
                },
            )

        # Find max area
        max_area = max(p["area"] for p in predictions)

        # Filter predictions within 90% of max area
        filtered = [p for p in predictions if p["area"] >= 0.9 * max_area]

        # Find highest confidence in filtered group
        if filtered:
            top_pred = max(filtered, key=lambda x: x["confidence"])
        else:
            top_pred = max(predictions, key=lambda x: x["area"])

        # Add area difference percentage
        for p in predictions:
            area_diff_pct = (1 - p["area"] / max_area) * 100
            p["area_diff_pct"] = round(area_diff_pct, 2)

        return {
            "top_result": top_pred,
            "all_predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ========================= Face Comparison =========================

class CompareRequest(BaseModel):
    url1: str
    url2: str
    threshold: float = 0.9

# Initialize face recognition models
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def auto_orient_image(img):
    """Auto-rotate image based on EXIF orientation"""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def get_face_embeddings(pil_image):
    """Extract face embeddings from image"""
    faces = mtcnn(pil_image)
    if faces is None:
        return None, None
    with torch.no_grad():
        embeddings = resnet(faces)
    return embeddings, faces

def pil_to_base64(pil_image, quality=70, resize_to=(128, 128)):
    """Convert PIL image to base64 with compression"""
    if resize_to:
        pil_image = pil_image.resize(resize_to)
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/compare_face", tags=["Face Comparison"])
def compare_face(req: CompareRequest):
    """Compare faces from two image URLs"""
    try:
        # Download and process images
        response1 = requests.get(req.url1)
        response2 = requests.get(req.url2)
        
        if response1.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download image 1")
        if response2.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download image 2")
        
        img1 = auto_orient_image(
            Image.open(io.BytesIO(response1.content)).convert("RGB")
        )
        img2 = auto_orient_image(
            Image.open(io.BytesIO(response2.content)).convert("RGB")
        )

        # Extract faces and embeddings from image 1
        emb1_faces = mtcnn(img1)
        if emb1_faces is None or emb1_faces.shape[0] == 0:
            return {"match": False, "message": "❌ No face found in image 1"}

        emb1 = emb1_faces[0].unsqueeze(0)
        with torch.no_grad():
            emb1_vec = resnet(emb1)

        # Extract faces and embeddings from image 2
        emb2, faces2 = get_face_embeddings(img2)
        if emb2 is None or faces2 is None or emb2.shape[0] == 0:
            return {"match": False, "message": "❌ No face found in image 2"}

        # Calculate distances
        distances = torch.norm(emb2 - emb1_vec, dim=1).cpu().numpy()
        min_distance = distances.min()
        min_index = distances.argmin()

        # Convert best matching face to base64
        best_face_tensor = faces2[min_index]
        face_img = Image.fromarray(
            (best_face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        
        face_base64 = pil_to_base64(face_img, quality=70, resize_to=(128, 128))

        if min_distance < req.threshold:
            return {
                "match": True,
                "distance": float(min_distance),
                "message": f"✅ Face match found (distance = {min_distance:.4f})",
                "base64Image": face_base64
            }
        else:
            return {
                "match": False,
                "distance": float(min_distance),
                "message": f"❌ No face match (min distance = {min_distance:.4f})",
                "base64Image": face_base64
            }

    except Exception as e:
        return {
            "match": False,
            "message": f"❌ Error occurred: {str(e)}"
        }

# ========================= Health Check & Root =========================

@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "message": "Vehicle Detection + Face Comparison API v0.1.0",
        "endpoints": {
            "vehicle_detection": "/predict",
            "face_comparison": "/compare_face",
            "health_check": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "services": ["vehicle_detection", "face_comparison"]
    }