from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import numpy as np
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

# Models and Pipelines
from models.text_model import text_model
from models.audio_model import audio_model
from pipelines.inference_image import image_model
from pipelines.inference_video import video_model

load_dotenv()

app = FastAPI(title="Multimodal Deepfake Detection API")

# Setup CORS to allow React frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to the specific origin in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_api_key = os.getenv('GROQ_API_KEY')
audio_api_key = os.getenv("ASSEMBLY_AI_API_KEY")

class TextQuery(BaseModel):
    query: str

@app.post("/api/check-text")
async def check_text(request: TextQuery):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Please enter some text before checking")
    try:
        model = text_model(text_api_key)
        response = model.predict(request.query)
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-audio")
async def check_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a file before checking")
    try:
        file_bytes = await file.read()
        # Create temp file
        temp_file_path = f"output/audio/temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)
            
        model = audio_model(text_api_key, audio_api_key)
        response = model.predict(temp_file_path)
        
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-image")
async def check_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Please upload an image before checking")
    try:
        file_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        query_array = np.array(pil_image)
        
        model = image_model() 
        response = model.predict(query_array)   
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-video")
async def check_video(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a video before checking")
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f'output/video/run_{timestamp}.mp4'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        file_bytes = await file.read()
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
            
        model = video_model()
        response = model.predict(file_path)
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

