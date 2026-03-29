# Multimodal Deepfake Detection (FastAPI Backend)

This project provides a robust, heavily scalable API backend built with **FastAPI** to detect deepfakes and verify information across four distinct media modalities: **Text, Audio, Image, and Video**.

The underlying inference pipelines leverage state-of-the-art models (like Groq for Text and AssemblyAI for Audio) integrated into a single, cohesive asynchronous web server.

---

## 🚀 Features

- **Blazing Fast API**: Powered by FastAPI, ensuring high performance, easy integration, and async support.
- **Multimodal Support**:
  - `Text`: Analyzes and verifies news articles or statements.
  - `Audio`: Transcribes and checks audio clips (.mp3, .wav, .m4a) for anomalies.
  - `Image`: Processes uploaded image files via an image detection pipeline.
  - `Video`: Processes and detects deepfakes on uploaded video files (.mp4, .avi, etc.).
- **Auto-Generated Documentation**: Every endpoint is automatically documented and easily testable via Swagger UI.

---

## 🛠️ Installation & Setup (Back-end)

### 1. Prerequisites 
- Python 3.8+
- Active API keys for Groq and AssemblyAI.

### 2. Navigate to the backend directory
```bash
cd backend
```

### 3. Install Dependencies
Install the required packages. (Optionally, use a virtual environment).
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart pydantic dotenv
```

### 4. Setup Environment Variables
Create a `.env` file inside the `backend` directory with local API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
ASSEMBLY_AI_API_KEY=your_assemblyai_api_key_here
```

### 5. Start the FastAPI Server
To run the server in development mode with live reloading:
```bash
uvicorn app:app --reload
```
The server will boot up at: `http://localhost:8000`

---

## 📖 API Endpoints

Once the server is running, FastAPI automatically generates an interactive documentation page. 
Navigate to: **[http://localhost:8000/docs](http://localhost:8000/docs)** to see the Swagger UI.

### Summary of Routes:

| Method | Endpoint | Description | Payload Type |
|--------|----------|-------------|--------------|
| `POST` | `/api/check-text` | Verifies news/text statements | JSON `{"query": "your text"}` |
| `POST` | `/api/check-audio`| Checks audio files for deepfakes | `multipart/form-data` |
| `POST` | `/api/check-image`| Checks static images for deepfakes | `multipart/form-data` |
| `POST` | `/api/check-video`| Checks video files for deepfakes | `multipart/form-data` |

*Note: The frontend implementation is isolated in the `frontend/` directory, built with React and Vite. It consumes these endpoints via standard HTTP Axios requests.*
