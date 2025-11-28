# 🍽️ Egyptian Food Detection Web Application

A complete web application for detecting Egyptian food items using a custom-trained YOLOv8 model. Upload images and get real-time predictions with bounding boxes for 9 Egyptian food categories.

## 📋 Detected Food Categories

- Bechamel
- Molokhya
- Ataif
- Besala
- Fool
- Konafa
- Koshary
- Pasposa
- Taamia

## 🏗️ Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **ML Model**: YOLOv8 (Ultralytics)
- **Deployment**: Docker

## 🚀 Running Locally

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

## 🐳 Running with Docker

### Build the Docker image:

```bash
docker build -t egyptian-food-detector .
```

### Run the container:

```bash
docker run -p 8000:8000 egyptian-food-detector
```

### Access the application:

Open your browser and navigate to:
```
http://localhost:8000
```

## 📦 Docker Hub Deployment

### Tag and push to Docker Hub:

```bash
# Tag the image
docker tag egyptian-food-detector your-username/egyptian-food-detector:latest

# Login to Docker Hub
docker login

# Push the image
docker push your-username/egyptian-food-detector:latest
```

### Pull and run from Docker Hub:

```bash
docker pull your-username/egyptian-food-detector:latest
docker run -p 8000:8000 your-username/egyptian-food-detector:latest
```

## 🔧 API Endpoints

### GET `/`
Serves the web interface

### GET `/health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST `/predict`
Upload an image for food detection

**Parameters:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "koshary",
      "confidence": 0.89,
      "bbox": {
        "x1": 120.5,
        "y1": 80.2,
        "x2": 450.8,
        "y2": 380.6
      }
    }
  ],
  "count": 1,
  "annotated_image": "data:image/jpeg;base64,..."
}
```

## 📁 Project Structure

```
survive_depi/
├── app.py                          # FastAPI backend
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── .dockerignore                   # Docker ignore file
├── static/
│   └── index.html                  # Frontend interface
└── runs/
    └── detect/
        └── train4/
            └── weights/
                └── best.pt         # Trained model
```

## 🌟 Features

- **Drag & Drop**: Easy image upload with drag-and-drop support
- **Real-time Detection**: Instant food detection results
- **Visual Feedback**: Annotated images with bounding boxes
- **Confidence Scores**: See detection confidence for each item
- **Responsive Design**: Works on desktop and mobile devices
- **REST API**: Easy integration with other applications

## 🛠️ Technology Stack

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **YOLOv8**: State-of-the-art object detection
- **OpenCV**: Image processing
- **Docker**: Containerization

## 📝 Notes

- The model is trained on Egyptian food images
- Minimum confidence threshold is set to 0.25
- Supports common image formats (JPG, PNG, WEBP)

## 🐛 Troubleshooting

### Model not found error
Ensure the model file exists at: `runs/detect/train4/weights/best.pt`

### Port already in use
Change the port in `app.py` or use:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Docker build fails
Make sure you have enough disk space and Docker is running

## 📄 License

This project uses a custom-trained YOLOv8 model based on the Egypt Food dataset from Roboflow.

## 👨‍💻 Development

To modify the application:

1. Update backend: Edit `app.py`
2. Update frontend: Edit `static/index.html`
3. Update dependencies: Edit `requirements.txt`
4. Rebuild Docker image after changes

---

Made with ❤️ for Egyptian food lovers!
