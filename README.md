# AI Voice Detector

A multilingual AI-generated speech detection system supporting English, Hindi, Malayalam, Telugu, and Tamil.

## Features

- Detects AI-generated vs human speech
- MultiLanguage support (English, Hindi, Malayalam, Telugu, Tamil)
- FastAPI REST API
- Real-time audio processing

## Deployment on Render

### Prerequisites
- GitHub account with repository containing this code
- Render account (https://render.com)

### Deployment Steps

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Render**
   - Go to https://dashboard.render.com
   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Click "Create Web Service"

3. **Configure Service**
   - **Name**: ai-voice-detector
   - **Environment**: Python 3
   - **Region**: Oregon (or your preferred region)
   - **Plan**: Free (or paid for better performance)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**
   - In Render dashboard, go to Environment
   - Add environment variables (see `.env.example`)

5. **Deploy**
   - Render will automatically deploy when you push to main branch

### Local Development

1. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

2. **Run the server**
   ```bash
   uvicorn app:app --reload
   ```

3. **Access API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
- **GET** `/` - Returns health status

### Prediction
- **POST** `/predict` - Detect if audio is AI-generated
  - Input: audio file or base64 encoded audio
  - Output: prediction result with confidence

## Requirements

See `requirements.txt` for all dependencies:
- FastAPI
- Uvicorn
- NumPy
- Librosa
- SoundFile
- Pandas
- SciPy
- Scikit-learn
- Pydantic

## Model

The detector uses a trained Random Forest classifier with 60+ extracted audio features.

### Supported Languages
- English
- Hindi
- Malayalam
- Telugu
- Tamil

## Files Required for Render Deployment

✅ `render.yaml` - Render configuration
✅ `Procfile` - Process file for deployment
✅ `runtime.txt` - Python runtime version
✅ `requirements.txt` - Python dependencies
✅ `.env.example` - Environment variable template
✅ `build.sh` - Build script
✅ `.gitignore` - Git ignore rules

## Troubleshooting

### Port Issues
- Render automatically assigns the PORT environment variable
- The app must listen on 0.0.0.0 and the assigned PORT

### Model Loading
- Ensure `ai_audio_detector.pkl` is in the project root
- Check file permissions if model fails to load

### Memory Issues
- Free tier on Render has limited memory (256MB)
- If running out of memory, upgrade to a paid plan

## Support

For issues or questions, check the Render documentation:
https://render.com/docs

---

**Deployed on Render** ☁️
