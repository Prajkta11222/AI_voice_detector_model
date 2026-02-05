import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import glob
import base64
import tempfile
import io
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional

class AIAudioDetector:
    """
    Unified AI Audio Detection System
    Single multilingual model for detecting AI-generated vs human speech
    Supports: English, Hindi, Malayalam, Telugu, Tamil
    """
    
    def __init__(self, model_path='ai_audio_detector.pkl'):
        """Initialize the detector"""
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.model_path = model_path
        
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, audio_path, sr=16000):
        """Extract 60+ audio features for detection"""
        try:
            audio, _ = librosa.load(audio_path, sr=sr)
            features = {}
            
            # Basic Statistics
            features['rms'] = np.sqrt(np.mean(audio**2))
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_mean_{i}'] = np.mean(mfccs[i])
                features[f'mfcc_std_{i}'] = np.std(mfccs[i])
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast_mean'] = np.mean(contrast)
            features['spectral_contrast_std'] = np.std(contrast)
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz)
            
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spec_mean'] = np.mean(mel_spec_db)
            features['mel_spec_std'] = np.std(mel_spec_db)
            
            # Harmonic-Percussive
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic_mean'] = np.mean(harmonic)
            features['percussive_mean'] = np.mean(percussive)
            features['harmonic_ratio'] = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
            
            # Statistical
            features['skewness'] = skew(audio)
            features['kurtosis'] = kurtosis(audio)
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            features['pitch_mean'] = np.mean(pitches) if len(pitches) > 0 else 0
            features['pitch_std'] = np.std(pitches) if len(pitches) > 0 else 0
            
            # Phase
            analytic_signal = signal.hilbert(audio)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            features['phase_std'] = np.std(np.diff(instantaneous_phase))
            
            # Silence Ratio
            silence_threshold = 0.01
            features['silence_ratio'] = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            
            # Spectral Flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            
            # Tempogram
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            features['tempogram_mean'] = np.mean(tempogram)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def train(self, dataset_dir, model_type='rf'):
        """
        Train single unified model on all languages combined
        Dataset structure:
        dataset/
          ├── ai/
          │   ├── english/
          │   ├── hindi/
          │   └── ...
          └── real/
              ├── english/
              ├── hindi/
              └── ...
        """
        print("\n" + "="*60)
        print("TRAINING UNIFIED MULTILINGUAL MODEL")
        print("="*60)
        
        real_files = []
        ai_files = []
        
        # Collect files from all languages
        languages = ['english', 'hindi', 'malayalam', 'telugu', 'tamil']
        
        for language in languages:
            real_path = os.path.join(dataset_dir, 'real', language)
            ai_path = os.path.join(dataset_dir, 'ai', language)
            
            if os.path.exists(real_path):
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    real_files.extend(glob.glob(os.path.join(real_path, ext)))
            
            if os.path.exists(ai_path):
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    ai_files.extend(glob.glob(os.path.join(ai_path, ext)))
        
        print(f"Real audio files: {len(real_files)}")
        print(f"AI audio files: {len(ai_files)}")
        
        if len(real_files) == 0 or len(ai_files) == 0:
            print("Error: No audio files found for training")
            return False
        
        # Extract features
        features_list = []
        labels = []
        
        print("\nExtracting features from real audio...")
        for i, path in enumerate(real_files):
            features = self.extract_features(path)
            if features:
                features_list.append(list(features.values()))
                labels.append(0)
                if i == 0:
                    self.feature_names = list(features.keys())
                if (i + 1) % 10 == 0:
                    print(f"  Processed: {i + 1}/{len(real_files)}")
        
        print("Extracting features from AI audio...")
        for i, path in enumerate(ai_files):
            features = self.extract_features(path)
            if features:
                features_list.append(list(features.values()))
                labels.append(1)
                if (i + 1) % 10 == 0:
                    print(f"  Processed: {i + 1}/{len(ai_files)}")
        
        if len(features_list) == 0:
            print("Error: No features extracted")
            return False
        
        X = np.array(features_list)
        y = np.array(labels)
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Feature count: {X.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining model...")
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced'
            )
        else:
            raise ValueError("model_type must be 'rf' or 'svm'")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Real', 'AI-Generated']))
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("✓ Training Complete")
        print("="*60)
        
        return True
    
    def predict(self, audio_path):
        """
        Predict whether audio is AI-generated
        Returns: (prediction, probability, confidence)
            prediction: 0 = Real, 1 = AI-Generated
            probability: P(AI-Generated)
            confidence: confidence level [0-1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")
        
        features = self.extract_features(audio_path)
        if features is None:
            raise ValueError("Could not extract features from audio")
        
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        ai_prob = probability[1]
        confidence = max(ai_prob, 1 - ai_prob)
        
        return prediction, ai_prob, confidence
    
    def analyze(self, audio_path):
        """Analyze audio and return detailed results"""
        print(f"\nAnalyzing: {audio_path}")
        print("="*60)
        
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
        
        try:
            pred, ai_prob, confidence = self.predict(audio_path)
            
            result = "AI-GENERATED" if pred == 1 else "REAL HUMAN"
            print(f"Result: {result}")
            print(f"AI Probability: {ai_prob:.2%}")
            print(f"Confidence: {confidence:.2%}")
            
            print("\n=== Audio Details ===")
            print(f"Duration: {len(audio)/sr:.2f}s")
            print(f"Sample Rate: {sr} Hz")
            print(f"Max Amplitude: {np.max(np.abs(audio)):.4f}")
            print(f"RMS Energy: {np.sqrt(np.mean(audio**2)):.6f}")
            
            return {
                'result': result,
                'prediction': pred,
                'ai_probability': ai_prob,
                'confidence': confidence,
                'duration': len(audio)/sr
            }
        
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None
    
    def save_model(self):
        """Save trained model to a fixed path: models/ai_audio_detector.pkl.

        This function always writes to the `models` directory in the current
        working directory and contains only model-saving logic (no prints
        or audio-related output). It also updates `self.model_path` to the
        canonical saved location.
        """
        import os
        import pickle

        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        target_path = os.path.join(models_dir, "ai_audio_detector.pkl")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        with open(target_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Keep self.model_path in sync with where we saved the file
        self.model_path = target_path
    
    def load_model(self, path=None):
        """Load trained model"""
        load_path = path or self.model_path
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', [])
            
            print(f"Model loaded from {load_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# ============================================================================
# Request/Response Models
# ============================================================================

class VoiceDetectionRequest(BaseModel):
    language: str = Field(
        ...,
        min_length=1,
        description="Language code (e.g., 'english', 'hindi', 'malayalam', 'telugu', 'tamil')"
    )
    audioFormat: str = Field(
        ...,
        min_length=1,
        description="Audio format (e.g., 'wav', 'mp3', 'flac')"
    )
    audioBase64: str = Field(
        ...,
        min_length=1,
        description="Base64 encoded audio data"
    )
    
    @validator('language')
    def validate_language(cls, v):
        if not v.strip():
            raise ValueError('language cannot be empty')
        return v.lower()
    
    @validator('audioFormat')
    def validate_audio_format(cls, v):
        if not v.strip():
            raise ValueError('audioFormat cannot be empty')
        return v.lower()
    
    @validator('audioBase64')
    def validate_audio_base64(cls, v):
        if not v.strip():
            raise ValueError('audioBase64 cannot be empty')
        return v

class VoiceDetectionResponse(BaseModel):
    success: bool
    is_ai: Optional[bool] = None
    ai_probability: Optional[float] = None
    confidence: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AI Voice Detection API",
    description="Unified multilingual AI-generated audio detection",
    version="1.0.0"
)

# Global detector instance
detector_instance: Optional[AIAudioDetector] = None
API_KEY = os.getenv('API_KEY', 'my-secret-key')


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global detector_instance
    model_path = os.getenv('MODEL_PATH', 'ai_audio_detector.pkl')
    detector_instance = AIAudioDetector(model_path=model_path)


@app.post("/api/voice-detection")
async def voice_detection(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(...)
) -> VoiceDetectionResponse:
    """
    Detect if audio is AI-generated
    
    Args:
        x-api-key: API key for authentication
        request: {
            language: str (e.g., 'english', 'hindi', 'malayalam', 'telugu', 'tamil'),
            audioFormat: str (e.g., 'wav', 'mp3', 'flac'),
            audioBase64: str (base64 encoded audio data)
        }
    
    Returns:
        {
            success: bool,
            is_ai: bool (if success),
            ai_probability: float 0-1 (if success),
            confidence: float 0-1 (if success),
            result: str 'REAL HUMAN' or 'AI-GENERATED' (if success),
            error: str (if failed)
        }
    """
    
    # Validate API key (case-insensitive comparison)
    if x_api_key.lower() != API_KEY.lower():
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Validate inputs
        if not request.audioBase64:
            return VoiceDetectionResponse(
                success=False,
                error="audioBase64 is required"
            )
        
        if not request.language:
            return VoiceDetectionResponse(
                success=False,
                error="language is required"
            )
        
        if not request.audioFormat:
            return VoiceDetectionResponse(
                success=False,
                error="audioFormat is required"
            )
        
        # Check model loaded
        if detector_instance is None or detector_instance.model is None:
            return VoiceDetectionResponse(
                success=False,
                error="Model not loaded. Train model first."
            )
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception as e:
            return VoiceDetectionResponse(
                success=False,
                error=f"Invalid base64 encoding: {str(e)}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f'.{request.audioFormat}',
            delete=False
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Perform detection
            pred, ai_prob, confidence = detector_instance.predict(tmp_path)
            
            return VoiceDetectionResponse(
                success=True,
                is_ai=bool(pred),
                ai_probability=float(ai_prob),
                confidence=float(confidence),
                result='AI-GENERATED' if pred == 1 else 'REAL HUMAN'
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        return VoiceDetectionResponse(
            success=False,
            error=f"Detection failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    model_loaded = detector_instance is not None and detector_instance.model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "languages": ["english", "hindi", "malayalam", "telugu", "tamil"],
        "endpoints": {
            "POST /api/voice-detection": "Detect if audio is AI-generated",
            "GET /health": "Health check"
        }
    }


# ============================================================================
# CLI Support (for backward compatibility)
# ============================================================================



# ============================================================================
# CLI Support Classes (Optional, for backward compatibility)
# ============================================================================

class AudioDetectionAPI:
    """CLI wrapper for audio detection"""
    
    def __init__(self, model_path='ai_audio_detector.pkl'):
        self.detector = AIAudioDetector(model_path)
    
    def detect(self, audio_path):
        """Detect if audio is AI-generated"""
        try:
            pred, ai_prob, confidence = self.detector.predict(audio_path)
            
            return {
                'is_ai': bool(pred),
                'ai_probability': float(ai_prob),
                'confidence': float(confidence),
                'result': 'AI-GENERATED' if pred == 1 else 'REAL HUMAN',
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def batch_detect(self, audio_folder):
        """Detect for multiple audio files"""
        results = {}
        audio_files = []
        
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(glob.glob(os.path.join(audio_folder, ext)))
        
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            result = self.detect(audio_file)
            results[filename] = result
        
        return results
    
    def train(self, dataset_dir, model_type='rf'):
        """Train the model on dataset"""
        return self.detector.train(dataset_dir, model_type=model_type)


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified AI Audio Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--audio', type=str, help='Audio file to analyze')
    parser.add_argument('--folder', type=str, help='Folder with audio files')
    parser.add_argument('--model', type=str, default='ai_audio_detector.pkl', help='Model file path')
    parser.add_argument('--model_type', type=str, default='rf', help='Model type: rf or svm')
    parser.add_argument('--info', action='store_true', help='Show dataset info')
    parser.add_argument('--api', action='store_true', help='Run FastAPI server')
    
    args = parser.parse_args()
    
    # FastAPI mode
    if args.api:
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    
    # CLI mode
    else:
        api = AudioDetectionAPI(model_path=args.model)
        
        if args.info:
            print("""
DATASET STRUCTURE
=================

dataset/
├── ai/                    (AI-generated speech)
│   ├── english/
│   ├── hindi/
│   ├── malayalam/
│   ├── telugu/
│   └── tamil/
└── real/                  (Real human speech)
    ├── english/
    ├── hindi/
    ├── malayalam/
    ├── telugu/
    └── tamil/

Place audio files (.wav, .mp3, .flac) in appropriate directories.
Minimum: 50+ files per category. Recommended: 100+.
            """)
        
        elif args.train:
            api.train(args.dataset, model_type=args.model_type)
        
        elif args.audio:
            api.detector.analyze(args.audio)
        
        elif args.folder:
            results = api.batch_detect(args.folder)
            
            df = pd.DataFrame([
                {
                    'file': k,
                    'result': v.get('result', 'ERROR'),
                    'ai_probability': v.get('ai_probability', None),
                    'confidence': v.get('confidence', None)
                }
                for k, v in results.items()
            ])
            
            print("\n" + "="*60)
            print("BATCH RESULTS")
            print("="*60)
            print(df.to_string(index=False))
            
            ai_count = sum(1 for v in results.values() if v.get('is_ai'))
            print(f"\nAI-Generated: {ai_count}/{len(results)} ({ai_count/len(results)*100:.1f}%)")
        
        else:
            print("""
╔════════════════════════════════════════════════════════════╗
║   UNIFIED AI AUDIO DETECTION SYSTEM                        ║
║   Single Multilingual Model                                ║
║   Supports: English, Hindi, Malayalam, Telugu, Tamil       ║
╚════════════════════════════════════════════════════════════╝

USAGE:

CLI MODE:

1. View dataset structure:
   python app.py --info

2. Train unified model:
   python app.py --train --dataset dataset

3. Analyze single audio:
   python app.py --audio path/to/audio.wav

4. Batch analyze folder:
   python app.py --folder path/to/folder

5. Train with SVM:
   python app.py --train --dataset dataset --model_type svm

API MODE:

1. Run FastAPI server:
   python app.py --api

2. Or with uvicorn:
   uvicorn app:app --host 0.0.0.0 --port 8000

3. Then POST to:
   curl -X POST http://localhost:8000/api/voice-detection \\
     -H "x-api-key: my-secret-key" \\
     -H "Content-Type: application/json" \\
     -d '{"language":"english","audioFormat":"wav","audioBase64":"..."}'

╚════════════════════════════════════════════════════════════╝
            """)
