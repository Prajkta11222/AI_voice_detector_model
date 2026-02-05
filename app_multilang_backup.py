import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import glob
import json
warnings.filterwarnings('ignore')

# For ML classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class AIGeneratedAudioDetector:
    """
    AI Generated Audio Detection System
    Detects synthetic/human speech using multiple audio feature analysis
    Supports multi-language training and detection
    """
    
    def __init__(self, model_path=None, language='english'):
        """Initialize the detector with or without a pre-trained model"""
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.language = language
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, audio_path, sr=16000):
        """
        Extract comprehensive audio features for detection
        """
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_path, sr=sr)
            
            features = {}
            
            # 1. Basic Statistics
            features['rms'] = np.sqrt(np.mean(audio**2))
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            
            # 2. MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_mean_{i}'] = np.mean(mfccs[i])
                features[f'mfcc_std_{i}'] = np.std(mfccs[i])
            
            # 3. Chroma Features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # 4. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast_mean'] = np.mean(contrast)
            features['spectral_contrast_std'] = np.std(contrast)
            
            # 5. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz)
            
            # 6. Mel Spectrogram Features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spec_mean'] = np.mean(mel_spec_db)
            features['mel_spec_std'] = np.std(mel_spec_db)
            
            # 7. Harmonic-Percussive Separation
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic_mean'] = np.mean(harmonic)
            features['percussive_mean'] = np.mean(percussive)
            features['harmonic_ratio'] = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
            
            # 8. Statistical Features
            features['skewness'] = skew(audio)
            features['kurtosis'] = kurtosis(audio)
            
            # 9. Pitch Features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            if len(pitches) > 0:
                features['pitch_mean'] = np.mean(pitches)
                features['pitch_std'] = np.std(pitches)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # 10. Phase Analysis (AI audio often has different phase characteristics)
            analytic_signal = signal.hilbert(audio)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            features['phase_std'] = np.std(np.diff(instantaneous_phase))
            
            # 11. Silence Ratio
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            features['silence_ratio'] = silence_ratio
            
            # 12. Spectral Flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            
            # 13. Tempogram (rhythm features)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            features['tempogram_mean'] = np.mean(tempogram)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def extract_features_batch(self, audio_paths, labels=None):
        """Extract features from multiple audio files"""
        features_list = []
        valid_labels = []
        valid_paths = []
        
        for i, audio_path in enumerate(audio_paths):
            features = self.extract_features(audio_path)
            if features:
                features_list.append(list(features.values()))
                if labels is not None:
                    valid_labels.append(labels[i])
                valid_paths.append(audio_path)
                if i == 0:
                    self.feature_names = list(features.keys())
        
        X = np.array(features_list)
        if labels is not None:
            y = np.array(valid_labels)
            return X, y, valid_paths
        return X, None, valid_paths
    
    def train(self, real_audio_paths, ai_audio_paths, model_type='rf', save_path=None):
        """
        Train the detector model
        real_audio_paths: list of paths to real human speech
        ai_audio_paths: list of paths to AI-generated speech
        """
        # Prepare data
        real_labels = [0] * len(real_audio_paths)  # 0 for real
        ai_labels = [1] * len(ai_audio_paths)      # 1 for AI-generated
        
        all_paths = real_audio_paths + ai_audio_paths
        all_labels = real_labels + ai_labels
        
        print(f"Training on {len(all_paths)} audio files...")
        print(f"Real: {len(real_audio_paths)}, AI: {len(ai_audio_paths)}")
        
        # Extract features
        X, y, valid_paths = self.extract_features_batch(all_paths, all_labels)
        
        if len(X) == 0:
            print("No valid audio files found for training!")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
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
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Real', 'AI-Generated']))
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return self.model
    
    def predict(self, audio_path, return_probability=True):
        """
        Predict whether audio is AI-generated
        Returns: prediction (0=real, 1=AI) and probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Train or load a model first.")
        
        # Extract features
        X, _, _ = self.extract_features_batch([audio_path])
        if X is None or len(X) == 0:
            raise ValueError("Could not extract features from audio file")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        if return_probability:
            return prediction, probability[1]  # Return AI probability
        return prediction
    
    def analyze_audio_detailed(self, audio_path):
        """
        Provide detailed analysis of audio file
        """
        print(f"\nAnalyzing: {audio_path}")
        print("=" * 50)
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
        
        # Get prediction
        try:
            prediction, ai_prob = self.predict(audio_path)
            
            result = "AI-GENERATED" if prediction == 1 else "REAL HUMAN"
            confidence = ai_prob if prediction == 1 else 1 - ai_prob
            
            print(f"Result: {result}")
            print(f"Confidence: {confidence:.2%}")
            print(f"AI Probability: {ai_prob:.2%}")
            
        except ValueError as e:
            print(f"Prediction error: {e}")
            ai_prob = 0.5
        
        # Additional audio analysis
        print("\n=== Audio Analysis ===")
        print(f"Language: {self.language.upper()}")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        print(f"Sample Rate: {sr} Hz")
        print(f"Max Amplitude: {np.max(np.abs(audio)):.4f}")
        print(f"RMS Energy: {np.sqrt(np.mean(audio**2)):.6f}")
        
        # Extract and display key features
        features = self.extract_features(audio_path)
        if features:
            print("\n=== Key Feature Values ===")
            key_features = [
                'zcr', 'spectral_centroid', 'spectral_bandwidth',
                'harmonic_ratio', 'silence_ratio', 'phase_std'
            ]
            
            for feat in key_features:
                if feat in features:
                    print(f"{feat.replace('_', ' ').title()}: {features[feat]:.6f}")
        
        return prediction, ai_prob if 'ai_prob' in locals() else 0.5
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'language': self.language
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', [])
            self.language = model_data.get('language', 'english')
            print(f"Model loaded from {filepath}")
            print(f"Language: {self.language.upper()}")
            print(f"Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"Error loading model: {e}")

# ============================================================================
# Multi-Language Training System
# ============================================================================

class MultiLanguageAudioDetector:
    """
    Manages training and inference for multiple languages
    """
    
    def __init__(self, models_dir='models'):
        """Initialize multi-language detector"""
        self.models_dir = models_dir
        self.models = {}
        self.languages = ['english', 'hindi', 'malayalam', 'telugu', 'tamil']
        self.load_all_models()
    
    def create_directory_structure(self):
        """Create the models directory structure for all languages"""
        print("\nCreating directory structure...")
        
        # Create main models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create subdirectories for each language
        for language in self.languages:
            lang_dir = os.path.join(self.models_dir, language)
            os.makedirs(lang_dir, exist_ok=True)
            print(f"✓ Created: {lang_dir}")
        
        # Create metadata file
        metadata = {
            'languages': self.languages,
            'models_dir': self.models_dir,
            'trained_models': {}
        }
        
        metadata_file = os.path.join(self.models_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Created metadata file: {metadata_file}")
        print("\nDirectory structure created successfully!")
    
    def train_all_languages(self, dataset_dir, model_type='rf'):
        """
        Train models for all languages
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
        print("MULTI-LANGUAGE MODEL TRAINING")
        print("="*60)
        
        # Create directory structure first
        self.create_directory_structure()
        
        trained_models = {}
        
        for language in self.languages:
            print(f"\n{'='*60}")
            print(f"Training model for: {language.upper()}")
            print(f"{'='*60}")
            
            # Get paths for this language
            real_dir = os.path.join(dataset_dir, 'real', language)
            ai_dir = os.path.join(dataset_dir, 'ai', language)
            
            # Check if directories exist
            if not os.path.exists(real_dir):
                print(f"⚠ Real audio directory not found: {real_dir}")
                continue
            
            if not os.path.exists(ai_dir):
                print(f"⚠ AI audio directory not found: {ai_dir}")
                continue
            
            # Collect audio files
            real_files = []
            ai_files = []
            
            for ext in ['*.wav', '*.mp3', '*.flac']:
                real_files.extend(glob.glob(os.path.join(real_dir, ext)))
                ai_files.extend(glob.glob(os.path.join(ai_dir, ext)))
            
            print(f"Found {len(real_files)} real audio files")
            print(f"Found {len(ai_files)} AI audio files")
            
            if len(real_files) == 0 or len(ai_files) == 0:
                print(f"⚠ Skipping {language} - insufficient training data")
                continue
            
            # Train model for this language
            detector = AIGeneratedAudioDetector(language=language)
            model_path = os.path.join(self.models_dir, language, f'ai_audio_model_{language}.pkl')
            
            detector.train(real_files, ai_files, model_type=model_type, save_path=model_path)
            
            self.models[language] = detector
            trained_models[language] = model_path
            
            print(f"✓ Model trained and saved for {language}")
        
        # Update metadata
        self.update_metadata(trained_models)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Trained models: {len(trained_models)}")
        print(f"{'='*60}")
        
        return trained_models
    
    def load_all_models(self):
        """Load all trained models from the models directory"""
        self.models = {}
        
        for language in self.languages:
            model_path = os.path.join(self.models_dir, language, f'ai_audio_model_{language}.pkl')
            
            if os.path.exists(model_path):
                try:
                    detector = AIGeneratedAudioDetector(model_path=model_path, language=language)
                    self.models[language] = detector
                    print(f"✓ Loaded model for {language}")
                except Exception as e:
                    print(f"✗ Error loading model for {language}: {e}")
            else:
                print(f"⚠ Model not found for {language}: {model_path}")
    
    def predict_with_language(self, audio_path, language):
        """Predict using language-specific model"""
        if language not in self.models:
            raise ValueError(f"Model for {language} not loaded. Available: {list(self.models.keys())}")
        
        return self.models[language].analyze_audio_detailed(audio_path)
    
    def predict_all_languages(self, audio_path):
        """Predict using all available models and return results"""
        print(f"\nAnalyzing: {audio_path}")
        print("=" * 60)
        
        results = {}
        
        for language, detector in self.models.items():
            try:
                pred, prob = detector.predict(audio_path, return_probability=True)
                results[language] = {
                    'prediction': 'AI' if pred == 1 else 'Real',
                    'ai_probability': prob,
                    'confidence': prob if pred == 1 else 1 - prob
                }
                print(f"{language.upper():15} -> {results[language]['prediction']:10} ({results[language]['confidence']:.2%} confidence)")
            except Exception as e:
                print(f"{language.upper():15} -> Error: {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def update_metadata(self, trained_models):
        """Update the metadata file with trained model information"""
        metadata_file = os.path.join(self.models_dir, 'metadata.json')
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'languages': self.languages, 'models_dir': self.models_dir}
        
        metadata['trained_models'] = trained_models
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

# ============================================================================
# Utility functions
# ============================================================================

def create_sample_dataset_info():
    """Show dataset structure for multi-language training"""
    print("""
    MULTI-LANGUAGE TRAINING DATASET STRUCTURE
    ==========================================
    
    Your dataset should be organized as:
    
    dataset/
    ├── ai/
    │   ├── english/
    │   │   ├── ai_speech_1.wav
    │   │   ├── ai_speech_2.wav
    │   │   └── ...
    │   ├── hindi/
    │   │   ├── ai_speech_1.wav
    │   │   ├── ai_speech_2.wav
    │   │   └── ...
    │   ├── malayalam/
    │   │   ├── ai_speech_1.wav
    │   │   └── ...
    │   ├── telugu/
    │   │   ├── ai_speech_1.wav
    │   │   └── ...
    │   └── tamil/
    │       ├── ai_speech_1.wav
    │       └── ...
    │
    └── real/
        ├── english/
        │   ├── human_speech_1.wav
        │   ├── human_speech_2.wav
        │   └── ...
        ├── hindi/
        │   ├── human_speech_1.wav
        │   └── ...
        ├── malayalam/
        │   ├── human_speech_1.wav
        │   └── ...
        ├── telugu/
        │   ├── human_speech_1.wav
        │   └── ...
        └── tamil/
            ├── human_speech_1.wav
            └── ...
    
    GENERATED MODELS STRUCTURE
    ==========================
    
    After training, models will be organized as:
    
    models/
    ├── english/
    │   ├── ai_audio_model_english.pkl
    │   └── features_english.json
    ├── hindi/
    │   ├── ai_audio_model_hindi.pkl
    │   └── features_hindi.json
    ├── malayalam/
    │   ├── ai_audio_model_malayalam.pkl
    │   └── features_malayalam.json
    ├── telugu/
    │   ├── ai_audio_model_telugu.pkl
    │   └── features_telugu.json
    ├── tamil/
    │   ├── ai_audio_model_tamil.pkl
    │   └── features_tamil.json
    │
    └── metadata.json
    
    AUDIO FILE REQUIREMENTS
    =======================
    - Format: WAV, MP3, or FLAC
    - Sample Rate: 16 kHz (recommended)
    - Mono or Stereo: Both supported
    - Minimum files: 50+ per category (real/ai) per language for good results
    - Recommended: 100+ files per category per language
    
    TRAINING COMMAND
    ================
    python app.py --train_multilang --dataset dataset
    """)

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Audio Detection System - Multi-Language Support')
    
    # Multi-language arguments
    parser.add_argument('--train_multilang', action='store_true', 
                       help='Train models for all languages')
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save language-specific models')
    
    # Single language arguments
    parser.add_argument('--audio', type=str, help='Path to audio file to analyze')
    parser.add_argument('--language', type=str, default='english',
                       help='Language for analysis (english, hindi, malayalam, telugu, tamil)')
    parser.add_argument('--analyze_all_langs', action='store_true',
                       help='Analyze audio with all available language models')
    
    parser.add_argument('--folder', type=str, help='Folder containing audio files')
    parser.add_argument('--create_dataset_info', action='store_true',
                       help='Show dataset structure information')
    parser.add_argument('--model_type', type=str, default='rf',
                       help='Model type: rf (RandomForest) or svm (SVM)')
    
    args = parser.parse_args()
    
    if args.create_dataset_info:
        create_sample_dataset_info()
    
    elif args.train_multilang:
        # Train models for all languages
        multi_detector = MultiLanguageAudioDetector(models_dir=args.models_dir)
        multi_detector.train_all_languages(args.dataset, model_type=args.model_type)
    
    elif args.analyze_all_langs and args.audio:
        # Analyze with all language models
        multi_detector = MultiLanguageAudioDetector(models_dir=args.models_dir)
        results = multi_detector.predict_all_languages(args.audio)
    
    elif args.audio:
        # Single language analysis
        detector = AIGeneratedAudioDetector(language=args.language)
        model_path = os.path.join(args.models_dir, args.language, f'ai_audio_model_{args.language}.pkl')
        
        if os.path.exists(model_path):
            detector.load_model(model_path)
            detector.analyze_audio_detailed(args.audio)
        else:
            print(f"Model for {language} not found at {model_path}")
            print(f"Please train the model first using: python app.py --train_multilang --dataset dataset")
    
    else:
        print("\n" + "="*60)
        print("AI AUDIO DETECTION SYSTEM - MULTI-LANGUAGE SUPPORT")
        print("="*60)
        print("\n📚 USAGE EXAMPLES:")
        print("\n1. View dataset structure:")
        print("   python app.py --create_dataset_info")
        
        print("\n2. Train models for all languages:")
        print("   python app.py --train_multilang --dataset dataset --models_dir models")
        
        print("\n3. Analyze single audio with specific language model:")
        print("   python app.py --audio path/to/audio.wav --language hindi")
        
        print("\n4. Analyze audio with all available language models:")
        print("   python app.py --audio path/to/audio.wav --analyze_all_langs")
        
        print("\n5. Analyze with different model type (SVM):")
        print("   python app.py --train_multilang --dataset dataset --model_type svm")
        
        print("\n📁 SUPPORTED LANGUAGES:")
        print("   - English")
        print("   - Hindi")
        print("   - Malayalam")
        print("   - Telugu")
        print("   - Tamil")
        
        print("\n" + "="*60 + "\n")
