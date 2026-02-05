import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
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
    """
    
    def __init__(self, model_path=None):
        """Initialize the detector with or without a pre-trained model"""
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
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
        
        for i, audio_path in enumerate(audio_paths):
            features = self.extract_features(audio_path)
            if features:
                features_list.append(list(features.values()))
                if labels is not None:
                    valid_labels.append(labels[i])
                if i == 0:
                    self.feature_names = list(features.keys())
        
        X = np.array(features_list)
        if labels is not None:
            y = np.array(valid_labels)
            return X, y
        return X
    
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
        X, y = self.extract_features_batch(all_paths, all_labels)
        
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
        X = self.extract_features_batch([audio_path])
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
        audio, sr = librosa.load(audio_path, sr=16000)
        
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
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data.get('feature_names', [])
        print(f"Model loaded from {filepath}")
        print(f"Model type: {type(self.model).__name__}")

# ============================================================================
# Utility functions and sample usage
# ============================================================================

def create_sample_dataset():
    """
    Create instructions for building a sample dataset
    """
    print("""
    To train the model, you need:
    
    1. REAL HUMAN SPEECH:
       - Collect human voice recordings
       - Use datasets like:
         * LibriSpeech
         * Common Voice
         * VoxCeleb
    
    2. AI-GENERATED SPEECH:
       - Generate speech using TTS systems:
         * ElevenLabs
         * Google TTS
         * Amazon Polly
         * OpenAI Whisper TTS
    
    3. File structure:
       dataset/
         ├── real/
         │   ├── human1.wav
         │   ├── human2.wav
         │   └── ...
         └── ai/
             ├── ai1.wav
             ├── ai2.wav
             └── ...
    
    Audio files should be in WAV format, 16kHz sampling rate.
    """)

def batch_process(detector, audio_folder):
    """Process multiple audio files in a folder"""
    import glob
    
    audio_files = glob.glob(os.path.join(audio_folder, "*.wav")) + \
                  glob.glob(os.path.join(audio_folder, "*.mp3")) + \
                  glob.glob(os.path.join(audio_folder, "*.flac"))
    
    results = []
    
    for audio_file in audio_files:
        try:
            pred, prob = detector.analyze_audio_detailed(audio_file)
            results.append({
                'file': os.path.basename(audio_file),
                'prediction': 'AI' if pred == 1 else 'Real',
                'ai_probability': prob
            })
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return pd.DataFrame(results)

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Audio Detection System')
    parser.add_argument('--audio', type=str, help='Path to audio file to analyze')
    parser.add_argument('--folder', type=str, help='Folder containing audio files')
    parser.add_argument('--train', action='store_true', help='Train mode')
    parser.add_argument('--real_dir', type=str, help='Directory with real audio files')
    parser.add_argument('--ai_dir', type=str, help='Directory with AI audio files')
    parser.add_argument('--model', type=str, default='ai_audio_model.pkl', 
                       help='Model file path')
    parser.add_argument('--create_dataset_info', action='store_true',
                       help='Show how to create dataset')
    
    args = parser.parse_args()
    
    detector = AIGeneratedAudioDetector()
    
    if args.create_dataset_info:
        create_sample_dataset()
    
    elif args.train:
        if not args.real_dir or not args.ai_dir:
            print("Please provide both --real_dir and --ai_dir for training")
        else:
            # Collect all audio files from directories
            real_files = []
            ai_files = []
            
            for ext in ['*.wav', '*.mp3', '*.flac']:
                real_files.extend(glob.glob(os.path.join(args.real_dir, ext)))
                ai_files.extend(glob.glob(os.path.join(args.ai_dir, ext)))
            
            if len(real_files) == 0 or len(ai_files) == 0:
                print(f"Found {len(real_files)} real files and {len(ai_files)} AI files")
                print("Need both real and AI audio files for training")
            else:
                detector.train(real_files, ai_files, model_type='rf', save_path=args.model)
    
    elif args.audio:
        # Load existing model or create a simple one for demo
        if os.path.exists(args.model):
            detector.load_model(args.model)
        else:
            print(f"Model file {args.model} not found.")
            print("Please train a model first or run with --train flag")
            print("Using basic heuristics for demo...")
            
            # Create a simple demo model (not accurate, just for demo)
            detector.model = RandomForestClassifier()
            detector.scaler = StandardScaler()
        
        detector.analyze_audio_detailed(args.audio)
    
    elif args.folder:
        if os.path.exists(args.model):
            detector.load_model(args.model)
        else:
            print(f"Model file {args.model} not found.")
            print("Please train a model first with --train flag")
            exit(1)
        
        results = batch_process(detector, args.folder)
        print("\n=== Batch Results ===")
        print(results.to_string())
        
        # Summary
        ai_count = (results['prediction'] == 'AI').sum()
        total_count = len(results)
        print(f"\nSummary: {ai_count}/{total_count} files detected as AI-generated ({ai_count/total_count:.1%})")
    
    else:
        print("\nAI Audio Detection System")
        print("=" * 50)
        print("\nUsage examples:")
        print("1. Analyze single audio file:")
        print("   python ai_audio_detector.py --audio path/to/audio.wav")
        print("\n2. Train new model:")
        print("   python ai_audio_detector.py --train --real_dir real_audio/ --ai_dir ai_audio/")
        print("\n3. Batch process folder:")
        print("   python ai_audio_detector.py --folder audio_samples/")
        print("\n4. Show dataset creation info:")
        print("   python ai_audio_detector.py --create_dataset_info")