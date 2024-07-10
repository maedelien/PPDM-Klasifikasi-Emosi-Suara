import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and necessary preprocessing tools
model = load_model('best_cnn_model.keras')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

# Function to extract features from an audio file
class FeatureExtractor:
    def __init__(self, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def zcr(self, data):
        return librosa.feature.zero_crossing_rate(data, frame_length=self.frame_length, hop_length=self.hop_length).flatten()

    def rmse(self, data):
        return librosa.feature.rms(y=data, frame_length=self.frame_length, hop_length=self.hop_length).flatten()

    def mfcc(self, data, sr, n_mfcc=13, flatten=True):
        mfcc_features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=self.hop_length)
        return mfcc_features.T.flatten() if flatten else mfcc_features.T

    def chroma(self, data, sr):
        chroma_features = librosa.feature.chroma_stft(y=data, sr=sr, hop_length=self.hop_length)
        return chroma_features.T.flatten()

    def spectral_contrast(self, data, sr):
        contrast_features = librosa.feature.spectral_contrast(y=data, sr=sr, hop_length=self.hop_length)
        return contrast_features.T.flatten()

    def mel_spectrogram(self, data, sr):
        mel_features = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=self.hop_length)
        return librosa.power_to_db(mel_features).flatten()

    def extract_features(self, data, sr):
        zcr_features = self.zcr(data)
        rmse_features = self.rmse(data)
        mfcc_features = self.mfcc(data, sr)
        chroma_features = self.chroma(data, sr)
        spectral_contrast_features = self.spectral_contrast(data, sr)
        mel_spectrogram_features = self.mel_spectrogram(data, sr)
        return np.concatenate([zcr_features, rmse_features, mfcc_features, chroma_features, spectral_contrast_features, mel_spectrogram_features])

# Function to preprocess and predict emotion from audio file
def predict_emotion(audio_file):
    data, sr = librosa.load(audio_file, duration=2.5, offset=0.6)
    extractor = FeatureExtractor()
    features = extractor.extract_features(data, sr)
    features = scaler.transform([features])
    features = features.reshape((features.shape[0], features.shape[1], 1, 1))
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    return emotion

# Streamlit app layout
st.set_page_config(page_title="Emotion Classification", page_icon="🎤")
st.title("Welcome to Emotion Classification Web")
st.caption("This web-based application classifies audio recordings into one of eight emotions. Please upload an audio file in .wav format.")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    button = st.button("Classify")
    if button:
        emotion = predict_emotion(uploaded_file)
        st.markdown(f"<h3 style='text-align: center;'>Predicted Emotion: {emotion}</h3>", unsafe_allow_html=True)
