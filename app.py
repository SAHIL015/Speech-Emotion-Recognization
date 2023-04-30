from keras.models import load_model
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf
import io
# Load the model
# replace 'my_model.h5' with your own saved model
model = load_model('my_model.h5')

# Define emotions
emotions = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Define function to extract MFCC features


def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# Define Streamlit app
st.title('Emotion Recognition')
st.write('Upload a .wav file to recognize the emotion')

# Get file from user
uploaded_file = st.file_uploader('Choose a .wav file', type='wav')

# If file is uploaded
if uploaded_file is not None:
    # Read the audio file as numpy array
    audio_bytes = uploaded_file.read()
    audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Extract MFCC features
    mfcc = extract_mfcc(audio_np)
    mfcc = np.reshape(mfcc, newshape=(1, 40, 1))

    # Make prediction using the model
    predictions = model.predict(mfcc)
    emotion = emotions[np.argmax(predictions[0])]

    # Display the predicted emotion
    st.write('Predicted emotion:', emotion)