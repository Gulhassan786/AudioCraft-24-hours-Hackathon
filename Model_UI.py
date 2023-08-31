import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import pandas as pd
import numpy as np
import librosa
import glob
import streamlit as st
from PIL import Image
import time
import humanize
import requests
from bs4 import BeautifulSoup
import csv
import io
from IPython.display import Audio
model = hub.load('https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/1')
labels_path = hub.resolve('https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/1') + "/assets/label.csv"
def frame_audio(
      audio_array: np.ndarray,
      window_size_s: float = 5.0,
      hop_size_s: float = 5.0,
      sample_rate = 32000,
      ) -> np.ndarray:
    with tf.name_scope("frame_audio"):
       
        if window_size_s is None or window_size_s < 0:
            return audio_array[np.newaxis, :]
        frame_length = int(window_size_s * sample_rate)
        hop_length = int(hop_size_s * sample_rate)
        framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=False)
        return framed_audio
# Find the name of the class with the top score when mean-aggregated across frames.

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    with open(class_map_csv_text) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        class_names = [mid for mid, desc in csv_reader]
        return class_names[1:]
train_metadata = pd.read_csv("E:/Bird-Chripping/train_metadata/train_metadata.csv")
train_metadata.head()
competition_classes = sorted(train_metadata.primary_label.unique())

forced_defaults = 0
competition_class_map = []
for c in competition_classes:
    try:
        i = classes.index(c)
        competition_class_map.append(i)
    except:
        competition_class_map.append(0)
        forced_defaults += 1

## note that the bird classifier classifies a much larger set of birds than the
## competition, so we need to load the model's set of class names or else our
## indices will be off.
classes = class_names_from_csv(labels_path)

def ensure_sample_rate(waveform, original_sample_rate,
                       desired_sample_rate=32000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        waveform = tfio.audio.resample(waveform, original_sample_rate, desired_sample_rate)
    return desired_sample_rate, waveform
# Set the app theme color to lilac

st.set_page_config(page_title="Lets Save our Eco-system", page_icon="🐦")
# Load and resize header image
# E:\Bird-Chripping\assets\robott.jpeg
# header_image = Image.open("E:\Bird-Chripping\assets\robott.jpeg")
# header_image = header_image.resize((500, 150))  # Resize the image to desired dimensions
# st.image(header_image, use_column_width=True)

# Title
st.title("Lets Save Our EcoSystem")
# Define the lilac color in hexadecimal format


# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
if audio_file:

    
    # Show loading bar
    progress_bar = st.progress(0)
    for percent_complete in range(101):
        time.sleep(0.02)  # Simulating file upload delay
        progress_bar.progress(percent_complete)
        
    st.success("Audio upload complete!")

    # Display audio file size
    audio_size = humanize.naturalsize(audio_file.size)
    st.write(f"Uploaded file size: {audio_size}")
    # Play audio
    st.audio(audio_file)

    # Display tick icon
    tick_icon = "✅"
    st.markdown(f"Uploaded: {tick_icon}")

    st.write("---")  # Separator
    # Add more content to the UI below if needed
    classes = class_names_from_csv(labels_path)
    # st.write(classes)
    audio, sample_rate = librosa.load(audio_file)

    sample_rate, wav_data = ensure_sample_rate(audio, sample_rate)

    fixed_tm = frame_audio(wav_data)


    # audio, sample_rate = librosa.load("E:/Bird-Chripping/train_audio/abethr1/XC128013.ogg")
    # sample_rate, wav_data = ensure_sample_rate(audio, sample_rate)
    # Audio(wav_data, rate=sample_rate)
    # fixed_tm = frame_audio(wav_data)
    logits, embeddings = model.infer_tf(fixed_tm[:1])
    probabilities = tf.nn.softmax(logits)
    argmax = np.argmax(probabilities)
    st.write('Audio  Compressed')
    # print(classes[argmax])
    # st.write(f"The audio is from the class {classes[argmax]} (element:{argmax} in the label.csv file)")

    st.write(f"The audio is from the class {classes[argmax]} (element:{argmax} in the label.csv file), with probability of {probabilities[0][argmax]}")
    # read line classes[argmax] from metadata.csv and extract common name from it:
    # Display information about the predicted bird species
    predicted_bird_info = train_metadata[train_metadata.primary_label == classes[argmax]]

# Extract the common name from the DataFrame:
    common_name = predicted_bird_info.common_name.values[0] if not predicted_bird_info.empty else None
  


   
    st.write(f'<p style="font-size: 15px; font-weight: bold; color: blue;">The predicted bird species is: <span style="color: green;font-size: 24px; font-weight: italic">{common_name}</span></p>', unsafe_allow_html=True)
    
 
    
    
    url= predicted_bird_info['url'].values[0] if not predicted_bird_info.empty else None
    if url is not None:
        st.write(f"More information about the bird can be found at: {url}")
    # st.write(url)