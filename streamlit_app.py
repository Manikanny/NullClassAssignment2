import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('lstm_sign_language_model.h5')

img_height = 64
img_width = 64
max_frames = 40

# Define a function to extract frames from a video
def extract_frames(video_path, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_height, img_width))
        frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)


def process_image(image):
    image = cv2.resize(image, (img_height, img_width))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


st.title("Sign Language Recognition")


uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    if uploaded_file.type == "video/mp4":
       
        with open('temp_video.mp4', 'wb') as f:
            f.write(file_bytes)
        
     
        frames = extract_frames('temp_video.mp4', max_frames)
        if frames.shape[0] == max_frames:
            frames = np.expand_dims(frames, axis=0)
            predictions = model.predict(frames)
            predicted_label = np.argmax(predictions, axis=1)
            st.write(f"Predicted Sign: {predicted_label[0]}")
        else:
            st.write("The video is too short. Please upload a video with at least 40 frames.")

    elif uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]:
        image = cv2.imdecode(file_bytes, 1)
        processed_image = process_image(image)
        predictions = model.predict(processed_image)
        predicted_label = np.argmax(predictions, axis=1)
        st.write(f"Predicted Sign: {predicted_label[0]}")
        st.image(image, caption='Uploaded Image', use_column_width=True)
