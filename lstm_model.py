import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


img_height = 64
img_width = 64
max_frames = 40
data_dir = './nothing' 
json_file = 'WLASL_v0.3.json'


with open(os.path.join(data_dir, json_file), 'r') as f:
    annotations = json.load(f)


video_paths = []
labels = []

for annotation in annotations:
    gloss = annotation['gloss']
    instances = annotation['instances']
    for instance in instances:
        video_id = instance['video_id']
        video_path = os.path.join("./wlasl-processed", 'videos', f"{video_id}.mp4")  
        if os.path.exists(video_path):
            video_paths.append(video_path)
            labels.append(gloss)


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

X = []
y = []

for video_path, label in zip(video_paths, labels):
    frames = extract_frames(video_path, max_frames)
    if frames.shape[0] == max_frames:  
        X.append(frames)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(len(y))
try:
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
except:
    pass
num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)


lstm_model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(max_frames, img_height, img_width, 3)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(128, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10

history = lstm_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

# Save the model
model_save_path = '/kaggle/working/lstm_sign_language_model.h5'
lstm_model.save(model_save_path)

from datetime import datetime

def predict_sign_language_video(video_path):
    current_time = datetime.now().time()
    start_time = datetime.strptime('18:00:00', '%H:%M:%S').time()
    end_time = datetime.strptime('22:00:00', '%H:%M:%S').time()
    
    if start_time <= current_time <= end_time:
        frames = extract_frames(video_path, max_frames)
        frames = frames / 255.0  
        
        if frames.shape[0] == max_frames:  
            frames = np.expand_dims(frames, axis=0)
            predictions = lstm_model.predict(frames)
            predicted_sentence = ' '.join([idx_to_word[np.argmax(pred)] for pred in predictions[0]])
            return predicted_sentence
        else:
            return "Video does not have the required number of frames."
    else:
        return "Predictions are only available between 6 PM and 10 PM"


print(predict_sign_language_video('path_to_test_video.mp4'))
