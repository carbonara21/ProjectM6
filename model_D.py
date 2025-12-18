import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

landmarks = 33  # Number of landmarks (not multiplied by 2)
expected_landmarks = landmarks * 3  # Each landmark has 3 coordinates (x, y, z)

dataset_dir = r"C:\\Users\\s2887800\\PycharmProjects\\ProjectM6\\data\\D_S"
categories = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]

data = []
labels = []

for dir_ in os.listdir(dataset_dir):
    dir_path = os.path.join(dataset_dir, dir_)
    if not os.path.isdir(dir_path):
        continue

    print("Processing:", dir_)

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            print(f"No pose landmarks detected: {img_name}")
            continue

        x_vals = []
        y_vals = []
        z_vals = []
        data_norm = []

        # Collect all x, y, z coordinates
        for lm in results.pose_landmarks.landmark:
            x_vals.append(lm.x)
            y_vals.append(lm.y)
            z_vals.append(lm.z)

        # Min-max normalization for all coordinates
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)

        # Normalize each landmark (x, y, z) within the range [0, 1]
        for x, y, z in zip(x_vals, y_vals, z_vals):
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)
            z_norm = (z - z_min) / (z_max - z_min)
            data_norm.extend([x_norm, y_norm, z_norm])

        # Only add data if the number of normalized coordinates matches the expected number
        if len(data_norm) == expected_landmarks:
            data.append(data_norm)
            labels.append(dir_)

# Prepare data for training
x = np.asarray(data, dtype=np.float32)
y = np.asarray(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, random_state=42)

# Define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(expected_landmarks,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=75, batch_size=16, verbose=1, validation_split=0.1)
model.summary()

# Convert model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("M_D_S.tflite", "wb") as f:
    f.write(tflite_model)
