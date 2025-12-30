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

# -----------------------
# MediaPipe
# -----------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

dataset_dir = "/Users/felipecarbone/PycharmProjects/ProjectM6/data/D_S"
categories = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]

data = []
labels = []

# -----------------------
# 2D Angle function (NO Z)
# -----------------------
def calculate_angle(a, b, c):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0))

# -----------------------
# Dataset loop
# -----------------------
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

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            continue

        lm = results.pose_landmarks.landmark

        # -----------------------
        # Right-side 2D angles ONLY
        # -----------------------
        angles = [
            calculate_angle(
                (lm[12].x, lm[12].y),
                (lm[14].x, lm[14].y),
                (lm[16].x, lm[16].y)
            ),  # elbow

            calculate_angle(
                (lm[14].x, lm[14].y),
                (lm[12].x, lm[12].y),
                (lm[24].x, lm[24].y)
            ),  # shoulder

            calculate_angle(
                (lm[12].x, lm[12].y),
                (lm[24].x, lm[24].y),
                (lm[26].x, lm[26].y)
            ),  # hip

            calculate_angle(
                (lm[24].x, lm[24].y),
                (lm[26].x, lm[26].y),
                (lm[28].x, lm[28].y)
            ),  # knee

            calculate_angle(
                (lm[26].x, lm[26].y),
                (lm[28].x, lm[28].y),
                (lm[32].x, lm[32].y)
            )   # ankle
        ]

        # Normalize angles → [0,1]
        angles_norm = [a / np.pi for a in angles]

        data.append(angles_norm)
        labels.append(dir_)

# -----------------------
# Prepare data
# -----------------------
x = np.asarray(data, dtype=np.float32)
y = np.asarray(labels)

encoder = LabelEncoder()
y_encoded = to_categorical(encoder.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.3, random_state=42
)

# -----------------------
# Model
# -----------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=75,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# -----------------------
# Export TFLite
# -----------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("M_D_S2.tflite", "wb") as f:
    f.write(tflite_model)
