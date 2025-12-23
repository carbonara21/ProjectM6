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

RIGHT_LANDMARK_IDS = [12, 14, 16, 24, 26, 28, 32]
NUM_LANDMARKS = len(RIGHT_LANDMARK_IDS)   # 7
NUM_ANGLES = 5
EXPECTED_FEATURES = (NUM_LANDMARKS * 3) + NUM_ANGLES  # 26


dataset_dir = "/Users/felipecarbone/PycharmProjects/ProjectM6/data/D_S"
categories = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]

data = []
labels = []


def calculate_angle(a, b, c):
    """
    Returns angle at point b (radians)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


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
            continue

        # -----------------------
        # Landmark normalization
        # -----------------------
        x_vals, y_vals, z_vals = [], [], []

        right_landmarks = [results.pose_landmarks.landmark[i] for i in RIGHT_LANDMARK_IDS]

        for lm in right_landmarks:
            x_vals.append(lm.x)
            y_vals.append(lm.y)
            z_vals.append(lm.z)

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)

        data_norm = []
        for x, y, z in zip(x_vals, y_vals, z_vals):
            x_norm = (x - x_min) / (x_max - x_min + 1e-6)
            y_norm = (y - y_min) / (y_max - y_min + 1e-6)
            z_norm = (z - z_min) / (z_max - z_min + 1e-6)
            data_norm.extend([x_norm, y_norm, z_norm])

        # -----------------------
        # Angle features (RAW coords)
        # -----------------------
        lm_xyz = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        angles = [
            calculate_angle(lm_xyz[12], lm_xyz[14], lm_xyz[16]),  # right elbow
            calculate_angle(lm_xyz[14], lm_xyz[12], lm_xyz[24]),  # right shoulder
            calculate_angle(lm_xyz[12], lm_xyz[24], lm_xyz[26]),  # right hip
            calculate_angle(lm_xyz[24], lm_xyz[26], lm_xyz[28]),  # right knee
            calculate_angle(lm_xyz[26], lm_xyz[28], lm_xyz[32])  # right ankle
        ]

        # Normalize angles → [0, 1]
        angles_norm = [a / np.pi for a in angles]

        data_norm.extend(angles_norm)

        # -----------------------
        # Final check
        # -----------------------
        if len(data_norm) == EXPECTED_FEATURES:
            data.append(data_norm)
            labels.append(dir_)

# -----------------------
# Training prep
# -----------------------
x = np.asarray(data, dtype=np.float32)
y = np.asarray(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.3, random_state=42
)

# -----------------------
# Model
# -----------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(EXPECTED_FEATURES,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
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

model.summary()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("M_D_S.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model saved as M_D_S.tflite")