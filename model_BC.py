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

landmarks = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28]
expected_landmarks = len(landmarks) * 3

dataset_dir = r"/Users/felipecarbone/PycharmProjects/ProjectM6/data/BC_D"
categories = ["BC_D_1", "BC_D_2", "BC_D_I1", "BC_D_I2"]

data = []
labels = []

def augment_landmarks(sample,xy_noise=0.005, z_noise=0.01, scale_range=(0.97, 1.03), rotate_deg=2):
    aug = sample.reshape(-1, 3).copy()

    # Small XY noise (pose jitter)
    aug[:, :2] += np.random.normal(0, xy_noise, aug[:, :2].shape)

    # Slight depth noise (camera distance variation)
    aug[:, 2] += np.random.normal(0, z_noise, aug[:, 2].shape)

    # Mild scaling
    scale = np.random.uniform(*scale_range)
    aug *= scale

    # Very small in-plane rotation (camera tilt)
    theta = np.deg2rad(np.random.uniform(-rotate_deg, rotate_deg))
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    aug = aug @ rot.T

    return aug.flatten()

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

        img_rgb = cv2.cvtColor(cv2.resize(img, (640, 480)), cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            continue

        lm = results.pose_landmarks.landmark

        # Center of shoulders
        left_shoulder = lm[11]
        right_shoulder = lm[12]
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        center_z = (left_shoulder.z + right_shoulder.z) / 2

        # Center of hips
        left_hip = lm[23]
        right_hip = lm[24]
        torso_center_x = (left_hip.x + right_hip.x) / 2
        torso_center_y = (left_hip.y + right_hip.y) / 2
        torso_center_z = (left_hip.z + right_hip.z) / 2

        # Torso length (3D)
        torso_length = np.sqrt(
            (center_x - torso_center_x) ** 2 +
            (center_y - torso_center_y) ** 2 +
            (center_z - torso_center_z) ** 2
        ) + 1e-6

        # Normalized 3D landmarks
        data_raw = []
        for idx in landmarks:
            x_norm = (lm[idx].x - center_x) / torso_length
            y_norm = (lm[idx].y - center_y) / torso_length
            z_norm = (lm[idx].z - center_z) / torso_length
            data_raw.extend([x_norm, y_norm, z_norm])

        # Original sample
        data.append(data_raw)
        labels.append(dir_)

        # Augmented samples
        for _ in range(2):  # number of augmentations per sample
            aug_sample = augment_landmarks(np.array(data_raw))
            data.append(aug_sample)
            labels.append(dir_)

x = np.asarray(data, dtype=np.float32)
y = np.asarray(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

x_train, x_test, y_train, y_test = train_test_split( x, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# ------------------- Model -------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(expected_landmarks,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=75, batch_size=16, validation_split=0.1, verbose=1)

model.summary()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("M_BC_D2.tflite", "wb") as f:
    f.write(tflite_model)

