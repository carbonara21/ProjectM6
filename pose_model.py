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

# Arm landmarks
arm_landmarks = [11, 12, 13, 14, 15, 16]

# Now 3D (x, y, z)
expected_landmarks = len(arm_landmarks) * 3

dataset_dir = r"C:\\Users\\s2887800\\PycharmProjects\\ProjectM6\\data\\BC_D"
categories = ["BC_D_1", "BC_D_2", "BC_D_3",
              "BC_D_I1", "BC_D_I2", "BC_D_I3"]

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

        # Torso length including z for proper 3D scaling
        torso_length = np.sqrt(
            (center_x - torso_center_x) ** 2 +
            (center_y - torso_center_y) ** 2 +
            (center_z - torso_center_z) ** 2
        ) + 1e-6

        # Collect normalized 3D arm coordinates
        data_raw = []
        for idx in arm_landmarks:
            x_norm = (lm[idx].x - center_x) / torso_length
            y_norm = (lm[idx].y - center_y) / torso_length
            z_norm = (lm[idx].z - center_z) / torso_length
            data_raw.extend([x_norm, y_norm, z_norm])

        data.append(data_raw)
        labels.append(dir_)

# ------------------- Prepare data -------------------
x = np.asarray(data, dtype=np.float32)
y = np.asarray(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.3, random_state=42
)

# ------------------- Build model -------------------
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

# ------------------- Convert to TFLite -------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("M_BC_D.tflite", "wb") as f:
    f.write(tflite_model)
