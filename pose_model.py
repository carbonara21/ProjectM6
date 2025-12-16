import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

expected_landmarks = 33 * 2
dataset_dir = r"C:\\Users\\s2887800\\PycharmProjects\\ProjectM6\\data\\BC_D"

categories = ["BC_D_1", "BC_D_2", "BC_D_3",
              "BC_D_I1", "BC_D_I2", "BC_D_I3"]

data, labels = [], []

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

        x_vals, y_vals = [], []
        for lm in results.pose_landmarks.landmark:
            x_vals.append(lm.x)
            y_vals.append(lm.y)

        x_min, y_min = min(x_vals), min(y_vals)
        data_norm = []
        for (x, y) in zip(x_vals, y_vals):
            data_norm.append(x - x_min)
            data_norm.append(y - y_min)

        if len(data_norm) == expected_landmarks:
            data.append(data_norm)
            labels.append(dir_)

x = np.asarray(data)
y = np.asarray(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)   # strings → integers
y_encoded = to_categorical(y_encoded)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.3, random_state=42
)


model = Sequential([
    Dense(128, activation='relu', input_shape=(expected_landmarks,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=75, batch_size=16, verbose=1, validation_split=0.1)
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open("M_BC_D.tflite", "wb") as f:
    f.write(tflite_model)
