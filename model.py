import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
import pickle
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)

expected_landmarks = 33 * 2
num_landmarks = 33

dataset_dir = r"C:\Users\s2887800\PycharmProjects\PythonProject1\data"

categories = ["1 Arms streched", "2 Curl halfway", "3 Full bicep curl", "4 Arms too open", "5 Elbows too in (Halfway)", "6 Elbows too in (Beggining)"]

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

        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            print(f"No pose landmarks detected: {img_name}")
            continue

        x_vals = []
        y_vals = []
        data_norm = []

        for lm in results.pose_landmarks.landmark:
            x_vals.append(lm.x)
            y_vals.append(lm.y)

        # Min-max normalization per sample
        x_min, y_min = min(x_vals), min(y_vals)

        for (x, y) in zip(x_vals, y_vals):
            data_norm.append(x - x_min)
            data_norm.append(y - y_min)

        if len(data_norm) == expected_landmarks:
            data.append(data_norm)
            labels.append(dir_)

x = np.asarray(data)
y = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, shuffle=True, random_state=100)

base_model = SVC(kernel='linear', class_weight='balanced')
base_model.fit(x_train, y_train)
y_pred = base_model.predict(x_test)

print("\nBASE MODEL:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': [0.00001, 0.0001, 0.001]
}

grid = GridSearchCV(SVC(class_weight='balanced'),
                    param_grid=param_grid,
                    cv=5,
                    verbose=3)
grid.fit(x_train, y_train)

print("\nBest parameters:", grid.best_params_)
best_model = grid.best_estimator_

y_pred_best = best_model.predict(x_test)

print("\nBEST MODEL:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print("Precision:", precision_score(y_test, y_pred_best, average='macro'))
print("Recall:", recall_score(y_test, y_pred_best, average='macro'))


plt.figure(figsize=(10, 8))
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_best)
sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Full-Body Pose – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


train_sizes, train_scores, test_scores = learning_curve(
    best_model, x_train, y_train, cv=3
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
plt.legend()
plt.title("Learning Curve – Full Body Pose")
plt.show()

with open("pose_model_full_body.p", "wb") as f:
    pickle.dump({'pose_model': best_model}, f)

print("Model saved as pose_model_full_body.p")
