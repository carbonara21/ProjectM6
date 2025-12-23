import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pygame
from collections import deque



def _transform_landmarks(pose_landmarks_list: list, landmark_buffers) -> list:
    transformed_landmarks_list = []

    if not pose_landmarks_list:
        return transformed_landmarks_list

    for pose_landmarks in pose_landmarks_list:
        transformed_landmarks = []
        for i, landmark in enumerate(pose_landmarks):
            x, y, z, v, p = landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence
            x_t, y_t, z_t, v_t, p_t = sma_transform_func(x, y, z, v, p, landmark_buffers, idx=i)
            transformed_landmarks.append((x_t, y_t, z_t, v_t, p_t))

        transformed_landmarks_list.append(transformed_landmarks)
    return transformed_landmarks_list

def sma_transform_func(x, y, z, visibility, presence, landmark_buffers, idx):
    buffer = landmark_buffers[idx]
    buffer.append([x, y, z])
    if len(buffer) == buffer.maxlen:
        mean_x, mean_y, mean_z = np.mean(buffer, axis=0)
        return mean_x, mean_y, mean_z, visibility, presence
    else:
        return x, y, z, visibility, presence
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


RIGHT_LANDMARK_IDS = [12, 14, 16, 24, 26, 28, 32]
EXPECTED_FEATURES = 26

CATEGORIES = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]


def start_pose_recognition():
    interpreter = tf.lite.Interpreter(model_path="M_D_S.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    landmark_buffers = [deque(maxlen=5) for _ in range(33)]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            pose_landmarks_list = [results.pose_landmarks.landmark]
            smoothed_landmarks = _transform_landmarks(pose_landmarks_list, landmark_buffers)[0]

            # =======================
            # RIGHT-SIDE FEATURE EXTRACTION ONLY
            # =======================
            right_landmarks = [smoothed_landmarks[i] for i in RIGHT_LANDMARK_IDS]

            x_vals, y_vals, z_vals = [], [], []
            data_norm = []

            for lm in right_landmarks:
                x_vals.append(lm[0])
                y_vals.append(lm[1])
                z_vals.append(lm[2])

            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            z_min, z_max = min(z_vals), max(z_vals)

            for x, y, z in zip(x_vals, y_vals, z_vals):
                x_norm = 0.0 if x_max - x_min == 0 else (x - x_min) / (x_max - x_min)
                y_norm = 0.0 if y_max - y_min == 0 else (y - y_min) / (y_max - y_min)
                z_norm = 0.0 if z_max - z_min == 0 else (z - z_min) / (z_max - z_min)
                data_norm.extend([x_norm, y_norm, z_norm])

            # =======================
            # RIGHT-SIDE ANGLES ONLY
            # =======================
            lm_xyz = [(lm[0], lm[1], lm[2]) for lm in smoothed_landmarks]

            angles = [
                calculate_angle(lm_xyz[12], lm_xyz[14], lm_xyz[16]),  # right elbow
                calculate_angle(lm_xyz[14], lm_xyz[12], lm_xyz[24]),  # right shoulder
                calculate_angle(lm_xyz[12], lm_xyz[24], lm_xyz[26]),  # right hip
                calculate_angle(lm_xyz[24], lm_xyz[26], lm_xyz[28]),  # right knee
                calculate_angle(lm_xyz[26], lm_xyz[28], lm_xyz[32])   # right ankle
            ]

            angles_norm = [a / np.pi for a in angles]
            data_norm.extend(angles_norm)

            # =======================
            # INFERENCE
            # =======================
            if len(data_norm) == EXPECTED_FEATURES:
                input_data = np.asarray([data_norm], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = CATEGORIES[np.argmax(output_data)]

                cv2.putText(
                    frame,
                    predicted_class,
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    4
                )

            # =======================
            # VISUALIZATION (UNCHANGED)
            # =======================
            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                x1 = int(smoothed_landmarks[start_idx][0] * W)
                y1 = int(smoothed_landmarks[start_idx][1] * H)
                x2 = int(smoothed_landmarks[end_idx][0] * W)
                y2 = int(smoothed_landmarks[end_idx][1] * H)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("Full Body Pose Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


start_pose_recognition()
