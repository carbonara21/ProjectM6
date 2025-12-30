import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# =======================
# SMA (UNCHANGED)
# =======================
def _transform_landmarks(pose_landmarks_list: list, landmark_buffers) -> list:
    transformed_landmarks_list = []

    if not pose_landmarks_list:
        return transformed_landmarks_list

    for pose_landmarks in pose_landmarks_list:
        transformed_landmarks = []
        for i, landmark in enumerate(pose_landmarks):
            x, y, z = landmark.x, landmark.y, landmark.z
            x_t, y_t, z_t = sma_transform_func(x, y, z, landmark_buffers, i)
            transformed_landmarks.append((x_t, y_t, z_t))
        transformed_landmarks_list.append(transformed_landmarks)

    return transformed_landmarks_list

def sma_transform_func(x, y, z, landmark_buffers, idx):
    buffer = landmark_buffers[idx]
    buffer.append([x, y, z])
    if len(buffer) == buffer.maxlen:
        return np.mean(buffer, axis=0)
    return x, y, z

# =======================
# 2D Angle function (NO Z)
# =======================
def calculate_angle(a, b, c):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0))

# =======================
# CONSTANTS
# =======================
EXPECTED_FEATURES = 5
CATEGORIES = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]

# =======================
# MAIN
# =======================
def start_pose_recognition():
    interpreter = tf.lite.Interpreter(model_path="M_D_S2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    landmark_buffers = [deque(maxlen=3) for _ in range(33)]

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
            smoothed_landmarks = _transform_landmarks(
                [results.pose_landmarks.landmark],
                landmark_buffers
            )[0]

            lm = smoothed_landmarks

            # =======================
            # RIGHT-SIDE 2D ANGLES ONLY
            # =======================
            angles = [
                calculate_angle(lm[12], lm[14], lm[16]),  # elbow
                calculate_angle(lm[14], lm[12], lm[24]),  # shoulder
                calculate_angle(lm[12], lm[24], lm[26]),  # hip
                calculate_angle(lm[24], lm[26], lm[28]),  # knee
                calculate_angle(lm[26], lm[28], lm[32])   # ankle
            ]

            angles_norm = [a / np.pi for a in angles]
            input_data = np.asarray([angles_norm], dtype=np.float32)

            # =======================
            # INFERENCE
            # =======================
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = CATEGORIES[np.argmax(output_data)]

            cv2.putText(
                frame,
                predicted_class,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                6,
                (255, 255, 255),
                7
            )

            if predicted_class in ["D_S_1", "D_S_2", "D_S_3"]:
                skeleton_color = (0, 255, 0)
            else:
                skeleton_color = (0, 0, 255)

            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                x1 = int(lm[start_idx][0] * W)
                y1 = int(lm[start_idx][1] * H)
                x2 = int(lm[end_idx][0] * W)
                y2 = int(lm[end_idx][1] * H)
                cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 2)

        cv2.imshow("Deadlift – Right Side Angles Only", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

start_pose_recognition()
