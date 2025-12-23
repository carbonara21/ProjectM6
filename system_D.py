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

def start_pose_recognition():
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="M_D_S.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Categories corresponding to the model
    categories = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]
    landmark_buffers = [deque(maxlen=5) for _ in range(33)]


    # MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Video capture
    cap = cv2.VideoCapture(0)

    pose_history = []
    counter = 0
    last_stage = None

    cv2.namedWindow("Full Body Pose Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Full Body Pose Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            pose_landmarks_list = [results.pose_landmarks.landmark]
            smoothed_landmarks_list = _transform_landmarks(pose_landmarks_list, landmark_buffers)
            smoothed_landmarks = smoothed_landmarks_list[0]
            skeleton_color = (0, 255, 0)  # default (white)

            # Extract the normalized 3D coordinates for all landmarks
            # --- collect smoothed x, y, z ---
            # --- collect smoothed x, y, z ---
            x_vals = []
            y_vals = []
            z_vals = []
            data_norm = []

            for lm in smoothed_landmarks:
                x_vals.append(lm[0])  # smoothed x
                y_vals.append(lm[1])  # smoothed y
                z_vals.append(lm[2])  # smoothed z

            # --- min-max normalization (same as training) ---
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            z_min, z_max = min(z_vals), max(z_vals)

            for x, y, z in zip(x_vals, y_vals, z_vals):
                # safety guard (training data likely never hit this, but runtime can)
                x_norm = 0.0 if x_max - x_min == 0 else (x - x_min) / (x_max - x_min)
                y_norm = 0.0 if y_max - y_min == 0 else (y - y_min) / (y_max - y_min)
                z_norm = 0.0 if z_max - z_min == 0 else (z - z_min) / (z_max - z_min)
                data_norm.extend([x_norm, y_norm, z_norm])

            # ---- ANGLE FEATURES (MUST MATCH TRAINING) ----
            lm_xyz = [(lm[0], lm[1], lm[2]) for lm in smoothed_landmarks]

            angles = [
                calculate_angle(lm_xyz[11], lm_xyz[13], lm_xyz[15]),  # left elbow
                calculate_angle(lm_xyz[12], lm_xyz[14], lm_xyz[16]),  # right elbow

                calculate_angle(lm_xyz[13], lm_xyz[11], lm_xyz[23]),  # left shoulder
                calculate_angle(lm_xyz[14], lm_xyz[12], lm_xyz[24]),  # right shoulder

                calculate_angle(lm_xyz[11], lm_xyz[23], lm_xyz[25]),  # left hip
                calculate_angle(lm_xyz[12], lm_xyz[24], lm_xyz[26]),  # right hip

                calculate_angle(lm_xyz[23], lm_xyz[25], lm_xyz[27]),  # left knee
                calculate_angle(lm_xyz[24], lm_xyz[26], lm_xyz[28]),  # right knee

                calculate_angle(lm_xyz[25], lm_xyz[27], lm_xyz[31]),  # left ankle
                calculate_angle(lm_xyz[26], lm_xyz[28], lm_xyz[32]),  # right ankle

                calculate_angle(lm_xyz[11], lm_xyz[12], lm_xyz[24]),  # torso tilt
                calculate_angle(lm_xyz[23], lm_xyz[11], lm_xyz[12])  # hip–shoulder alignment
            ]

            data_norm.extend(angles)

            # Ensure the data matches the expected format
            if len(data_norm) == 111:  # 33 landmarks * 3 (x, y, z)
                input_data = np.asarray([data_norm], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_index = np.argmax(output_data)
                predicted_class = categories[predicted_index]
                if predicted_class in ["D_S_1", "D_S_2", "D_S_3"]:
                    skeleton_color = (0, 255, 0)
                else:
                    skeleton_color = (0, 0, 255)

                if predicted_class != last_stage:
                    last_stage = predicted_class
                    pose_history.append(predicted_class)
                    if len(pose_history) > 3:
                        pose_history = pose_history[-3:]  # Keep the last 3 poses
                    print("POSE HISTORY:", pose_history)


                if pose_history == ["D_S_1", "D_S_2","D_S_3", "D_S_2" "D_S_1"]:
                    counter += 1
                    print(f">>> FULL CURL COUNTED: {counter}")
                    pose_history = [pose_history[-1]]

            # Visualize the skeleton
            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                x1 = int(smoothed_landmarks[start_idx][0] * W)
                y1 = int(smoothed_landmarks[start_idx][1] * H)
                x2 = int(smoothed_landmarks[end_idx][0] * W)
                y2 = int(smoothed_landmarks[end_idx][1] * H)
                cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 4)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Display the predicted action
            cv2.putText(frame, predicted_class, ((W - 300) // 2, H - 60), font, 5, (255, 0, 0), 5)

            # Display the curl counter (or other action counter)
            ##cv2.putText(frame, f"Curls: {counter}", ((W - 300) // 2, H - 60), font, 1.5, (0, 0, 0), 3)

        # Show the processed image
        cv2.imshow("Full Body Pose Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

start_pose_recognition()
