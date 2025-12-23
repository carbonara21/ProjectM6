import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pygame
from collections import deque

## https://medium.com/@debasishraut.dev/setting-up-smoothing-filters-for-mediapipe-pose-estimation-pipeline-a-practical-guide-fcc03f462196
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



def start_pose_recognition():
    pygame.init()
    interpreter = tf.lite.Interpreter(model_path="M_BC_D.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    categories = ["BC_D_1", "BC_D_2", "BC_D_I1", "BC_D_I2"]
    landmark_indices = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28]

    landmark_buffers = [deque(maxlen=5) for _ in range(33)]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    curl_sequence = ["BC_D_1", "BC_D_2", "BC_D_1"]
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

            left_shoulder = smoothed_landmarks[11]
            right_shoulder = smoothed_landmarks[12]
            center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            center_z = (left_shoulder[2] + right_shoulder[2]) / 2

            left_hip = smoothed_landmarks[23]
            right_hip = smoothed_landmarks[24]
            torso_center_x = (left_hip[0] + right_hip[0]) / 2
            torso_center_y = (left_hip[1] + right_hip[1]) / 2
            torso_center_z = (left_hip[2] + right_hip[2]) / 2

            torso_length = np.sqrt(
                (center_x - torso_center_x) ** 2 +
                (center_y - torso_center_y) ** 2 +
                (center_z - torso_center_z) ** 2
            ) + 1e-6

            data_raw = []
            for idx in landmark_indices:
                x_norm = (smoothed_landmarks[idx][0] - center_x) / torso_length
                y_norm = (smoothed_landmarks[idx][1] - center_y) / torso_length
                z_norm = (smoothed_landmarks[idx][2] - center_z) / torso_length
                data_raw.extend([x_norm, y_norm, z_norm])

            input_data = np.asarray([data_raw], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_index = np.argmax(output_data)
            predicted_class = categories[predicted_index]

            # Skeleton color
            skeleton_color = (0, 255, 0) if predicted_class in ["BC_D_1", "BC_D_2"] else (0, 0, 255)

            # Update pose history
            if predicted_class != last_stage:
                last_stage = predicted_class
                pose_history.append(predicted_class)
                if len(pose_history) > 3:
                    pose_history = pose_history[-3:]
                # Count curls
                if pose_history == curl_sequence:
                    counter += 1
                    pose_history = [pose_history[-1]]

            arm_connections = [
                (11, 13), (13, 15),  # left arm
                (12, 14), (14, 16)  # right arm
            ]

            for start_idx, end_idx in arm_connections:
                x1 = int(smoothed_landmarks[start_idx][0] * W)  # use smoothed landmarks
                y1 = int(smoothed_landmarks[start_idx][1] * H)
                x2 = int(smoothed_landmarks[end_idx][0] * W)
                y2 = int(smoothed_landmarks[end_idx][1] * H)
                cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 4)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Curls: {counter}", ((W - 300)//2, H - 60), font, 1.5, (0, 0, 0), 3)
            line_feedback(frame, smoothed_landmarks, W, H, predicted_class)

        cv2.imshow("Full Body Pose Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def line_feedback(output_image, landmarks, W, H, predicted_class):
    if predicted_class == "BC_D_1":
        right_wrist = landmarks[16]
        left_wrist = landmarks[15]

        rw_x, rw_y = int(right_wrist[0] * W), int(right_wrist[1] * H)
        lw_x, lw_y = int(left_wrist[0] * W), int(left_wrist[1] * H)
        arrow_height = 50
        arrow_offset = 20

        cv2.arrowedLine(output_image, (rw_x, rw_y - arrow_offset), (rw_x, rw_y - arrow_offset - arrow_height), (0,255,0), 7, tipLength=0.3)
        cv2.arrowedLine(output_image, (lw_x, lw_y - arrow_offset), (lw_x, lw_y - arrow_offset - arrow_height), (0,255,0), 7, tipLength=0.3)
    elif predicted_class == "BC_D_2":
        right_wrist = landmarks[16]
        left_wrist = landmarks[15]
        rw_x, rw_y = int(right_wrist[0] * W), int(right_wrist[1] * H)
        lw_x, lw_y = int(left_wrist[0] * W), int(left_wrist[1] * H)
        arrow_height = 50
        arrow_offset = 20

        cv2.arrowedLine(output_image, (rw_x, rw_y - arrow_offset), (rw_x, rw_y - arrow_offset + arrow_height), (0,255,0), 7, tipLength=0.3)
        cv2.arrowedLine(output_image, (lw_x, lw_y - arrow_offset), (lw_x, lw_y - arrow_offset + arrow_height), (0,255,0), 7, tipLength=0.3)
    elif predicted_class == "BC_D_I1":
        right_elbow = landmarks[14]
        left_elbow = landmarks[13]
        rw_x, rw_y = int(right_elbow[0]*W), int(right_elbow[1]*H)
        lw_x, lw_y = int(left_elbow[0]*W), int(left_elbow[1]*H)
        arrow_length = 50
        arrow_offset = 20

        cv2.arrowedLine(output_image, (rw_x, rw_y - arrow_offset), (rw_x + arrow_length, rw_y - arrow_offset), (0,0,255),7,tipLength=0.3)
        cv2.arrowedLine(output_image, (lw_x, lw_y - arrow_offset), (lw_x + arrow_length, lw_y - arrow_offset), (0,0,255),7,tipLength=0.3)
    elif predicted_class == "BC_D_I2":
        right_elbow = landmarks[14]
        left_elbow = landmarks[13]
        rw_x, rw_y = int(right_elbow[0]*W), int(right_elbow[1]*H)
        lw_x, lw_y = int(left_elbow[0]*W), int(left_elbow[1]*H)
        arrow_length = 50
        arrow_offset = 20

        cv2.arrowedLine(output_image, (rw_x, rw_y - arrow_offset), (rw_x - arrow_length, rw_y - arrow_offset), (0,0,255),7,tipLength=0.3)
        cv2.arrowedLine(output_image, (lw_x, lw_y - arrow_offset), (lw_x - arrow_length, lw_y - arrow_offset), (0,0,255),7,tipLength=0.3)

start_pose_recognition()
