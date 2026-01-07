import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

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
def calculate_angle(a, b, c):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine, -1.0, 1.0))

def start_pose_recognition():
    categories = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]
    landmark_buffers = [deque(maxlen=3) for _ in range(33)]
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    sequence = ["D_S_1", "D_S_2", "D_S_3", "D_S_2", "D_S_1"]
    pose_history = []
    counter = 0
    last_pose = None


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            interpreter = tf.lite.Interpreter(model_path="models/DeadLift_Model.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            smoothed_landmarks = _transform_landmarks(
                [results.pose_landmarks.landmark],
                landmark_buffers
            )[0]

            lm = smoothed_landmarks


            angles = [
                calculate_angle(lm[12], lm[14], lm[16]),  # elbow
                calculate_angle(lm[14], lm[12], lm[24]),  # shoulder
                calculate_angle(lm[12], lm[24], lm[26]),  # hip
                calculate_angle(lm[24], lm[26], lm[28]),  # knee
                calculate_angle(lm[26], lm[28], lm[32])   # ankle
            ]

            angles_norm = [a / np.pi for a in angles]
            input_data = np.asarray([angles_norm], dtype=np.float32)


            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = categories[np.argmax(output_data)]

            cv2.putText(frame, predicted_class, (W//2, H//4), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255),7)
            cv2.putText(frame, f"Count: {counter}", ((W - 300)//2, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            line_feedback(frame, smoothed_landmarks, W, H, predicted_class, pose_history)

            if predicted_class in ["D_S_1", "D_S_2", "D_S_3"]:
                skeleton_color = (0, 255, 0)
            else:
                skeleton_color = (0, 0, 255)

            if predicted_class != last_pose:
                last_pose = predicted_class
                pose_history.append(predicted_class)
                if len(pose_history) > 3:
                    pose_history = pose_history[-3:]

                if pose_history == sequence:
                    counter += 1
                    pose_history = [pose_history[-1]]

            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                x1 = int(lm[start_idx][0] * W)
                y1 = int(lm[start_idx][1] * H)
                x2 = int(lm[end_idx][0] * W)
                y2 = int(lm[end_idx][1] * H)
                cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 2)

        cv2.imshow("Deadlift", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def line_feedback(output_image, landmarks, W, H, predicted_class, pose_history):
    if len(pose_history) >= 2:
        previous_pose = pose_history[-2]
    else:
        previous_pose = None
    arrow_offset = 20
    arrow_height = 100
    arrow_length = 100


    if predicted_class in ["D_S_1", "D_S_2", "D_S_3"]:
        arrow_color = (0, 255, 0)   # GREEN = correct
    else:
        arrow_color = (0, 0, 255)   # RED = incorrect

    if predicted_class == "D_S_1":
        head = landmarks[0]
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]

        hx, hy = int(head[0] * W), int(head[1] * H)
        sx, sy = int(right_shoulder[0] * W), int(right_shoulder[1] * H)
        hip_x, hip_y = int(right_hip[0] * W), int(right_hip[1] * H)

        cv2.arrowedLine(output_image,(hx, hy - arrow_offset),(hx, hy - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(sx, sy - arrow_offset),(sx, sy - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(hip_x, hip_y - arrow_offset),(hip_x, hip_y - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
    elif predicted_class == "D_S_2" and previous_pose == "D_S_1":
        right_hip = landmarks[24]
        right_shoulder = landmarks[12]

        hip_x, hip_y = int(right_hip[0] * W), int(right_hip[1] * H)
        sh_x, sh_y = int(right_shoulder[0] * W), int(right_shoulder[1] * H)

        cv2.arrowedLine(output_image,(hip_x, hip_y - arrow_offset),(hip_x + arrow_length, hip_y - arrow_offset),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(sh_x, sh_y - arrow_offset),(sh_x, sh_y - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
    elif predicted_class == "D_S_2" and previous_pose == "D_S_3":
        right_hip = landmarks[24]
        right_shoulder = landmarks[12]

        hip_x, hip_y = int(right_hip[0] * W), int(right_hip[1] * H)
        sh_x, sh_y = int(right_shoulder[0] * W), int(right_shoulder[1] * H)

        cv2.arrowedLine(output_image,(hip_x, hip_y + arrow_offset),(hip_x, hip_y + arrow_offset + arrow_height),arrow_color,7,tipLength=0.3)
        cv2.arrowedLine(output_image,(sh_x, sh_y + arrow_offset),(sh_x, sh_y + arrow_offset + arrow_height),arrow_color,7,tipLength=0.3)

    elif predicted_class == "D_S_I1":
        head = landmarks[0]
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]

        hx, hy = int(head[0] * W), int(head[1] * H)
        sx, sy = int(right_shoulder[0] * W), int(right_shoulder[1] * H)
        hip_x, hip_y = int(right_hip[0] * W), int(right_hip[1] * H)

        cv2.arrowedLine(output_image,(hx, hy - arrow_offset),(hx, hy - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(sx, sy - arrow_offset),(sx, sy - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(hip_x, hip_y + arrow_offset),(hip_x, hip_y + arrow_offset + arrow_height),arrow_color, 7, tipLength=0.3)
    elif predicted_class == "D_S_I2":
        head = landmarks[0]
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]

        hx, hy = int(head[0] * W), int(head[1] * H)
        sh_x, sh_y = int(right_shoulder[0] * W), int(right_shoulder[1] * H)
        hip_x, hip_y = int(right_hip[0] * W), int(right_hip[1] * H)

        cv2.arrowedLine(output_image,(hx, hy - arrow_offset),(hx, hy - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(sh_x, sh_y - arrow_offset),(sh_x, sh_y - arrow_offset - arrow_height),arrow_color, 7, tipLength=0.3)
        cv2.arrowedLine(output_image,(hip_x, hip_y - arrow_offset),(hip_x, hip_y - arrow_offset + arrow_height),arrow_color, 7, tipLength=0.3)

start_pose_recognition()
