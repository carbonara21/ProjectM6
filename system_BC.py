import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

def start_pose_recognition():
    interpreter = tf.lite.Interpreter(model_path="M_BC_D.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    categories = ["BC_D_1", "BC_D_2",
                  "BC_D_I1", "BC_D_I2", "BC_D_I3"]
    landmark_indices = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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

        seg_results = selfie_segmentation.process(frame_rgb)
        condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = cv2.GaussianBlur(frame, (55, 55), 0)
        output_image = np.where(condition, frame, bg_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            ## Center as midpoint between shoulder
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            center_y = (left_shoulder.y + right_shoulder.y) / 2
            center_z = (left_shoulder.z + right_shoulder.z) / 2

            left_hip = landmarks[23]
            right_hip = landmarks[24]
            ## Torso Center as midpoint between hips
            torso_center_x = (left_hip.x + right_hip.x) / 2
            torso_center_y = (left_hip.y + right_hip.y) / 2
            torso_center_z = (left_hip.z + right_hip.z) / 2

            ## Euclidean
            torso_length = np.sqrt(
                (center_x - torso_center_x) ** 2 +
                (center_y - torso_center_y) ** 2 +
                (center_z - torso_center_z) ** 2
            ) + 1e-6

            # Collect normalized 3D arm coordinates
            data_raw = []
            for idx in landmark_indices:
                lm = landmarks[idx]
                x_norm = (lm.x - center_x) / torso_length
                y_norm = (lm.y - center_y) / torso_length
                z_norm = (lm.z - center_z) / torso_length
                data_raw.extend([x_norm, y_norm, z_norm])

            input_data = np.asarray([data_raw], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_index = np.argmax(output_data)
            predicted_class = categories[predicted_index]

            if predicted_class in ["BC_D_1", "BC_D_2", "BC_D_3"]:
                skeleton_color = (0, 255, 0)
            else:
                skeleton_color = (0, 0, 255)

            if predicted_class != last_stage:
                last_stage = predicted_class
                pose_history.append(predicted_class)
                if len(pose_history) > 3:
                    pose_history = pose_history[-5:]
                print("POSE HISTORY:", pose_history)

                # Count curls
                if pose_history == curl_sequence:
                    counter += 1
                    print(">>> FULL CURL COUNTED:", counter)
                    pose_history = [pose_history[-1]]

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < 11 or end_idx < 11 or start_idx > 16 or end_idx > 16:
                    continue
                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue
                # Access landmarks directly
                x1 = int(landmarks[start_idx].x * W)
                y1 = int(landmarks[start_idx].y * H)
                x2 = int(landmarks[end_idx].x * W)
                y2 = int(landmarks[end_idx].y * H)
                cv2.line(output_image, (x1, y1), (x2, y2), skeleton_color, 4)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if predicted_class == "BC_D_1":
                cv2.putText(output_image, "Arms stretched", ((W - 300)//2, 60), font, 1.3, (0, 0, 0), 2)
            elif predicted_class == "BC_D_2":
                cv2.putText(output_image, "Full Curl", ((W - 300)//2, 60), font, 1.3, (0, 0, 0), 2)
            elif predicted_class == "BC_D_3":
                cv2.putText(output_image, "Full curl", ((W - 300)//2, 60), font, 1.3, (0, 0, 0), 2)
            elif predicted_class == "BC_D_I1":
                cv2.putText(output_image, "Elbows are back and not close to torso", ((W - 300) // 2, 60), font, 1.3, (0, 0, 0), 2)
            elif predicted_class == "BC_D_I2":
                cv2.putText(output_image, "Elbows are back and not close to torso", ((W - 300) // 2, 60), font, 1.3, (0, 0, 0), 2)
            elif predicted_class == "BC_D_I3":
                cv2.putText(output_image, "Legs are not far apart", ((W - 300) // 2, 60), font, 1.3, (0, 0, 0), 2)

            cv2.putText(output_image, f"Curls: {counter}", ((W - 300)//2, H - 60), font, 1.5, (0, 0, 0), 3)

        cv2.imshow("Full Body Pose Recognition", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

start_pose_recognition()
