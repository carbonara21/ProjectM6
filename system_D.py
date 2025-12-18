import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

def start_pose_recognition():
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="M_D_S.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Categories corresponding to the model
    categories = ["D_S_1", "D_S_2", "D_S_3", "D_S_I1", "D_S_I2"]

    # MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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

        # Segmentation processing
        seg_results = selfie_segmentation.process(frame_rgb)
        condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = cv2.GaussianBlur(frame, (55, 55), 0)
        output_image = np.where(condition, frame, bg_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract the normalized 3D coordinates for all landmarks
            x_vals = []
            y_vals = []
            z_vals = []

            # Collect x, y, z coordinates for all landmarks
            for lm in results.pose_landmarks.landmark:
                x_vals.append(lm.x)
                y_vals.append(lm.y)
                z_vals.append(lm.z)

            # Min-max normalization for all coordinates
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            z_min, z_max = min(z_vals), max(z_vals)

            data_norm = []
            for x, y, z in zip(x_vals, y_vals, z_vals):
                x_norm = (x - x_min) / (x_max - x_min)
                y_norm = (y - y_min) / (y_max - y_min)
                z_norm = (z - z_min) / (z_max - z_min)
                data_norm.extend([x_norm, y_norm, z_norm])

            # Ensure the data matches the expected format
            if len(data_norm) == 99:  # 33 landmarks * 3 (x, y, z)
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
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                x1 = int(landmarks[start_idx].x * W)
                y1 = int(landmarks[start_idx].y * H)
                x2 = int(landmarks[end_idx].x * W)
                y2 = int(landmarks[end_idx].y * H)
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Display the predicted action
            cv2.putText(output_image, predicted_class, (50, 50), font, 1.5, (255, 255, 255), 2)

            # Display the curl counter (or other action counter)
            cv2.putText(output_image, f"Curls: {counter}", ((W - 300) // 2, H - 60), font, 1.5, (0, 0, 0), 3)

        # Show the processed image
        cv2.imshow("Full Body Pose Recognition", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

start_pose_recognition()
