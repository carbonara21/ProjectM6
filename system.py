import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf


pygame.mixer.init()
def play_video_then_start_pose():
    video_path = r"C:\Users\s2887800\PycharmProjects\ProjectM6"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        start_pose_recognition()
        return

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            video_label.destroy()
            start_pose_recognition()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, (root.winfo_width(), root.winfo_height()))
        img = ImageTk.PhotoImage(Image.fromarray(frame_resized))

        video_label.config(image=img)
        video_label.image = img

        video_label.after(15, update_frame)

    update_frame()
def start_pose_recognition():
    root.destroy()

    interpreter = tf.lite.Interpreter(model_path="pose_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sounds = {
        "close": r"C:\Users\s2887800\PycharmProjects\ProjectM6\audios\CloseArms.mp3",
        "elbows": r"C:\Users\s2887800\PycharmProjects\ProjectM6\audios\ElbowsTorso.mp3"
    }

    categories = ["BC_D_1", "BC_D_2", "BC_D_3", "BC_D_I1",
                  "BC_D_I2", "BC_D_I3"]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Full Body Pose Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Full Body Pose Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    curl_sequence = [
        "BC_D_1",
        "BC_D_2",
        "BC_D_3",
        "BC_D_2",
        "BC_D_1"
    ]

    pose_history = []
    counter = 0
    last_stage = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        data_norm = []
        x_vals, y_vals = [], []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            for lm in landmarks:
                x_vals.append(lm.x)
                y_vals.append(lm.y)

            if len(x_vals) == 33:
                x_min, y_min = min(x_vals), min(y_vals)

                for x, y in zip(x_vals, y_vals):
                    data_norm.append(x - x_min)
                    data_norm.append(y - y_min)

                if len(data_norm) == 66:
                    input_data = np.asarray([data_norm], dtype=np.float32)

                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])

                    predicted_char_index = np.argmax(output_data)
                    predicted_char = categories[predicted_char_index]

                    if predicted_char != last_stage:
                        last_stage = predicted_char
                        pose_history.append(predicted_char)

                        if len(pose_history) > 5:
                            pose_history = pose_history[-5:]

                        print("POSE HISTORY:", pose_history)

                        if pose_history == curl_sequence:
                            counter += 1
                            print(">>> FULL CURL COUNTED:", counter)
                            if pose_history:
                                last_pose = pose_history[-1]
                                pose_history = [last_pose]
                            else:
                                pose_history = []

                    if predicted_char in ["BC_D_1", "BC_D_2", "BC_D_3"]:
                        skeleton_color_outer = (0, 255, 0)
                    else:
                        skeleton_color_outer = (0, 0, 255)




                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection

                        # skip anything outside the arm range
                        if start_idx < 11 or end_idx < 11 or start_idx > 16 or end_idx > 16:
                            continue

                        # draw arm connection here

                        x1 = int(landmarks[start_idx].x * W)
                        y1 = int(landmarks[start_idx].y * H)
                        x2 = int(landmarks[end_idx].x * W)
                        y2 = int(landmarks[end_idx].y * H)
                        cv2.line(frame, (x1, y1), (x2, y2), skeleton_color_outer, 4)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.3
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(predicted_char, font, scale, thickness)
                    cv2.putText(frame, predicted_char, ((W - text_w)//2, 60), font, scale, (255, 255, 255), thickness)

                    counter_text = f"Curls: {counter}"
                    (text_w, text_h), _ = cv2.getTextSize(counter_text, font, 1.5, 3)
                    cv2.putText(frame, counter_text, ((W - text_w)//2, H - 60), font, 1.5, (255, 255, 255), 3)

        cv2.imshow("Full Body Pose Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


root = tk.Tk()
root.attributes('-fullscreen', True)
root.title("Project")

color1 = "#020f12"
color2 = "#05d7ff"
color3 = "#65e7ff"
color4 = "BLACK"

main_frame = tk.Frame(root, bg="#1a1a1a", pady=40)
main_frame.pack(fill=tk.BOTH, expand=True)

video_label = tk.Label(root, bg="#1a1a1a")
video_label.place(relx=0.5, rely=0.5, anchor="center")

title_label = tk.Label(
    main_frame,
    text="Welcome to the Bicep Curl Trainer",
    font=("Arial", 36, "bold"),
    fg="white",
    bg="#1a1a1a"
)
title_label.pack(pady=20)

button1 = tk.Button(
    main_frame,
    background=color2,
    foreground=color4,
    activebackground=color3,
    activeforeground=color4,
    highlightthickness=2,
    width=13,
    height=2,
    border=0,
    cursor="hand2",
    text="Start",
    font=("Arial", 16, "bold"),
    command=play_video_then_start_pose
)

button1.pack()
root.mainloop()
