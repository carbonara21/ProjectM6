import os
import cv2
import time

DATA_DIR = 'data/BC_D_2'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 600

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, f"BC_D_{j}")
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user to press Q
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 10-second delay with countdown
    for t in range(10, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f'Starting in {t}...', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

    # Start capturing images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(1)  # small delay
        filename = os.path.join(class_dir, f'Person {counter}.jpg')
        cv2.imwrite(filename, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
