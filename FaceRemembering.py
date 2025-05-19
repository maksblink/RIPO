import cv2
import face_recognition
import os
import pickle
import numpy as np

# Select camera index
camera_index = input("Enter camera index (default 0): ").strip()
if camera_index == '':
    camera_index = 0
else:
    try:
        camera_index = int(camera_index)
    except ValueError:
        print("Invalid input. Using default camera 0.")
        camera_index = 0

person_name = input("Enter the person's name: ").strip()
if not person_name:
    print("No name provided.")
    exit()

pickle_file = "known_faces.pickle"
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

if person_name not in known_faces:
    known_faces[person_name] = []

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Cannot open camera {camera_index}.")
    exit()

output_folder = "extracted_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
sampling_interval = 10
min_face_size = 100  # minimum face width/height in pixels

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    frame_count += 1

    if frame_count % sampling_interval != 0:
        cv2.imshow("Preview - Extracted Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_width = right - left
        face_height = bottom - top

        if face_width < min_face_size or face_height < min_face_size:
            print(f"Skipped small face at frame {frame_count}, index {i}")
            continue

        face_image = frame[top:bottom, left:right]
        filename = os.path.join(output_folder, f"{person_name}_face_{frame_count}_{i}.jpg")
        cv2.imwrite(filename, face_image)

        cropped_rgb = np.ascontiguousarray(rgb_frame[top:bottom, left:right])

        # Check for landmarks to filter bad faces
        landmarks = face_recognition.face_landmarks(cropped_rgb)
        if not landmarks:
            print(f"Skipped face with no landmarks at frame {frame_count}, index {i}")
            continue

        encoding = face_recognition.face_encodings(cropped_rgb)
        if encoding:
            known_faces[person_name].append(encoding[0])
            print(f"Saved encoding for {filename}")
        else:
            print(f"Failed to compute encoding for {filename}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Preview - Extracted Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with open(pickle_file, "wb") as f:
    pickle.dump(known_faces, f)

print(f"Saved face encodings for '{person_name}'")
cap.release()
cv2.destroyAllWindows()
