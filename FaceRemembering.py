import cv2
import face_recognition
import os
import pickle
import numpy as np

#Name to label encoding in dictionary
person_name = input("Enter the person's name: ").strip()
if not person_name:
    print("No name provided.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Folder for extracted face images
output_folder = "extracted_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
extracted_count = 0
sampling_interval = 10  # Process every x frame

# Dict to store the faces using the person_name as key
known_faces = {person_name: []}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    frame_count += 1

    # Skip frames to reduce processing load
    if frame_count % sampling_interval != 0:
        cv2.imshow("Preview - Extracted Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Convert frame to RGB for face_recognition
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Save the face image from the frame (in BGR)
        face_image = frame[top:bottom, left:right]
        filename = os.path.join(output_folder, f"{person_name}_face_{frame_count}_{i}.jpg")
        cv2.imwrite(filename, face_image)
        extracted_count += 1

        # Compute the face encoding from the cropped RGB face
        cropped_rgb = np.ascontiguousarray(rgb_frame[top:bottom, left:right])
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

with open("known_faces.pickle", "wb") as f:
    pickle.dump(known_faces, f)

print(f"Saved face encodings for '{person_name}'")

cap.release()
cv2.destroyAllWindows()
