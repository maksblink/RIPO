import cv2
import face_recognition
import pickle
import numpy as np

# Load the known face encodings from the pickle file.
with open("known_faces.pickle", "rb") as f:
    known_faces = pickle.load(f)

# Flatten the dictionary to create a list of encodings and a corresponding list of names.
known_encodings = []
known_names = []
for name, encodings in known_faces.items():
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# Define a function to compute a confidence score from a face distance.
def face_confidence(face_distance, threshold=0.6):
    """
    Returns a confidence score as a percentage based on the face distance.
    A distance of 0 returns 100% confidence, and a distance equal to the threshold returns 0%.
    """
    if face_distance > threshold:
        return 0.0
    else:
        # Linear mapping from distance to confidence
        confidence = (1.0 - face_distance / threshold)
        return confidence * 100

# Open the webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    # Convert the frame from BGR to RGB as required by face_recognition.
    rgb_frame = frame[:, :, ::-1]
    
    # Find all face locations and encodings in the current frame.
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compute distances between the detected face and all known face encodings.
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if len(distances) > 0:
            # Identify the best match (smallest distance).
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            # If the best distance is below the threshold, assign the corresponding name.
            name = known_names[best_match_index] if best_distance < 0.6 else "Unknown"
            confidence = face_confidence(best_distance)
        else:
            name = "Unknown"
            confidence = 0.0

        # Draw a rectangle around the face.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Prepare and put the label text with name and confidence percentage.
        label = f"{name}: {confidence:.2f}%"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Display the frame with annotations.
    cv2.imshow("Real-time Face Recognition", frame)
    
    # Exit the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
