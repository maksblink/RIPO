import cv2
import face_recognition
import pickle
import numpy as np
import dlib  # For type checking

# Load known face encodings from the pickle file.
with open("known_faces.pickle", "rb") as f:
    known_faces = pickle.load(f)

# Flatten the dictionary into lists of encodings and names.
known_encodings = []
known_names = []
for name, encodings in known_faces.items():
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# Function to compute a simple confidence percentage from face distance.
def face_confidence(face_distance, threshold=0.6):
    if face_distance > threshold:
        return 0.0
    return (1.0 - face_distance / threshold) * 100

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

    # Convert frame from BGR to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Explicitly use the "hog" model.
    raw_face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # Convert each face location to a tuple (top, right, bottom, left) 
    converted_face_locations = []
    for loc in raw_face_locations:
        # If it's already a tuple/list of length 4, use it.
        if isinstance(loc, (tuple, list)) and len(loc) == 4:
            converted_face_locations.append(tuple(loc))
        # If it has top/right/bottom/left methods, use those.
        elif hasattr(loc, "top") and hasattr(loc, "right") and hasattr(loc, "bottom") and hasattr(loc, "left"):
            try:
                converted_face_locations.append((loc.top(), loc.right(), loc.bottom(), loc.left()))
            except Exception as e:
                print("Error converting using methods:", e)
        # Otherwise, if it has a rect attribute, use that.
        elif hasattr(loc, "rect"):
            try:
                converted_face_locations.append((loc.rect.top(), loc.rect.right(), loc.rect.bottom(), loc.rect.left()))
            except Exception as e:
                print("Error converting using rect attribute:", e)
        else:
            print("Unrecognized face location format:", loc)
    face_locations = converted_face_locations


    # Compute face encodings with jittering disabled
    try:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=0)
    except Exception as e:
        print("Error computing face encodings:", e)
        continue

    # Loop over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            name = known_names[best_match_index] if best_distance < 0.6 else "Unknown"
            confidence = face_confidence(best_distance)
        else:
            name = "Unknown"
            confidence = 0.0

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name}: {confidence:.2f}%"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Real-time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
