import cv2
import face_recognition
import pickle
import numpy as np

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

# Detection parameters
BRIGHTNESS_THRESHOLD = 75   # mean gray level below which eyes are considered dark
CASCADE_NEIGHBORS = 4       # tuning for glasses_cascade
CASCADE_MIN_SIZE = (40, 40) # minimum size for sunglasses detection

# Load sunglasses Haar cascade
glasses_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

# Open the webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect faces
    raw_locs = face_recognition.face_locations(rgb_frame, model='hog')
    # Normalize locations
    face_locations = []
    for loc in raw_locs:
        if isinstance(loc, (tuple, list)):
            face_locations.append(tuple(loc))
        elif hasattr(loc, 'rect'):
            r = loc.rect
            face_locations.append((r.top(), r.right(), r.bottom(), r.left()))

    # Compute encodings
    try:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception:
        continue

    # Get landmarks for eyes
    landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, landmarks_list):
        # Recognition
        name = 'Unknown'; confidence = 0.0
        if known_encodings:
            dists = face_recognition.face_distance(known_encodings, face_encoding)
            idx = np.argmin(dists)
            if dists[idx] < 0.6:
                name = known_names[idx]
                confidence = face_confidence(dists[idx])

        # Eye region
        eye_pts = landmarks.get('left_eye', []) + landmarks.get('right_eye', [])
        wearing_sunglasses = False
        if eye_pts:
            xs, ys = zip(*eye_pts)
            x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, frame.shape[1])
            y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, frame.shape[0])
            eye_roi = frame[y1:y2, x1:x2]
            if eye_roi.size:
                gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                # cascade detection
                glasses = glasses_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=CASCADE_NEIGHBORS,
                    minSize=CASCADE_MIN_SIZE
                )
                # combine: glasses detected OR very low brightness
                if len(glasses) > 0 or mean_brightness < BRIGHTNESS_THRESHOLD:
                    wearing_sunglasses = True

        # Draw
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"{name}: {confidence:.0f}%", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if wearing_sunglasses:
            msg = 'Please take off your sunglasses'
            (w,h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (left, bottom+5), (left+w, bottom+5+h), (0,0,255), cv2.FILLED)
            cv2.putText(frame, msg, (left, bottom+5+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Real-time Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
