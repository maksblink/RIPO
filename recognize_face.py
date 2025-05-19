import cv2
import face_recognition
import pickle
import numpy as np
from PIL import Image, ImageDraw

# Ask user to select camera index
camera_index = input("Enter camera index (default 0): ").strip()
if camera_index == '':
    camera_index = 0
else:
    try:
        camera_index = int(camera_index)
    except ValueError:
        print("Invalid input. Using default camera 0.")
        camera_index = 0

# Load known face encodings from the pickle file.
with open("known_faces.pickle", "rb") as f:
    known_faces = pickle.load(f)

known_encodings = []
known_names = []
for name, encodings in known_faces.items():
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

def face_confidence(face_distance, threshold=0.6):
    if face_distance > threshold:
        return 0.0
    return (1.0 - face_distance / threshold) * 100

BRIGHTNESS_THRESHOLD = 75
CASCADE_NEIGHBORS = 4
CASCADE_MIN_SIZE = (40, 40)

glasses_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Cannot open camera {camera_index}.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0,0), fx=1.25, fy=1.25)

    # Detect faces on the normal RGB frame (no CLAHE here)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    raw_locs = face_recognition.face_locations(rgb_frame, model='hog')

    face_locations = []
    for loc in raw_locs:
        if isinstance(loc, (tuple, list)):
            face_locations.append(tuple(loc))
        elif hasattr(loc, 'rect'):
            r = loc.rect
            face_locations.append((r.top(), r.right(), r.bottom(), r.left()))

    try:
        # We will recompute encodings conditionally below
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception:
        continue

    landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

    for idx, ((top, right, bottom, left), face_encoding, landmarks) in enumerate(zip(face_locations, face_encodings, landmarks_list)):
        wearing_sunglasses = False
        eye_pts = landmarks.get('left_eye', []) + landmarks.get('right_eye', [])

        # Detect sunglasses on eyes
        if eye_pts:
            xs, ys = zip(*eye_pts)
            x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, frame.shape[1])
            y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, frame.shape[0])
            eye_roi = frame[y1:y2, x1:x2]
            if eye_roi.size:
                gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray_eye)
                glasses = glasses_cascade.detectMultiScale(
                    gray_eye,
                    scaleFactor=1.1,
                    minNeighbors=CASCADE_NEIGHBORS,
                    minSize=CASCADE_MIN_SIZE
                )
                if len(glasses) > 0 or mean_brightness < BRIGHTNESS_THRESHOLD:
                    wearing_sunglasses = True

        # Crop face for conditional encoding
        face_img = rgb_frame[top:bottom, left:right]

        # If wearing sunglasses, mask eyes before encoding and recompute encoding
        if wearing_sunglasses and eye_pts:
            pil_img = Image.fromarray(face_img)
            draw = ImageDraw.Draw(pil_img)
            # Translate eye points to local face coordinates
            eye_pts_local = [(x - left, y - top) for (x, y) in eye_pts]
            draw.polygon(eye_pts_local, fill=(0, 0, 0))
            face_img_masked = np.array(pil_img)
            encodings = face_recognition.face_encodings(face_img_masked)
            threshold = 0.75  # More lenient threshold for sunglasses
        else:
            encodings = [face_encoding]
            threshold = 0.6

        if not encodings:
            name = 'Unknown'
            confidence = 0.0
        else:
            encoding = encodings[0]
            name = 'Unknown'
            confidence = 0.0
            if known_encodings:
                dists = face_recognition.face_distance(known_encodings, encoding)
                best_idx = np.argmin(dists)
                if dists[best_idx] < threshold:
                    name = known_names[best_idx]
                    confidence = face_confidence(dists[best_idx], threshold)
                    if wearing_sunglasses:
                        confidence *= 0.85  # reduce confidence a bit if sunglasses

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name}: {confidence:.0f}%"
        if wearing_sunglasses:
            label += " (sunglasses)"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sunglasses warning box
        if wearing_sunglasses:
            msg = 'Please take off your sunglasses'
            (w, h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (left, bottom + 5), (left + w, bottom + 5 + h), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, msg, (left, bottom + 5 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Real-time Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
