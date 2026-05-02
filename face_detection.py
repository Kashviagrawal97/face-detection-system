import cv2
import numpy as np

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(image_bytes):
    img   = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces, _, scores = face_cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), outputRejectLevels=True
    )

    if len(faces) == 0: return None, "No face detected."

    best_index = int(np.argmax(scores))
    x, y, w, h = faces[best_index]
    return cv2.resize(gray[y:y+h, x:x+w], (100, 100)), None

# Open webcam (0 = default camera)
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    print("Webcam started... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

# Convert to grayscale (Haar Cascade works on grayscale)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
 # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
              scaleFactor=1.1, 
              minNeighbors=8, 
              minSize=(80, 80))
        
# Draw rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show face count
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
     
# Display the frame
        cv2.imshow('Face Detection', frame)

# Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

