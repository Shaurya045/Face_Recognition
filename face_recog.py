import face_recognition
import numpy as np
import cv2
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load Known faces
shaurya_image = face_recognition.load_image_file("Face recognition/faces/shaurya.jpg", "RGB")
shaurya_encoding = face_recognition.face_encodings(shaurya_image)[0]

swarn_image = face_recognition.load_image_file("Face recognition/faces/swarn.jpg")
swarn_encoding = face_recognition.face_encodings(swarn_image)[0]

sanjay_image = face_recognition.load_image_file("Face recognition/faces/sanjay.jpeg")
sanjay_encoding = face_recognition.face_encodings(sanjay_image)[0]

known_face_encodings = [shaurya_encoding,swarn_encoding,sanjay_encoding]
known_face_names = ["Shaurya", "Swarn", "Sanjay"]

# List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = open(f"Attendance_{current_date}.csv", "w+", newline="")
writer = csv.writer(csv_file)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize Faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]

        # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale = 1.5
                fontColor = (0, 100, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
            
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                writer.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
csv_file.close()