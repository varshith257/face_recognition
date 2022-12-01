import face_recognition
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Loads the images and proess encodings of it 
img1 = face_recognition.load_image_file("C:\\Users\\Manoj\\Downloads\\Advance-Face-Recognition--master\\Advance-Face-Recognition--master\\1.jpg")
img1_encoding = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("C:\\Users\\Manoj\\Downloads\\Advance-Face-Recognition--master\\Advance-Face-Recognition--master\\2.jpg")
img2_encoding = face_recognition.face_encodings(img2)[0]

# Creates arrays for Known Faces encodings 
known_faces_encodings = [
    img1_encoding,
    img2_encoding,
    
]

# Mark names for known faces
known_faces_names = [
    "Elon_Musk",
    "Vamshi"
]

## Image Processing

# Initializing variables
faces_locations = []
faces_encodings = []
faces_names = []
process_frame = True

while True:
    # Picks single frame of video
    ret, frame = cap.read()

    # Resizes the video frame  to 1/4 for faster face recognition 
    resize_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converts the image from BGR (openCv) tp RGB (face_recognition)
    rgb_resize_frame = resize_frame[:, :, ::-1]


    # Processes every other frame of video to save time
    if process_frame:
        # Detects the faces and encodings of it in the cureent video frame
        faces_locations = face_recognition.face_locations(rgb_resize_frame)
        faces_encodings = face_recognition.face_encodings(rgb_resize_frame, faces_locations)

        faces_names = []
        for face_encoding in faces_encodings:
            # Searches for data of known face
            face_matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            name = "Unknown"

            
            # If FACE matches in the known_faces_encodings then follows
            # Instead,Use the known face for the small distance to the new face
            face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_match = np.argmin(face_distances)
            if face_matches[best_match]:
                name = known_faces_names[best_match]

            faces_names.append(name)

    process_frame = not process_frame


    # Displays the Result
    for (top, right, bottom, left), name in zip(faces_locations, faces_names):
        # Resizes to the full frame which resized previous
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draws Rectangular box arounnd face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draws the label with  name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Webcam Releases from Task
cap.release()

cv2.destroyAllWindows()