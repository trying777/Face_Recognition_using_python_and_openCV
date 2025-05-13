import face_recognition
import cv2

# Define file paths for the images
obama_image_path = "obama.jpg"
biden_image_path = "C:/face_recognition-master/examples/knn_examples/train/biden/biden.jpg"
alex_lacamoire_image_path = "C:/face_recognition-master/examples/knn_examples/train/alex_lacamoire/alex-lacamoire.png"
kit_harington_image_path = "C:/face_recognition-master/examples/knn_examples/train/kit_harington/john1.jpeg"
jungkook_image_path = "C:/face_recognition-master/examples/knn_examples/train/jungkook/jungkook.jpg"
shahrukh_khan_image_path = "C:/face_recognition-master/examples/knn_examples/train/shahrukh_khan/shahrukh khan.jpg"
mahesh_babu_image_path = "C:/face_recognition-master/examples/knn_examples/train/mahesh_babu/mahesh.jpg"
varsha_image_path = "C:/face_recognition-master/examples/knn_examples/train/varsha/varsha.jpeg"

# Load images and encode known faces
obama_image = face_recognition.load_image_file(obama_image_path)
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file(biden_image_path)
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

alex_lacamoire_image = face_recognition.load_image_file(alex_lacamoire_image_path)
alex_lacamoire_face_encoding = face_recognition.face_encodings(alex_lacamoire_image)[0]

kit_harington_image = face_recognition.load_image_file(kit_harington_image_path)
kit_harington_face_encoding = face_recognition.face_encodings(kit_harington_image)[0]

jungkook_image = face_recognition.load_image_file(jungkook_image_path)
jungkook_face_encoding = face_recognition.face_encodings(jungkook_image)[0]

shahrukh_khan_image = face_recognition.load_image_file(shahrukh_khan_image_path)
shahrukh_khan_face_encoding = face_recognition.face_encodings(shahrukh_khan_image)[0]

mahesh_babu_image = face_recognition.load_image_file(mahesh_babu_image_path)
mahesh_babu_face_encoding = face_recognition.face_encodings(mahesh_babu_image)[0]

varsha_image = face_recognition.load_image_file(varsha_image_path)
varsha_face_encoding = face_recognition.face_encodings(varsha_image)[0]

known_face_encodings = [obama_face_encoding, biden_face_encoding, alex_lacamoire_face_encoding, kit_harington_face_encoding, jungkook_face_encoding, shahrukh_khan_face_encoding, mahesh_babu_face_encoding, varsha_face_encoding]
known_face_names = ["Barack Obama", "Joe Biden", "Alex Lacamoire", "Kit Harington", "Jungkook", "Shahrukh Khan", "Mahesh Babu", "Varsha"]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame to save time
    if process_this_frame:
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare each face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Check for best match
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
