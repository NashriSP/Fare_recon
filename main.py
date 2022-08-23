from ast import NotIn
from datetime import datetime
import cv2
import os
import sys
import numpy as np
import csv
from datetime import datetime
import face_recognition
# # # # Register our cam
def main():
    # # # # Path sortcut for project's dir)
    script_dir = sys.path[0]
    students = os.listdir('students')
    # print(students)
    # img_path = os.path.join(script_dir, f'students\{students[0]}')
    # print(img_path)
    # print(os.listdir(img_path))
    # # # # # data handler
    face_path = []
    student_name = []
    for student in students:
        sFile = os.path.join(script_dir, f'students\{student}')
        sFaces = os.listdir(sFile)
        # print(f"{sFile}\{sFaces[0]}")
        for face in sFaces:
            # print(faces)
            face_path.append(cv2.imread(f"{sFile}\{face}"))
            # name = os.path.splitext(face)[0]
            student_name.append(student)
    # print(student_name)
    # # # # encode faces function
    def encode_face(face_path):
        encoded_list = []
        for face in face_path:
            face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
            encode = face_recognition.face_encodings(face)[0]
            encoded_list.append(encode)
        return encoded_list
    # # # # pre start the program so we can encode all the data first
    print("Encoding in Progress may take some time :)")
    encode_list = encode_face(face_path)
    print("Encoding Completed!!!")
    # # # # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # # # # Register our cam
    video_capture = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
    
    def attn(name):
        with open("name.csv", "r+") as f:
            mydatlist = f.readlines()
            mydat=[]
            for line in mydatlist:
                entry = line.split(",")
                mydat.append(entry[0])
            if name not in mydat:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    while True:
        # # # # Grab a frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(encode_list, face_encoding)
                name = "Unknown"
                # # If a match was found in encode_list, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = student_name[first_match_index]
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(encode_list, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = student_name[best_match_index]
                face_names.append(name)
                attn(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()