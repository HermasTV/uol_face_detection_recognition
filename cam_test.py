from detection.detectors import Detector

import cv2
import numpy as np
import time 

if __name__ == '__main__':

    face_detector = Detector("mediapipe")
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    # set text font
    font = cv2.FONT_HERSHEY_SIMPLEX
    process_this_frame = True
    while video_capture.isOpened():
        prev_time = time.time()
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = res1 = face_detector.detect(rgb_small_frame)
        for (x1, y1, x2, y2) in face_locations:

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        exec_time ="fps : " + str(int((time.time()-prev_time)*1000)) + " ms"
        # putting the FPS count on the frame
        cv2.putText(frame, exec_time, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()