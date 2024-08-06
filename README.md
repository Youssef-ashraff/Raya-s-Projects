# Raya-s-Projects
People Counter and Face Recognition System

This project demonstrates my skills in real-time image capture, face detection, face recognition technology & Machine learning. It provides practical applications for security and personal use by combining several advanced technologies.
### Real-time People Counting:
Utilizes YOLO (You Only Look Once) object detection model, OpenCV, and SORT (Simple Online and Realtime Tracking) algorithm to track and count people in a video feed or from a webcam. This feature is ideal for monitoring entries and exits in a specified area.
### Face Recognition Integration:
Employs face recognition technology to identify and display matching faces from a database, enhancing security measures or personal identification tasks.
### Graphical User Interface (GUI): 
Incorporates a user-friendly GUI to interactively display results, making it easy to use for non-technical users. This interface provides real-time feedback and visual representation of detected and recognized faces.

This Project includes a few of python files try running them after installing the required libraries
## FaceDetection.py:
identify faces in images either captured from a webcam or provided via an image file. The tool uses Haar Cascades for face detection, resizing images to manageable dimensions for efficient processing. Users can capture images from their webcam with a single click or load an image from their computer, and the program will detect and highlight faces in the image.
## FaceComparison.py:
processes two selected images, detects faces, and encodes them for comparison. The results are then displayed in a Tkinter window, with visual markers on the detected faces and a text output indicating whether the faces match. This project showcases my ability to integrate face recognition technology with a graphical user interface for an interactive user experience.
## FolderLooping:
allows users to compare a selected image with images in a specified folder to find matching faces. The tool utilizes the face_recognition library to detect and encode faces, and then compares them based on a defined tolerance level. A Tkinter-based graphical user interface makes it easy to select the image and folder, and displays the matched images along with their similarity distances
## Final.py:
It is a Combination of the FaceComparison & FolderLooping with few buttons using GUI development
## PeopleCounter.py & PeopleCounterLive.py:
I created a sophisticated people counting system using YOLO (You Only Look Once) object detection model, OpenCV, and SORT (Simple Online and Realtime Tracking) algorithm. This project can track and count people in a video feed or from a webcam in real-time. The system detects people crossing predefined lines, differentiating between entries and exits, and logs the counts along with timestamps. The tool uses a YOLOv8 model to identify people and other objects, applies a mask to focus on a specific region of interest, and uses the SORT algorithm to maintain tracking of detected individuals.






