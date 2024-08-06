import os
import face_recognition
import cv2
from tkinter import *
from PIL import Image, ImageTk


# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size):
    height, width, _ = image.shape
    scale = min(max_size / width, max_size / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))


# Function to extract frames from video
def get_frames_from_video(video_path, times_in_seconds):
    if not os.path.exists(video_path):
        raise Exception(f"Video file does not exist: {video_path}")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    frames = []
    for time_in_seconds in times_in_seconds:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
        success, frame = video_capture.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            print(f"Could not extract frame at {time_in_seconds} seconds")

    return frames


# Function to display images
def display_images():
    # Extract frames from the video at multiple time points
    video_frames = get_frames_from_video(r"C:\Users\future\Pictures\Camera Roll\haidy's.mp4", [1, 2, 3, 4])

    # Process the first frame for face detection and encoding
    face_locations1 = face_recognition.face_locations(video_frames[0])
    if face_locations1:
        encode1 = face_recognition.face_encodings(video_frames[0])[0]
        rectangle1 = cv2.rectangle(video_frames[0], (face_locations1[0][3], face_locations1[0][0]),
                                   (face_locations1[0][1], face_locations1[0][2]), (255, 0, 0), 2)
        # Resize the first frame
        video_frame_resized = resize_image(video_frames[0], 400)
    else:
        raise Exception("No face detected in the first frame of the video")

    # Load and process the second image
    image2 = face_recognition.load_image_file(r"C:\Users\future\Pictures\Camera Roll\haidy's.jpeg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    face_locations2 = face_recognition.face_locations(image2)
    if face_locations2:
        encode2 = face_recognition.face_encodings(image2)[0]
        rectangle2 = cv2.rectangle(image2, (face_locations2[0][3], face_locations2[0][0]),
                                   (face_locations2[0][1], face_locations2[0][2]), (255, 0, 0), 2)
        # Resize the second image
        image2 = resize_image(image2, 600)
    else:
        raise Exception("No face detected in the second image")

    # Compare faces using multiple frames
    match_found = False
    for frame in video_frames:
        face_locations_frame = face_recognition.face_locations(frame)
        if face_locations_frame:
            encode_frame = face_recognition.face_encodings(frame)[0]
            result = face_recognition.compare_faces([encode_frame], encode2, 0.7)
            if result == [True]:
                match_found = True
                break

    if match_found:
        print("Same person")
    else:
        print("Different person")

    # Convert images to a format Tkinter can handle
    video_frame_resized = Image.fromarray(video_frame_resized)
    image2 = Image.fromarray(image2)
    video_frame_tk = ImageTk.PhotoImage(video_frame_resized)
    image2_tk = ImageTk.PhotoImage(image2)

    # Add the images to the window
    label1.config(image=video_frame_tk)
    label1.image = video_frame_tk  # Keep a reference to avoid garbage collection
    label2.config(image=image2_tk)
    label2.image = image2_tk  # Keep a reference to avoid garbage collection


# Create a Tkinter window
root = Tk()
root.title("The 2 pictures Result")

# Create labels for the images
label1 = Label(root)
label1.pack(side="left", padx=10, pady=10)
label2 = Label(root)
label2.pack(side="right", padx=10, pady=10)

# Call the function to display images
display_images()

# Start the Tkinter event loop
root.mainloop()
