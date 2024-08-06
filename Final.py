import os
import face_recognition
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, UnidentifiedImageError
import numpy as np


# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size):
    height, width, _ = image.shape
    scale = min(max_size / width, max_size / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))


# Function to extract frames from video
def get_frames_from_video(video_path, num_frames):
    if not os.path.exists(video_path):
        raise Exception(f"Video file does not exist: {video_path}")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) / int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_times = [i * video_length / (num_frames + 1) for i in range(1, num_frames + 1)]

    frames = []
    for time_in_seconds in frame_times:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
        success, frame = video_capture.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            print(f"Could not extract frame at {time_in_seconds} seconds")

    return frames


# Function to compare two frames and determine if they are the same
def are_frames_similar(frame1, frame2):
    encodings1 = face_recognition.face_encodings(frame1)
    encodings2 = face_recognition.face_encodings(frame2)

    if not encodings1 or not encodings2:
        return False

    results = face_recognition.compare_faces([encodings1[0]], encodings2[0], 0.53)
    return results[0]


# Function to process and compare images or frames
def compare_images(image1_path, image2_path, is_video=False):
    video_frames = []
    frame_encodings = []
    image2_resized = None

    try:
        if is_video:
            video_frames = get_frames_from_video(image1_path, 10)
            for i in range(len(video_frames)):
                face_locations = face_recognition.face_locations(video_frames[i])
                if face_locations:
                    encode = face_recognition.face_encodings(video_frames[i])[0]
                    frame_encodings.append(encode)
                    cv2.rectangle(video_frames[i], (face_locations[0][3], face_locations[0][0]),
                                  (face_locations[0][1], face_locations[0][2]), (255, 0, 0), 2)
                else:
                    print(f"No face detected in frame {i + 1}")

            # Load and process the second image
            image2 = face_recognition.load_image_file(image2_path)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            face_locations2 = face_recognition.face_locations(image2)
            if face_locations2:
                encode2 = face_recognition.face_encodings(image2)[0]
                cv2.rectangle(image2, (face_locations2[0][3], face_locations2[0][0]),
                              (face_locations2[0][1], face_locations2[0][2]), (255, 0, 0), 2)
                image2_resized = resize_image(image2, 600)

                # Compare every pair of frames
                live_video_verified = False
                for i in range(len(video_frames)):
                    for j in range(i + 1, len(video_frames)):
                        if not are_frames_similar(video_frames[i], video_frames[j]):
                            live_video_verified = True
                            break
                    if live_video_verified:
                        break

                if live_video_verified:
                    print("Live Verified")
                else:
                    print("Not Live")

                    # Compare every frame encoding with the second image encoding
                same_person_found = False
                for frame_encoding in frame_encodings:
                    result = face_recognition.compare_faces([frame_encoding], encode2, 0.53)
                    if result[0]:
                        same_person_found = True
                        break

                if same_person_found:
                    print("Same person")
                else:
                    print("Different person")
            else:
                raise Exception("No face detected in the second image")

        else:
            # Process the images if not a video
            image1 = face_recognition.load_image_file(image1_path)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(image1)
            if face_locations:
                encode1 = face_recognition.face_encodings(image1)[0]
                frame_encodings.append(encode1)
                cv2.rectangle(image1, (face_locations[0][3], face_locations[0][0]),
                              (face_locations[0][1], face_locations[0][2]), (0, 0, 255), 2)
                video_frames.append(image1)
            else:
                raise Exception("No face detected in the first image")

            # Load and process the second image
            image2 = face_recognition.load_image_file(image2_path)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            face_locations2 = face_recognition.face_locations(image2)
            if face_locations2:
                encode2 = face_recognition.face_encodings(image2)[0]
                cv2.rectangle(image2, (face_locations2[0][3], face_locations2[0][0]),
                              (face_locations2[0][1], face_locations2[0][2]), (255, 0, 0), 2)
                image2_resized = resize_image(image2, 600)
                result = face_recognition.compare_faces([frame_encodings[0]], encode2, 0.55)
                if result[0]:
                    print("Same person")
                else:
                    print("Different person")
            else:
                raise Exception("No face detected in the second image")

    except Exception as e:
        print(f"Error occurred: {e}")

    # Display frames regardless of face detection
    display_frames(video_frames)

    # Display the second image if it was successfully resized
    if image2_resized is not None:
        image2_resized = Image.fromarray(image2_resized)
        image2_tk = ImageTk.PhotoImage(image2_resized)
        label2.config(image=image2_tk)
        label2.image = image2_tk  # Keep a reference to avoid garbage collection


# Function to display frames in a scrollable manner
def display_frames(frames):
    for widget in frame_container.winfo_children():
        widget.destroy()

    for i, frame in enumerate(frames):
        frame_resized = resize_image(frame, 300)
        frame_pil = Image.fromarray(frame_resized)
        frame_tk = ImageTk.PhotoImage(frame_pil)
        frame_label = Label(frame_container, image=frame_tk)
        frame_label.image = frame_tk  # Keep a reference to avoid garbage collection
        frame_label.grid(row=0, column=i, padx=5, pady=5)

    # Adjust the canvas scroll region
    frame_canvas.update_idletasks()
    frame_canvas.config(scrollregion=frame_canvas.bbox("all"))


# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = face_recognition.load_image_file(img_path)
            if img is not None:
                images.append((filename, img))
        except (PermissionError, UnidentifiedImageError):
            continue
    return images


# Function to get face encodings from images
def get_face_encodings(images):
    encodings = []
    for filename, image in images:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        for face_encoding in face_encodings:
            encodings.append((filename, face_encoding))
    return encodings


# Function to compare faces
def compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.53):
    distances = face_recognition.face_distance([encoding for _, encoding in known_face_encodings],
                                               unknown_face_encoding)
    matches = [(known_face_encodings[i][0], distances[i]) for i, distance in enumerate(distances) if
               distance <= tolerance]
    return matches


# Function to display matched images
def display_matched_images(matches, folder_path):
    for widget in result_frame.winfo_children():
        widget.destroy()

    for i, (filename, distance) in enumerate(matches):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)

        label = Label(result_frame, image=img_tk)
        label.image = img_tk
        label.grid(row=i, column=0, padx=10, pady=10)

        text_label = Label(result_frame, text=f"{filename} (distance: {distance:.2f})")
        text_label.grid(row=i, column=1, padx=10, pady=10)


# Main function to compare a single image with images in a folder
def main(image_path, folder_path, tolerance=0.53):
    print(f"Loading input image: {image_path}")
    input_image = face_recognition.load_image_file(image_path)
    input_face_locations = face_recognition.face_locations(input_image)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    if not input_face_encodings:
        print("No faces found in the input image.")
        return

    print(f"Found {len(input_face_encodings)} face(s) in the input image.")

    images = load_images_from_folder(folder_path)
    known_face_encodings = get_face_encodings(images)

    matched_images = []
    for i, input_face_encoding in enumerate(input_face_encodings):
        matches = compare_faces(known_face_encodings, input_face_encoding, tolerance)
        if matches:
            print(f"Match found for face {i + 1} with the following image(s):")
            matched_images.extend(matches)
        else:
            print(f"No matches found for face {i + 1}.")

    display_matched_images(matched_images, folder_path)


# Function to select an image
def select_image():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        return filename
    return None


# Function to select a folder
def select_folder():
    foldername = filedialog.askdirectory(title="Select Folder")
    return foldername


# Function to select an image and a folder, and then compare them
def select_and_compare():
    image_path = select_image()
    if image_path:
        folder_path = select_folder()
        if folder_path:
            main(image_path, folder_path, tolerance=0.55)


# Function to load files
def load_files():
    file_type = file_type_var.get()
    if file_type == "Image":
        image1_path = filedialog.askopenfilename(title="Select First Image",
                                                 filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        image2_path = filedialog.askopenfilename(title="Select Second Image",
                                                 filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image1_path and image2_path:
            compare_images(image1_path, image2_path, is_video=False)
    elif file_type == "Video":
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
        image2_path = filedialog.askopenfilename(title="Select Image File",
                                                 filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if video_path and image2_path:
            compare_images(video_path, image2_path, is_video=True)


# Create a Tkinter window
root = Tk()
root.title("Raya's Face Recognition")
root.geometry("1000x800")

# Set window background color
root.configure(bg="#F0F8FF")

# Create a frame for input options with blue background
frame = Frame(root, bg="#0000FF", padx=10, pady=10)
frame.pack(padx=10, pady=10)

# Load and display the logo image
try:
    logo_image = Image.open(
        r"C:\Users\future\Pictures\Camera Roll\Logo.png")  # Replace with the path to your logo image
    logo_image = logo_image.resize((100, 100), Image.LANCZOS)  # Resize the logo using LANCZOS filter
    logo_image_tk = ImageTk.PhotoImage(logo_image)
    logo_label = Label(frame, image=logo_image_tk, bg="#0000FF")
    logo_label.image = logo_image_tk  # Keep a reference to avoid garbage collection
    logo_label.pack(pady=10)
except Exception as e:
    print(f"Error loading logo: {e}")

# Create a title label with yellow text
title_label = Label(frame, text="Raya's Face Comparison Tool", font=("Helvetica", 18, "bold"), bg="#0000FF",
                    fg="#FFFF00")
title_label.pack(pady=10)

# Create radio buttons for selecting input type with a yellow background
file_type_var = StringVar(value="Image")
image_radio = Radiobutton(frame, text="Image", variable=file_type_var, value="Image", bg="#0000FF", fg="#FFFF00",
                          selectcolor="#000000", font=("Helvetica", 12))
video_radio = Radiobutton(frame, text="Video", variable=file_type_var, value="Video", bg="#0000FF", fg="#FFFF00",
                          selectcolor="#000000", font=("Helvetica", 12))
image_radio.pack(anchor="w")
video_radio.pack(anchor="w")

# Create a button to load files with a blue background and yellow text
load_button = Button(root, text="Load Files", command=load_files, bg="#0000FF", fg="#FFFF00",
                     font=("Helvetica", 14, "bold"), relief="raised", padx=20, pady=10)
load_button.pack(pady=10)

# Create a button to select image and folder for batch processing with a blue background and yellow text
select_button = Button(root, text="Select Image and Folder", command=select_and_compare, bg="#0000FF", fg="#FFFF00",
                       font=("Helvetica", 14, "bold"), relief="raised", padx=20, pady=10)
select_button.pack(pady=20)

# Create a scrollable frame container for the frames with blue background
frame_canvas = Canvas(root, bg="#F0F8FF")
frame_container = Frame(frame_canvas, bg="#F0F8FF")
scrollbar = Scrollbar(root, orient="horizontal", command=frame_canvas.xview)
# scrollbar = Scrollbar(root, orient="vertical", command=frame_canvas.yview)
frame_canvas.configure(xscrollcommand=scrollbar.set)

scrollbar.pack(side="bottom", fill="x")
frame_canvas.pack(side="top", fill="both", expand=True)
frame_canvas.create_window((0, 0), window=frame_container, anchor="nw")


def update_scrollregion(event):
    frame_canvas.configure(scrollregion=frame_canvas.bbox("all"))


frame_container.bind("<Configure>", update_scrollregion)

# Create a label for the second image with padding
label2 = Label(root, bg="#F0F8FF", padx=10, pady=10)
label2.pack(side="right", padx=10, pady=10)

# Create a frame to display matched images with padding and a yellow background
result_frame = Frame(root, bg="#FFFF00", padx=10, pady=10)
result_frame.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
