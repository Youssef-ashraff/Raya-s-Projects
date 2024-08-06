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
        frame_resized = resize_image(frame, 800)  # Adjust max_size as needed
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
def main(image_path, folder_path, is_video=False):
    try:
        if is_video:
            compare_images(image_path, folder_path, is_video=True)
        else:
            images = load_images_from_folder(folder_path)
            face_encodings = get_face_encodings(images)
            unknown_image = face_recognition.load_image_file(image_path)
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
            matches = compare_faces(face_encodings, unknown_face_encoding)
            display_matched_images(matches, folder_path)
    except Exception as e:
        print(f"Error occurred: {e}")


# Initialize Tkinter window
root = Tk()
root.title("Face Recognition")
root.geometry("1200x800")

# Create a scrollable canvas with horizontal and vertical scrollbars
frame_canvas = Canvas(root, bg="#F0F8FF")
frame_container = Frame(frame_canvas, bg="#F0F8FF")
scrollbar_y = Scrollbar(root, orient="vertical", command=frame_canvas.yview)
scrollbar_x = Scrollbar(root, orient="horizontal", command=frame_canvas.xview)

frame_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

scrollbar_y.pack(side="right", fill="y")
scrollbar_x.pack(side="bottom", fill="x")
frame_canvas.pack(side="left", fill="both", expand=True)
frame_canvas.create_window((0, 0), window=frame_container, anchor="nw")


def update_scrollregion(event):
    frame_canvas.configure(scrollregion=frame_canvas.bbox("all"))


frame_container.bind("<Configure>", update_scrollregion)

# Create UI elements for selecting files and folders
label1 = Label(root, text="Select Image or Video:")
label1.pack(pady=10)

image_path_entry = Entry(root, width=80)
image_path_entry.pack(pady=5)

browse_image_button = Button(root, text="Browse Image/Video", command=lambda: browse_file(image_path_entry))
browse_image_button.pack(pady=5)

folder_path_entry = Entry(root, width=80)
folder_path_entry.pack(pady=5)

browse_folder_button = Button(root, text="Browse Folder", command=lambda: browse_folder(folder_path_entry))
browse_folder_button.pack(pady=5)

compare_button = Button(root, text="Compare",
                        command=lambda: main(image_path_entry.get(), folder_path_entry.get(), is_video=video_var.get()))
compare_button.pack(pady=20)

result_frame = Frame(root, bg="#F0F8FF")
result_frame.pack(fill="both", expand=True)

label2 = Label(root)
label2.pack(pady=20)

video_var = IntVar()
video_checkbox = Checkbutton(root, text="Is Video", variable=video_var)
video_checkbox.pack(pady=10)


def browse_file(entry):
    filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.mp4")])
    if filename:
        entry.delete(0, END)
        entry.insert(0, filename)


def browse_folder(entry):
    foldername = filedialog.askdirectory()
    if foldername:
        entry.delete(0, END)
        entry.insert(0, foldername)


# Run the Tkinter event loop
root.mainloop()
