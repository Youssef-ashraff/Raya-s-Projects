import face_recognition
import os
import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
from tkinter import filedialog


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


def get_face_encodings(images):
    encodings = []
    for filename, image in images:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        for face_encoding in face_encodings:
            encodings.append((filename, face_encoding))
    return encodings


def compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.53):
    distances = face_recognition.face_distance([encoding for _, encoding in known_face_encodings],
                                               unknown_face_encoding)
    matches = [(known_face_encodings[i][0], distances[i]) for i, distance in enumerate(distances) if
               distance <= tolerance]
    return matches


def display_matched_images(matches, folder_path):
    for widget in result_frame.winfo_children():
        widget.destroy()

    for i, (filename, distance) in enumerate(matches):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(result_frame, image=img_tk)
        label.image = img_tk
        label.grid(row=i, column=0, padx=10, pady=10)

        text_label = tk.Label(result_frame, text=f"{filename} (distance: {distance:.2f})")
        text_label.grid(row=i, column=1, padx=10, pady=10)


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


def select_image():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        return filename
    return None


def select_folder():
    foldername = filedialog.askdirectory(title="Select Folder")
    return foldername


def select_and_compare():
    image_path = select_image()
    if image_path:
        folder_path = select_folder()
        if folder_path:
            main(image_path, folder_path, tolerance=0.55)


root = tk.Tk()
root.title("Face Comparison")
root.geometry("1000x800")

button = tk.Button(root, text="Select Image and Folder", command=select_and_compare)
button.pack(pady=20)

result_frame = tk.Frame(root)
result_frame.pack(pady=10)

root.mainloop()
