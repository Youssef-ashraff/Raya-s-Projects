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


# Function to display images
def display_images():
    # Load and process the first image
    image1 = face_recognition.load_image_file(r"C:\Users\future\Pictures\Camera Roll\Ziad.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    location1 = face_recognition.face_locations(image1)[0]
    encode1 = face_recognition.face_encodings(image1)[0]
    rectangle1 = cv2.rectangle(image1, (location1[3], location1[0]), (location1[1], location1[2]), (0, 0, 255), 2)

    # Resize the first image
    image1 = resize_image(image1, 400)

    # Load and process the second image
    image2 = face_recognition.load_image_file(r"C:\Users\future\Pictures\Camera Roll\Ziad Id.jpeg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    location2 = face_recognition.face_locations(image2)[0]
    encode2 = face_recognition.face_encodings(image2)[0]
    rectangle2 = cv2.rectangle(image2, (location2[3], location2[0]), (location2[1], location2[2]), (255, 0, 0), 2)

    # Resize the second image
    image2 = resize_image(image2, 400)

    # Compare faces
    result = face_recognition.compare_faces([encode1], encode2, 0.6)
    distance = face_recognition.face_distance([encode1], encode2)

    if result == [True]:
        print("Same person")
    else:
        print("Different person")

    # Convert images to a format Tkinter can handle
    image1 = Image.fromarray(image1)
    image2 = Image.fromarray(image2)
    image1_tk = ImageTk.PhotoImage(image1)
    image2_tk = ImageTk.PhotoImage(image2)

    # Add the images to the window
    label1.config(image=image1_tk)
    label1.image = image1_tk  # Keep a reference to avoid garbage collection
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
