import cv2 as cv

# Function to resize images to a manageable size
def resize_image(image, min_width=250, min_height=250, max_width=500, max_height=500):
    height, width = image.shape[:2]
    scaling_factor = 1

    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
    elif width < min_width or height < min_height:
        scaling_factor = max(min_width / width, min_height / height)

    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    resized_image = cv.resize(image, new_size, interpolation=cv.INTER_AREA if scaling_factor < 1 else cv.INTER_LINEAR)
    return resized_image

# Function to detect humans in an image
def detect_human(img):
    # Load the Haar Cascade for face detection
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Resize the image to a manageable size
    img = resize_image(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # If faces are found, we consider that the image contains a human
    if len(faces) > 0:
        return len(faces), img
    return 0, img

# Function to capture image from webcam with a click
def capture_image_from_webcam_with_click():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv.imshow('Press SPACE to capture', frame)

        # Wait for a key press
        key = cv.waitKey(1)
        if key == 32:  # Space bar to capture
            cap.release()
            cv.destroyAllWindows()
            return frame
        elif key == 27:  # Escape key to exit without capturing
            cap.release()
            cv.destroyAllWindows()
            return None

# Function to get image from either webcam or file path
def get_image(source_type, image_path=None):
    if source_type == 'camera':
        return capture_image_from_webcam_with_click()
    elif source_type == 'file' and image_path:
        return cv.imread(image_path)
    else:
        print("Invalid source type or image path not provided.")
        return None

# Main code
source_type = input("Enter 'camera' to capture from webcam or 'file' to provide an image path: ").strip().lower()

if source_type == 'file':
    image_path = input("Enter the image path: ").strip()
    img = get_image(source_type, image_path)
else:
    img = get_image(source_type)

if img is not None:
    num_faces, result_img = detect_human(img)
    if num_faces > 0:
        print(f"The image contains {num_faces} face(s).")
        cv.imshow('Human detected', result_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("The image does not contain a human.")
        cv.imshow('No Human detected', result_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
else:
    print("No image captured.")

    #C:\Users\future\Pictures\Saved Pictures\Family Picture.jpeg