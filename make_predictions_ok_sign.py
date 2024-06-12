from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import sqlite3

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the gesture detection model
gesture_model = load_model("models/keras_Model.h5", compile=False)

# Load the gesture labels
gesture_class_names = open("models/labels.txt", "r").readlines()

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Font and display settings
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30
nametagColor = (100, 180, 0)
nametagHeight = 50
faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Start looping
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face found
    for (x, y, w, h) in faces:
        # Recognize the face
        customer_uid, Confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])

        # Connect to SQLite database
        try:
            conn = sqlite3.connect('customer_faces_data.db')
            c = conn.cursor()
        except sqlite3.Error as e:
            print("SQLite error:", e)
            continue

        c.execute("SELECT customer_name, confirmed FROM customers WHERE customer_uid = ?", (customer_uid,))
        row = c.fetchone()
        if row:
            customer_name = row[0].split(" ")[0]
            confirmed = row[1]
        else:
            customer_name = "Unknown"
            confirmed = 0

        print(f"Detected face: {customer_name}, UID: {customer_uid}, Confidence: {Confidence}")

        if 45 < Confidence < 85:
            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleBorderColor, faceRectangleBorderSize)

            # Display name tag
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, str(customer_name) + ": " + str(round(Confidence, 2)) + "%", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

    # Perform gesture detection on the same frame
    gesture_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    gesture_image = np.asarray(gesture_image, dtype=np.float32).reshape(1, 224, 224, 3)
    gesture_image = (gesture_image / 127.5) - 1

    # Predict the gesture
    gesture_prediction = gesture_model.predict(gesture_image)
    gesture_index = np.argmax(gesture_prediction)
    gesture_class_name = gesture_class_names[gesture_index]
    gesture_confidence_score = gesture_prediction[0][gesture_index]

    # Display gesture prediction and confidence score
    cv2.putText(frame, "Gesture: " + gesture_class_name[2:], (10, 30), fontFace, fontScale, fontColor, fontWeight)
    cv2.putText(frame, "Confidence: " + str(np.round(gesture_confidence_score * 100))[:-2] + "%", (10, 60), fontFace, fontScale, fontColor, fontWeight)

    # Update the database if "okay_sign" gesture is detected
    if gesture_class_name.strip() == "okay_sign":
        print("Okay sign detected with confidence:", gesture_confidence_score)
        if customer_uid is not None:
            print(f"Updating database for customer UID: {customer_uid}")
            try:
                c.execute("UPDATE customers SET confirmed = 1 WHERE customer_uid = ?", (customer_uid,))
                conn.commit()
                print("Database updated successfully.")
            except sqlite3.Error as e:
                print("SQLite update error:", e)

    # Display the resulting frame
    cv2.imshow('Face and Gesture Detection', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
# Close the database connection
conn.close()
