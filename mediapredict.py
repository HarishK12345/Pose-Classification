import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import mediapipe as mp
import os
# Load the trained model
model = load_model('newdataset.h5')

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Class names (ensure these match the class names used during training)
class_names = sorted(os.listdir(r'newdataset'))  # Same folder structure used for training

# Function to extract landmarks from an image (same as used during training)
def extract_pose_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    
    if result.pose_landmarks:
        landmarks = []
        for landmark in result.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).flatten()  # Flatten to a 1D array
    else:
        return np.zeros(99)  # 33 landmarks * 3 (x, y, z)

# Function to predict the pose and display the image with the result
def predict_and_display(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    
    # Extract pose landmarks
    landmarks = extract_pose_landmarks(image)
    
    # Reshape the landmarks for model input
    landmarks = landmarks.reshape(1, -1)  # Reshape to match input shape
    
    # Predict the class (pose)
    prediction = model.predict(landmarks)
    
    # Get the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    # Display the image with predicted class name
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Predicted: {predicted_class_name}', (30, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the image with the prediction result
    cv2.imshow('Predicted Pose', image)
    
    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with a new image
image_path = 'leenam.png'  # Replace with the path to the image you want to predict
predict_and_display(image_path)
