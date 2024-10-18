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

# Output folder where the images with predictions will be saved
output_folder = "predicted_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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

# Function to predict the pose and save the result
def predict_and_save(image_path):
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
    
    # Display the predicted class name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Predicted: {predicted_class_name}', (30, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Save the image with the prediction result
    output_path = os.path.join(output_folder, os.path.basename(image_path))  # Save to the output folder
    cv2.imwrite(output_path, image)
    print(f"Saved prediction for {image_path} as {output_path}")

# Function to process a folder of test images
def process_test_folder(test_folder):
    # Get all image files from the folder
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        
        # Process only if it's a valid image file
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            predict_and_save(img_path)

# Test with a folder of images
test_folder = 'test'  # Replace with the path to the folder containing test images
process_test_folder(test_folder)
