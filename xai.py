import numpy as np
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/content/drive/MyDrive/cnn_pose_model3.h5')

# Mediapipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Example class names
class_names = [
    'Aavartam', 'Akshiptam', 'Akshiptarechitam', 'Alaatakam', 'Anchitam', 'Apakraantam', 'Apaviddham',
    # ... (remaining classes) ...
    'Vrishchikarechitam', 'Vyamsitam'
]

def extract_pose_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    
    if result.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        return landmarks.reshape(33, 3)
    else:
        return np.zeros((33, 3))

def explain_image(image_path):
    # Load and preprocess the image
    orig_image = cv2.imread(image_path)
    landmarks_2d = extract_pose_landmarks(orig_image)  # shape (33, 3)
    
    # Prepare data for CNN: (batch, height, width, channel=1)
    landmarks_for_model = landmarks_2d.reshape(1, 33, 3, 1)
    
    # Create pseudo-RGB for LIME: shape (1, 33, 3, 3)
    landmarks_for_lime = np.repeat(landmarks_for_model, 3, axis=-1)
    # Remove batch dimension for LIME segmentation: shape (33, 3, 3)
    lime_input_image = landmarks_for_lime[0]
    
    # Define a custom prediction function
    def model_predict(input_images):
        # Convert from pseudo-RGB (n, 33, 3, 3) back to (n, 33, 3, 1)
        input_single_channel = input_images[..., :1]
        return model.predict(input_single_channel)
    
    # Initialize LIME Image Explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate LIME explanation on the pseudo-RGB image
    explanation = explainer.explain_instance(
        lime_input_image.astype('double'),
        model_predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )
    
    # (Optional) Show the standard LIME output for reference
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=True
    )
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title(f"Explanation for {class_names[explanation.top_labels[0]]} with mask")
    plt.show()
    
    # -------------------------------
    # NEW: Overlay only the most important landmarks on the original image
    # -------------------------------
    
    # Build a matrix of importance for each pixel (i, j)
    segment_weights = dict(explanation.local_exp[explanation.top_labels[0]])
    pixel_importance = np.zeros_like(explanation.segments, dtype=float)
    for i in range(33):
        for j in range(3):
            seg_id = explanation.segments[i, j]
            pixel_importance[i, j] = segment_weights.get(seg_id, 0.0)
    
    # Aggregate importance for each landmark across the 3 coordinates (x, y, z)
    # Here we simply sum the values (you might consider absolute values or another metric)
    landmark_importance = np.sum(pixel_importance, axis=1)  # shape (33,)
    
    # Select the top 5 landmarks based on their aggregated importance
    num_top = 5
    top_indices = np.argsort(landmark_importance)[-num_top:]
    
    # Overlay only the top important landmarks on the original image
    h, w, _ = orig_image.shape
    for i in top_indices:
        # Convert normalized x,y coordinates to pixel values
        x_px = int(landmarks_2d[i, 0] * w)
        y_px = int(landmarks_2d[i, 1] * h)
        
        # Choose color: green for positive contribution, red for negative
        imp_val = landmark_importance[i]
        color = (0, 255, 0) if imp_val >= 0 else (0, 0, 255)
        
        # Scale radius by the magnitude of importance (adjust the multiplier as needed)
        base_radius = 3
        radius = int(base_radius + 10 * abs(imp_val))
        radius = max(radius, 1)
        
        cv2.circle(orig_image, (x_px, y_px), radius, color, thickness=2)
        cv2.putText(orig_image, f"{i}:{imp_val:.2f}", (x_px, y_px-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Annotate with predicted class label
    pred_class = class_names[explanation.top_labels[0]]
    cv2.putText(orig_image, f"LIME: {pred_class}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Display the annotated image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Important Keypoints for Predicted Class")
    plt.show()

# Example usage
test_image = r"/content/drive/MyDrive/aavartam.jpg"
explain_image(test_image)
