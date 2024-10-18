import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Paths
data_dir = r"augmented_newdataset"  # Replace with your folder path

# Function to extract landmarks
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

# Load dataset and extract landmarks
def load_dataset(data_dir):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)

            landmarks = extract_pose_landmarks(image)
            X.append(landmarks)
            y.append(label)

    return np.array(X), np.array(y), class_names

# Load data
X, y, class_names = load_dataset(data_dir)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train_cat = to_categorical(y_train, num_classes=len(class_names))
y_test_cat = to_categorical(y_test, num_classes=len(class_names))

# Build a simple classifier model
model = Sequential([
    Dense(128, activation='relu', input_shape=(99,)),  # 33 landmarks * 3 (x, y, z)
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')  # 108 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=120, batch_size=32)

# Save the model
model.save('newdataset.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
