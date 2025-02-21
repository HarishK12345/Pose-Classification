import faiss
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pickle
from IPython.display import display

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

image_dir=r"D:\Sem7-project-folders\augmented_dataset\augmented_dataset"
json_path=r"DD:\Arun\SSN\FYP\aj-sanskrit\description_sanskrit.json"
for_displaying_image=r"D:\Arun\SSN\FYP\aj-sanskrit\saved-images"


# Ensure correct formatting of the FAISS index path
faiss_index_path = r"D:\Arun\SSN\FYP\aj-sanskrit\Karana_sanskrit.index"
metadata_path = r"D:\Arun\SSN\FYP\aj-sanskrit\metadata_sanskrit.pth"

text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index(faiss_index_path)
metadata = torch.load(metadata_path, map_location="cpu")

# Function to convert a query text to an embedding
def query_to_embedding(query_text):

    # Convert the query text to an embedding with truncation to max length
    inputs = tokenizer(query_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        query_embedding = text_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    # print("query embedding:",query_embedding)
    return query_embedding

def retrieve_top_images_from_text(query_text, k=1):
    # Convert the query to a text vector
    query_embedding = query_to_embedding(query_text)

    # Search Faiss index for the closest k matching text vectors
    D, I = index.search(query_embedding.reshape(1, -1), k=k)

    for rank in range(k):
        closest_text_index = I[0][rank]  # Get the index of the current closest match
        print(f"Closest Text Index: {closest_text_index}")
        distance = D[0][rank]  # Get the corresponding distance score

        # Retrieve the full relative path from metadata
        relative_path = metadata[closest_text_index]
        print(f"Retrieved Path from Metadata: {relative_path}")

        # Construct the full absolute path using `image_dir`
        closest_image_path = os.path.join(image_dir, relative_path).replace("\\", "/")
        print(f"Rank {rank + 1}: Image Location: {closest_image_path}, Distance: {distance}")

        # Load and display the image along with its score
        if os.path.exists(closest_image_path):
            image = Image.open(closest_image_path)
            save_path=os.path.join(for_displaying_image,"img_retrieved.png")
            image.save(save_path)
            # display(image)  # Display the image inline
        else:
            print(f"Image not found: {closest_image_path}")

# Example usage
query = "समन्नतरश्चवसौष्ठवनामतभ्द्वत |वामऩष्ऩऩट्ऩाश्वऩादोॱग्रतऱसउञचर् ||६१||"
retrieve_top_images_from_text(query)
