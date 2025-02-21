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

save_img_dir="./static/images"
image_dir="./augumented_dataset/augmented_dataset"
# json_path="./description.json"
for_displaying_image="./static/images"
faiss_index_path = "./karana.index"
metadata_path = "./metadata.pth"

os.makedirs(for_displaying_image,exist_ok=True)

def query_to_embedding(query_text):
    inputs = tokenizer(query_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        query_embedding = text_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return query_embedding

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model.eval()

index = faiss.read_index(faiss_index_path)
metadata = torch.load(metadata_path, map_location="cpu")

# def retrieve_top_images_from_text(query_text, k=1):

#     query_embedding = query_to_embedding(query_text)

#     D, I = index.search(query_embedding.reshape(1, -1), k=k)

#     for rank in range(k):
#         closest_text_index = I[0][rank]  
#         print(f"Closest Text Index: {closest_text_index}")
#         distance = D[0][rank]  

#         relative_path = metadata[closest_text_index]
#         print(f"Retrieved Path from Metadata: {relative_path}")

#         folder_name = os.path.basename(os.path.dirname(relative_path))
#         print(f"Folder Name: {folder_name}")

#         closest_image_path = os.path.join(image_dir, relative_path).replace("\\", "/")
#         print(f"Rank {rank + 1}: Image Location: {closest_image_path}, Distance: {distance}")

#         if os.path.exists(closest_image_path):
#             image = Image.open(closest_image_path)
#             save_path=os.path.join(save_img_dir,"img_retrieved.png")
#             image.save(save_path)
#         else:
#             save_path=os.path.join(save_img_dir,"img_retrieved.png")
#             print(os.path.abspath(save_path))
#             print(f"Image not found: {closest_image_path}")
#     return "done"

def retrieve_top_images_from_text(query_text, k=1):
    folder_names = []  # List to store folder names
    
    query_embedding = query_to_embedding(query_text)

    D, I = index.search(query_embedding.reshape(1, -1), k=k)

    for rank in range(k):
        closest_text_index = I[0][rank]
        print(f"Closest Text Index: {closest_text_index}")
        distance = D[0][rank]  

        relative_path = metadata[closest_text_index]
        print(f"Retrieved Path from Metadata: {relative_path}")

        folder_name = os.path.basename(os.path.dirname(relative_path))
        print(f"Folder Name: {folder_name}")
        folder_names.append(folder_name)  # Append the folder name to the list

        closest_image_path = os.path.join(image_dir, relative_path).replace("\\", "/")
        print(f"Rank {rank + 1}: Image Location: {closest_image_path}, Distance: {distance}")

        if os.path.exists(closest_image_path):
            image = Image.open(closest_image_path)
            save_path = os.path.join(save_img_dir, "img_retrieved.png")
            image.save(save_path)
        else:
            save_path = os.path.join(save_img_dir, "img_retrieved.png")
            print(os.path.abspath(save_path))
            print(f"Image not found: {closest_image_path}")

    # Return the folder names as well, along with other data
    return folder_names
