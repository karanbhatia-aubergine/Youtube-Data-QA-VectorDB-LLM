import os
import cv2
import numpy as np
from decouple import config

def generate_embedding(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img.flatten()


image_folder_path = config("IMAGE_FOLDER_PATH")

image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_embeddings = []

for img_file in image_files:
    img_path = os.path.join(image_folder_path, img_file)

    img_embedding = generate_embedding(img_path)
    image_embeddings.append(img_embedding)

image_embeddings_np = np.array(image_embeddings)

np.savetxt('image_embeddings.tsv', image_embeddings_np, delimiter='\t')
