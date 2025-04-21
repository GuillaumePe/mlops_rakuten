import time
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
from bs4 import BeautifulSoup
import re
import os
from PIL import Image

def clean_description(text: str) -> str:
    
    # Nettoyage HTML avec BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator="\n", strip=True)
    
    # Transformer les retours à la ligne en phrases continues
    cleaned_text = cleaned_text.replace("\n", ". ")
    # Remplacer les entités HTML comme &#39; ou autres
    cleaned_text = re.sub(r'&#\d+;', '', cleaned_text)  

     # Remplacer les caractères non désirés comme '¿¿' par des espaces
    cleaned_text = re.sub(r'¿{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('¿', '')
    # Supprimer tous les autres caractères spéciaux non désirés (tels que ®, ©, ™, etc.)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Enlever les caractères non-ASCII
    return cleaned_text


def log_progress(current, total, start_time):

    elapsed = time.time() - start_time
    speed = current / elapsed if elapsed > 0 else 0
    remaining = (total - current) / speed if speed > 0 else float('inf')
    print(f"[{current}/{total}] - {speed:.2f} batchs/sec - ETA: {remaining:.1f} sec", end='\r')



def extract_text_features_in_batches(texts,tokenizer = None, model= None, batch_size=32, max_length=128):
    
    if tokenizer == None:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    if model == None:
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    model.eval()  # Mode évaluation

    all_embeddings = []

    with torch.no_grad(): 
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            # Tokenisation par batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            # Passage dans bert modele
            outputs = model(**inputs)

            # Récupération des embeddings du token [CLS]
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  

            all_embeddings.append(batch_embeddings)
            
    # Fusionner tous les batches en une seule matrice numpy
    embeddings_array = np.vstack(all_embeddings)

    return embeddings_array


def extract_images_features(input_dir,image_paths,preprocess,model= None):

    all_embeddings = []

    with torch.no_grad():
        for img_id in image_paths:
            img_path = os.path.join(input_dir,img_id)
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(image).unsqueeze(0)
                batch_embeddings = model(input_tensor).squeeze().numpy()
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Erreur sur image {img_id}: {e}")
                all_embeddings.append([float('nan')] * 512)  # En cas d'erreur, on place un NaN vector de taille 512 (ResNet18)
    return all_embeddings