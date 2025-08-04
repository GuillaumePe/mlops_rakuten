from dagshub import download

if __name__ == "__main__":
    print("Téléchargement des données raw...")
    download("GuillaumePe/mar25_cmlops_rakuten", "data/raw", "data/raw")
    print("Téléchargement des données preprocessed image...")
    download("GuillaumePe/mar25_cmlops_rakuten", "data/preprocessed/chunked_image_files", "data/preprocessed/chunked_image_files")
    print("Téléchargement des données preprocessed textes...")
    download("GuillaumePe/mar25_cmlops_rakuten", "data/preprocessed/chunked_text_files", "data/preprocessed/chunked_text_files")
    print("Données téléchargées avec succès.")