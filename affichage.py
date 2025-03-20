import os
import matplotlib.pyplot as plt
from PIL import Image

# Chemin du dossier contenant les images générées
samples_dir = 'samples'

# Lister tous les fichiers d'image valides dans le dossier
image_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Vérifier si des images sont présentes
if not image_files:
    raise FileNotFoundError("Aucune image valide n'a été trouvée dans le dossier spécifié.")

# Pagination : définir le nombre d'images par page
images_per_page = 100  # Ajustez en fonction des capacités de votre système
num_pages = (len(image_files) + images_per_page - 1) // images_per_page

def display_page(page_index):
    """Affiche une page spécifique d'images."""
    start = page_index * images_per_page
    end = min(start + images_per_page, len(image_files))
    num_images = end - start

    # Définir les dimensions de la grille
    num_cols = 10  # Par exemple, 10 images par ligne
    num_rows = (num_images + num_cols - 1) // num_cols

    # Taille de la figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 2))

    # Aplatir les axes pour faciliter l'itération
    axes = axes.flatten()

    for i, image_file in enumerate(image_files[start:end]):
        try:
            image_path = os.path.join(samples_dir, image_file)
            img = Image.open(image_path)
            axes[i].imshow(img)
            axes[i].axis('off')  # Désactiver les axes
        except Exception as e:
            print(f"Erreur lors de l'ouverture de l'image {image_file}: {e}")
            axes[i].axis('off')  # Toujours désactiver les axes

    # Masquer les sous-graphiques supplémentaires
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('100 images.png')
    plt.show()

# Exemple : afficher la première page
display_page(1)

# Vous pouvez ajouter une interface pour naviguer entre les pages :
# Par exemple, `display_page(1)` pour la page suivante, etc.
