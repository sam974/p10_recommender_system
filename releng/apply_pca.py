
import pickle
import numpy as np
from sklearn.decomposition import PCA
import os

# --- Configuration des chemins ---
# Les chemins sont relatifs au dossier 'flask-api'
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'flask-api')
DATA_PATH = os.path.join(BASE_PATH, 'data')

# Fichiers d'entrée et de sortie
EMBEDDINGS_IN_FILE = os.path.join(DATA_PATH, 'articles_embeddings_v2.pickle')
EMBEDDINGS_OUT_FILE = os.path.join(DATA_PATH, 'articles_embeddings_reduced.pickle')
PCA_MODEL_OUT_FILE = os.path.join(DATA_PATH, 'pca_model.pickle')

# --- Paramètres PCA ---
# On vise à conserver 95% de la variance expliquée
N_COMPONENTS = 0.95 

def apply_pca():
    """
    Charge les embeddings, applique une PCA pour réduire leur dimensionnalité
    tout en conservant 95% de la variance, puis sauvegarde le modèle PCA
    et les embeddings réduits.
    """
    print("Début du processus de réduction de dimensionnalité (PCA)...")

    # 1. Charger les embeddings
    try:
        with open(EMBEDDINGS_IN_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings chargés avec succès. Shape: {embeddings.shape}")
    except FileNotFoundError:
        print(f"Erreur: Le fichier d'embeddings '{EMBEDDINGS_IN_FILE}' n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors du chargement des embeddings: {e}")
        return

    # 2. Appliquer la PCA
    print(f"Application de la PCA pour conserver {N_COMPONENTS*100}% de la variance...")
    pca = PCA(n_components=N_COMPONENTS)
    
    # Assurer que les données sont dans un format compatible (ex: np.array)
    if isinstance(embeddings, dict):
        # Si c'est un dictionnaire, on prend les valeurs
        ids = list(embeddings.keys())
        embedding_matrix = np.array(list(embeddings.values()))
    elif isinstance(embeddings, np.ndarray):
        embedding_matrix = embeddings
    else:
        print("Format d'embeddings non supporté. Attendu: dict ou numpy array.")
        return

    reduced_embeddings_matrix = pca.fit_transform(embedding_matrix)
    
    print(f"Réduction de dimensionnalité terminée.")
    print(f"Nombre de composantes sélectionnées: {pca.n_components_}")
    print(f"Nouvelle shape des embeddings: {reduced_embeddings_matrix.shape}")
    print(f"Variance totale expliquée: {np.sum(pca.explained_variance_ratio_):.4f}")

    # (Optionnel) Recréer le dictionnaire si le format original était un dict
    if isinstance(embeddings, dict):
        reduced_embeddings = {id: vec for id, vec in zip(ids, reduced_embeddings_matrix)}
    else:
        reduced_embeddings = reduced_embeddings_matrix

    # 3. Sauvegarder le modèle PCA
    try:
        with open(PCA_MODEL_OUT_FILE, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Modèle PCA sauvegardé dans '{PCA_MODEL_OUT_FILE}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle PCA: {e}")

    # 4. Sauvegarder les embeddings réduits
    try:
        with open(EMBEDDINGS_OUT_FILE, 'wb') as f:
            pickle.dump(reduced_embeddings, f)
        print(f"Embeddings réduits sauvegardés dans '{EMBEDDINGS_OUT_FILE}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des embeddings réduits: {e}")

if __name__ == '__main__':
    apply_pca()
