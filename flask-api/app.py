
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

app = Flask(__name__)

# --- Chemins des fichiers de données locaux ---
# Assurez-vous que ces chemins correspondent à l'endroit où vos fichiers sont stockés dans Colab
DATA_FILE = "/content/drive/My Drive/Colab Notebooks/projet10/data/articles_embeddings_v2.pickle"
ARTICLES_FILE = "/content/drive/My Drive/Colab Notebooks/projet10/data/articles_metadata.csv"
CLICKS_FILE = "/content/drive/My Drive/Colab Notebooks/projet10/data/clicks_sample.csv"

# --- Chargement global des données au démarrage de l'application ---
# Ceci est crucial pour éviter de recharger les données à chaque requête API
try:
    print("Chargement des embeddings...")
    with open(DATA_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Embeddings chargés. Forme: {embeddings.shape}")

    print("Chargement des métadonnées des articles...")
    df_articles = pd.read_csv(ARTICLES_FILE)
    print(f"Articles chargés. Forme: {df_articles.shape}")

    print("Chargement des données de clics...")
    df_clicks = pd.read_csv(CLICKS_FILE)
    print(f"Clics chargés. Forme: {df_clicks.shape}")

except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    embeddings = None
    df_articles = None
    df_clicks = None

# --- Fonctions de recommandation (identiques à celles préparées pour le déploiement) ---

def get_user_profile_deployment(user_id, df_clicks_data, embeddings_data):
    user_clicks = df_clicks_data[df_clicks_data['user_id'] == user_id]['click_article_id'].values
    if len(user_clicks) == 0:
        return None
    user_embeddings = embeddings_data[user_clicks]
    user_profile = np.mean(user_embeddings, axis=0)
    return user_profile.reshape(1, -1)

def recommend_for_user_deployment(user_id, df_clicks_data, embeddings_data, n_recos=5):
    user_profile = get_user_profile_deployment(user_id, df_clicks_data, embeddings_data)
    if user_profile is None:
        return []
    scores = cosine_similarity(user_profile, embeddings_data)[0]
    # Exclure les articles déjà cliqués par l'utilisateur pour ne pas les recommander
    clicked_articles = df_clicks_data[df_clicks_data['user_id'] == user_id]['click_article_id'].values
    # Assigner un score très bas aux articles déjà cliqués pour qu'ils ne soient pas recommandés
    # S'assurer que 'clicked_articles' n'est pas vide avant d'indexer
    if clicked_articles.size > 0:
        scores[clicked_articles] = -1
    top_indices = np.argsort(scores)[::-1][:n_recos]
    return top_indices

def get_popular_articles_deployment(df_clicks_data, n_recos=5):
    return df_clicks_data['click_article_id'].value_counts().head(n_recos).index.values

def get_recommendations_for_deployment(user_id, df_clicks_data, embeddings_data, n_recos=5):
    if embeddings_data is None or df_clicks_data is None or df_articles is None:
        print("[API] Erreur: Les données nécessaires ne sont pas chargées.")
        return [] # Gérer l'erreur de chargement de manière robuste

    user_profile = get_user_profile_deployment(user_id, df_clicks_data, embeddings_data)

    if user_profile is not None:
        print(f"[API] Utilisateur {user_id} connu : Recommandation par similarité.")
        return recommend_for_user_deployment(user_id, df_clicks_data, embeddings_data, n_recos)
    else:
        print(f"[API] Nouvel utilisateur {user_id} : Recommandation par popularité (Cold Start).")
        return get_popular_articles_deployment(df_clicks_data, n_recos)

# --- Point d'accès de l'API Flask ---
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    n_recos = request.args.get('num_recommendations', default=5, type=int)

    if user_id is None:
        return jsonify({"error": "Paramètre 'user_id' manquant"}), 400

    recommendation_ids = get_recommendations_for_deployment(user_id, df_clicks, embeddings, n_recos)

    # Récupérer les métadonnées des articles recommandés
    recommended_articles_details = []
    if df_articles is not None:
        for article_id in recommendation_ids:
            article_detail = df_articles[df_articles['article_id'] == article_id].to_dict(orient='records')
            if article_detail:
                recommended_articles_details.append(article_detail[0])

    return jsonify({"user_id": user_id, "recommendations": recommended_articles_details})


# --- Lancement de l'application Flask (pour le développement local) ---
if __name__ == '__main__':
    # Pour tester cette API localement dans Google Colab, vous aurez besoin de `flask-ngrok`.
    # Installez-le avec : !pip install flask-ngrok
    # Ensuite, importez et utilisez run_with_ngrok(app) avant app.run()
    try:
        from flask_ngrok import run_with_ngrok
        run_with_ngrok(app) # Crée un tunnel ngrok pour exposer l'app localement
        app.run()
    except ImportError:
        print("flask-ngrok n'est pas installé. L'application ne peut pas être exposée.")
        print("Pour tester : !pip install flask-ngrok puis réexécutez cette cellule.")
        # Fallback pour le développement local pur (accessible uniquement depuis localhost)
        # app.run(host='0.0.0.0', port=5000, debug=True)
