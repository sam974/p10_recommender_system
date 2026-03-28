
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

app = Flask(__name__)

# --- Chemins des fichiers ---
DATA_FILE = "data/articles_embeddings_v2.pickle"
ARTICLES_FILE = "data/articles_metadata.csv"
CLICKS_FILE = "data/clicks_sample.csv"

# --- Chargement et Précalcul ---
try:
    print("Chargement des données...")
    with open(DATA_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    df_articles = pd.read_csv(ARTICLES_FILE)
    df_clicks = pd.read_csv(CLICKS_FILE)

    print("Précalcul de la matrice de similarité utilisateurs (Collaborative Filtering)...")
    # Matrice binaire Utilisateurs-Articles
    user_item_matrix = df_clicks.pivot_table(index='user_id', columns='click_article_id', values='session_id', aggfunc='count').fillna(0)
    user_item_matrix = (user_item_matrix > 0).astype(int)
    # Similarité Cosinus entre utilisateurs
    user_sim_matrix = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    print("Initialisation terminée.")

except Exception as e:
    print(f"Erreur Initialisation: {e}")
    embeddings, df_articles, df_clicks, user_sim_df = None, None, None, None

# --- Logique de Recommandation ---

def get_collab_recos(user_id, n_recos):
    if user_sim_df is None or user_id not in user_sim_df.index:
        return []
    # Top 5 voisins
    neighbors = user_sim_df[user_id].sort_values(ascending=False).iloc[1:6].index
    target_articles = set(df_clicks[df_clicks['user_id'] == user_id]['click_article_id'])
    
    recommendations = []
    for neighbor in neighbors:
        neighbor_articles = set(df_clicks[df_clicks['user_id'] == neighbor]['click_article_id'])
        new_ones = neighbor_articles - target_articles
        recommendations.extend(list(new_ones))
        if len(set(recommendations)) >= n_recos: break
    return list(set(recommendations))[:n_recos]

def get_content_recos(user_id, n_recos):
    user_clicks = df_clicks[df_clicks['user_id'] == user_id]['click_article_id'].values
    if len(user_clicks) == 0: return []
    profile = np.mean(embeddings[user_clicks], axis=0).reshape(1, -1)
    scores = cosine_similarity(profile, embeddings)[0]
    scores[user_clicks] = -1
    return np.argsort(scores)[::-1][:n_recos].tolist()

# --- API Endpoints ---

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    method = request.args.get('method', default='content', type=str) # 'content' ou 'collab'
    n = request.args.get('n', default=5, type=int)

    if user_id is None: return jsonify({"error": "user_id missing"}), 400

    # Choix de la méthode ou fallback Popularité
    if user_id not in df_clicks['user_id'].unique():
        reco_ids = df_clicks['click_article_id'].value_counts().head(n).index.tolist()
    elif method == 'collab':
        reco_ids = get_collab_recos(user_id, n)
    else:
        reco_ids = get_content_recos(user_id, n)

    recos_details = df_articles[df_articles['article_id'].isin(reco_ids)].to_dict(orient='records')
    return jsonify({"user_id": user_id, "method": method, "recommendations": recos_details})

if __name__ == '__main__':
    try:
        from flask_ngrok import run_with_ngrok
        run_with_ngrok(app)
        app.run()
    except:
        app.run(host='0.0.0.0', port=5000)
