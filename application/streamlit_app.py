
import streamlit as st
import requests
import pickle
import pandas as pd

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        with open('dev/deployment_data/user_profiles.pkl', 'rb') as f:
            user_profiles = pickle.load(f)
        user_ids = list(user_profiles.keys())
        return user_ids
    except FileNotFoundError:
        st.error("Le fichier 'user_profiles.pkl' est introuvable. Assurez-vous que le chemin d'accès est correct.")
        return []
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement des profils utilisateur : {e}")
        return []

# Fonction pour appeler l'Azure Function
def get_recommendations(user_id, recommendation_type):
    # Get the Azure Function URL from Streamlit's secrets
    # Make sure to create a .streamlit/secrets.toml file with the following content:
    # AZURE_FUNCTION_URL = "your_azure_function_url_with_code"
    url = st.secrets["AZURE_FUNCTION_URL"]
    params = {'user_id': user_id, 'recommendation_type': recommendation_type}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API de recommandation : {e}")
        return None

# Interface Streamlit
st.title("Système de Recommandation d'Articles")

# Chargement des IDs utilisateur
user_ids = load_data()

if user_ids:
    # Sélection de l'utilisateur
    selected_user_id = st.selectbox("Sélectionnez un ID utilisateur :", user_ids)

    # Sélection du type de recommandation
    recommendation_type = st.selectbox("Sélectionnez le type de recommandation :", ['item-based-cf', 'content-based'])

    if st.button("Obtenir les Recommandations"):
        if selected_user_id:
            with st.spinner("Appel de l'API de recommandation..."):
                recommendations = get_recommendations(selected_user_id, recommendation_type)
                if recommendations and 'recommendations' in recommendations:
                    st.success("Voici les 5 articles recommandés :")
                    # Affiche les recommandations sous forme de liste
                    for article_id in recommendations['recommendations']:
                        st.write(f"- Article ID : {article_id}")
                else:
                    st.warning("Aucune recommandation n'a été retournée ou le format de la réponse est incorrect.")
else:
    st.warning("Aucun ID utilisateur n'a pu être chargé.")

