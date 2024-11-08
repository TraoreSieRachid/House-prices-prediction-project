import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Application de gestion des prix immobiliers", layout="wide")

# Chargement des données
@st.cache_data  # Met en cache les données pour éviter un rechargement à chaque changement de page
def load_data():
    # Remplacez 'house_prices.csv' par le chemin de votre fichier de données
    data = pd.read_csv("data/train_cleaned.csv")
    return data

data = load_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Analyse", "Prédiction", "Performance"])

# Section Accueil
if page == "Accueil":
    st.title("Bienvenue dans l'application de gestion des prix immobiliers")
    st.write("Cette application permet de prédire les prix des maisons, d'analyser les données, et d'évaluer les performances des modèles de prédiction.")

# Section Analyse
elif page == "Analyse":
    st.title("Analyse des Données")
    st.write("Exploration des données des prix immobiliers.")

    # Affichage des données
    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(data)  # Affiche le dataframe

    # Statistiques descriptives
    st.write("### Statistiques descriptives")
    st.write(data.describe())  # Affiche un résumé statistique des données

# Section Prédiction
elif page == "Prédiction":
    st.title("Prédiction des Prix")
    st.write("Utilisez cette section pour prédire les prix des maisons en fonction des caractéristiques.")

# Section Performance
elif page == "Performance":
    st.title("Évaluation des Performances du Modèle")
    st.write("Examinez les performances des modèles utilisés pour la prédiction des prix.")
