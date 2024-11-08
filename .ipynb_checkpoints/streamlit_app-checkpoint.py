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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Analyse", "Prédiction", "Performance"])
