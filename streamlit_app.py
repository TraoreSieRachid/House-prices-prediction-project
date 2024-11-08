import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Application de gestion des prix immobiliers", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("data/train_cleaned.csv")  # Remplacez par le chemin réel de votre fichier
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
        st.dataframe(data)

    # Statistiques descriptives
    st.write("### Statistiques descriptives")
    st.write(data.describe())

# Section Prédiction
elif page == "Prédiction":
    st.title("Prédiction des Prix")
    st.write("Utilisez ce formulaire pour entrer les valeurs des caractéristiques et prédire le prix d'une maison.")

    # Création d'un formulaire de saisie pour chaque variable
    form_data = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            # Champ de saisie de texte pour les variables catégorielles
            form_data[col] = st.selectbox(f"{col}", data[col].unique())
        elif data[col].dtype in ['int64', 'float64']:
            # Champ de saisie numérique pour les variables numériques
            min_val = data[col].min()
            max_val = data[col].max()
            form_data[col] = st.number_input(f"{col}", min_value=float(min_val), max_value=float(max_val), value=float(min_val))
    
    # Bouton pour prédire
    if st.button("Prédire le Prix"):
        st.write("Lancer la prédiction avec les valeurs suivantes :")
        st.write(form_data)
        # Code pour la prédiction (exemple)
        # predicted_price = model.predict(pd.DataFrame([form_data]))
        # st.write(f"Le prix prédit est : {predicted_price}")

# Section Performance
elif page == "Performance":
    st.title("Évaluation des Performances du Modèle")
    st.write("Examinez les performances des modèles utilisés pour la prédiction des prix.")
