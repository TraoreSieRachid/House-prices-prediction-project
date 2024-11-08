import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Configuration de la page
st.set_page_config(page_title="Application de gestion des prix immobiliers", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("data/train_cleaned.csv")  # Remplacez par le chemin réel de votre fichier
    return data

data = load_data()

# Barre de navigation horizontale
st.title("Application de gestion des prix immobiliers")

# Créer des colonnes pour placer les boutons de navigation horizontalement
col1, col2, col3, col4 = st.columns(4)

with col1:
    accueil = st.button("🏠 Accueil")
with col2:
    analyse = st.button("📊 Analyse")
with col3:
    prediction = st.button("🔍 Prédiction")
with col4:
    performance = st.button("📈 Performance")

# Déterminer la page active en fonction du bouton cliqué
page = "Accueil"  # Page par défaut
if accueil:
    page = "Accueil"
elif analyse:
    page = "Analyse"
elif prediction:
    page = "Prédiction"
elif performance:
    page = "Performance"

# Section Accueil
if page == "Accueil":
    st.subheader("🏠 Bienvenue dans l'application de gestion des prix immobiliers")
    st.write("Cette application permet de prédire les prix des maisons, d'analyser les données, et d'évaluer les performances des modèles de prédiction.")

# Section Analyse
elif page == "Analyse":
    st.subheader("📊 Analyse des Données")
    st.write("Exploration des données des prix immobiliers.")

    # Rapport interactif avec pandas-profiling
    if st.checkbox("Afficher le rapport de profilage des données"):
        st.subheader("Rapport de Profiling des Données")
        profile = ProfileReport(data, minimal=True)
        st_profile_report(profile)

# Section Prédiction
elif page == "Prédiction":
    st.subheader("🔍 Prédiction des Prix")
    st.write("Utilisez ce formulaire pour entrer les valeurs des caractéristiques et prédire le prix d'une maison.")
    
    # Formulaire de saisie pour chaque variable
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
    st.subheader("📈 Évaluation des Performances du Modèle")
    st.write("Examinez les performances des modèles utilisés pour la prédiction des prix.")
    # Ajoutez ici le code pour afficher les métriques de performance du modèle

# ---------------------------------------------------
#import joblib

# Charger le modèle
#@st.cache_data
#def load_model():
 #   model = joblib.load("path_to_your_model/model.pkl")  # Remplacez par le chemin réel de votre modèle
  #  return model

#model = load_model()

# Prédire
#if st.button("Prédire le Prix"):
 #   # Convertir le dictionnaire form_data en DataFrame pour l'utiliser avec le modèle
  #  form_df = pd.DataFrame([form_data])
   # predicted_price = model.predict(form_df)[0]
   # st.write(f"Le prix prédit est : {predicted_price}")
