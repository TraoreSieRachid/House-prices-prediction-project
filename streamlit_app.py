import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Application de gestion des prix immobiliers", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("data/train_cleaned.csv")  # Remplacez par le chemin réel de votre fichier
    return data

data = load_data()

# Préparation des modèles de régression
@st.cache_resource
def train_models(data):
    # Préparation des données pour l'entraînement
    X = data.drop(columns=['SalePrice'])  # Remplacer 'SalePrice' par la colonne cible dans votre dataset
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Régression Linéaire": LinearRegression(),
        "Régression Ridge": Ridge(),
        "Régression Lasso": Lasso()
    }

    trained_models = {}
    performances = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        trained_models[name] = model
        performances[name] = mae  # Enregistrer le MAE pour chaque modèle

    return trained_models, performances

trained_models, performances = train_models(data)

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

    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(data)

    st.write("### Statistiques descriptives")
    st.write(data.describe())

# Section Prédiction
elif page == "Prédiction":
    st.title("Prédiction des Prix")
    st.write("Sélectionnez un modèle et entrez les valeurs des caractéristiques pour prédire le prix d'une maison.")

    # Sélection du modèle
    model_choice = st.selectbox("Choisissez un modèle de régression", ["Régression Linéaire", "Régression Ridge", "Régression Lasso"])

    # Création d'un formulaire de saisie pour chaque variable
    form_data = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            form_data[col] = st.selectbox(f"{col}", data[col].unique())
        elif data[col].dtype in ['int64', 'float64']:
            min_val = data[col].min()
            max_val = data[col].max()
            form_data[col] = st.number_input(f"{col}", min_value=float(min_val), max_value=float(max_val), value=float(min_val))
    
    # Bouton pour effectuer la prédiction
    if st.button("Prédire le Prix"):
        st.write("Lancer la prédiction avec les valeurs suivantes :")
        st.write(form_data)
        
        # Convertir les valeurs de `form_data` en DataFrame pour la prédiction
        input_df = pd.DataFrame([form_data])

        # Prédiction
        model = trained_models[model_choice]
        predicted_price = model.predict(input_df)
        st.write(f"Le prix prédit est : {predicted_price[0]:,.2f} €")

# Section Performance
elif page == "Performance":
    st.title("Évaluation des Performances du Modèle")
    st.write("Visualisez les performances des modèles de régression avec le MAE (Mean Absolute Error) comme métrique.")

    # Afficher l'histogramme des performances
    fig, ax = plt.subplots()
    ax.bar(performances.keys(), performances.values(), color=['blue', 'green', 'orange'])
    ax.set_xlabel("Modèles")
    ax.set_ylabel("MAE (Mean Absolute Error)")
    ax.set_title("Performance des modèles de régression")
    
    # Afficher la figure avec Streamlit
    st.pyplot(fig)
