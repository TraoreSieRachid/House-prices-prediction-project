import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Configuration de la page Streamlit
st.set_page_config(page_title="Prédiction des prix immobiliers", layout="wide")

# Fonction pour charger le modèle Ridge (mise en cache)
@st.cache_resource
def load_ridge_model():
    return joblib.load('code/ridge_model.pkl')  # Remplacez par le chemin réel de votre modèle

ridge_model = load_ridge_model()

# Fonction pour charger les données (mise en cache)
@st.cache_data
def load_data():
    return pd.read_csv("data/train_df.csv")  # Remplacez par le chemin réel de vos données

data = load_data()

# Initialisation de l'état de la page (si ce n'est pas déjà fait)
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# Titre de l'application
st.title("Application de prédiction des prix immobiliers")

# Fonction pour changer la page active dans st.session_state
def set_page(page_name):
    st.session_state.page = page_name

# Barre de navigation horizontale avec des boutons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Accueil"):
        set_page("Accueil")
with col2:
    if st.button("📊 Analyse"):
        set_page("Analyse")
with col3:
    if st.button("🔍 Prédiction"):
        set_page("Prédiction")
with col4:
    if st.button("📈 Performance"):
        set_page("Performance")

# Section Accueil
if st.session_state.page == "Accueil":
    st.subheader("🏠 Bienvenue dans l'application de prédiction des prix immobiliers")
    st.write("""
        Cette application vous permet de :
        - Prédire les prix des maisons
        - Analyser les données des prix immobiliers
        - Évaluer les performances des modèles de prédiction
    """)

# Section Analyse des données
elif st.session_state.page == "Analyse":
    st.subheader("📊 Analyse des Données")
    st.write("Exploration des données des prix immobiliers.")

    # Affichage des données brutes si l'option est activée
    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(data)

    # Statistiques descriptives
    st.write("### Statistiques descriptives")
    st.write(data.describe())

    # Sélection des variables pour la visualisation
    st.write("### Visualisation de deux variables")
    variable_x = st.selectbox("Sélectionnez la première variable (axe X)", data.columns)
    variable_y = st.selectbox("Sélectionnez la deuxième variable (axe Y)", data.columns)

    # Visualisation des relations entre les variables
    fig, ax = plt.subplots(figsize=(10, 8))
    if data[variable_x].dtype in ['int64', 'float64'] and data[variable_y].dtype in ['int64', 'float64']:
        sns.scatterplot(data=data, x=variable_x, y=variable_y, ax=ax, color="teal", s=100, edgecolor='black')
        ax.set_title(f"Nuage de points entre {variable_x} et {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    elif data[variable_x].dtype == 'object' and data[variable_y].dtype == 'object':
        grouped_data = data.groupby([variable_x, variable_y]).size().unstack()
        grouped_data.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm')
        ax.set_title(f"Graphique en barres empilées de {variable_x} par {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel("Effectifs", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(title=variable_y, fontsize=12)
    else:
        sns.boxplot(data=data, x=variable_x, y=variable_y, ax=ax, palette="Set2")
        ax.set_title(f"Graphique de boîte de {variable_y} par {variable_x}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    st.pyplot(fig)

    # Matrice de corrélation
    st.write("### Matrice de Corrélation")
    correlation_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".1f", ax=ax_corr, cbar=True, annot_kws={'size': 10}, mask=mask)
    ax_corr.set_title("Matrice de Corrélation")
    st.pyplot(fig_corr)

# Section Prédiction des prix
elif st.session_state.page == "Prédiction":
    st.subheader("🔍 Prédiction des Prix")
    st.write("Entrez les caractéristiques de la maison pour prédire son prix.")
    
    # Formulaire pour saisir les caractéristiques
    form_data = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            form_data[col] = st.selectbox(f"{col}", data[col].unique())
        elif data[col].dtype in ['int64', 'float64']:
            min_val = data[col].min()
            max_val = data[col].max()
            form_data[col] = st.number_input(f"{col}", min_value=float(min_val), max_value=float(max_val), value=float(min_val))
    
    # Prédiction du prix lorsque le bouton est cliqué
    if st.button("Prédire le Prix"):
        input_data = pd.DataFrame([form_data])
        st.write("Données d'entrée pour la prédiction :", input_data)
        predicted_price = ridge_model.predict(input_data)
        st.write(f"Le prix prédit par le modèle Ridge est : {predicted_price[0]:,.2f}")

# Section Performance des modèles
elif st.session_state.page == "Performance":
    st.subheader("📈 Évaluation des Performances du Modèle")
    st.write("""
        Cette section permet d'examiner les performances du modèle de prédiction des prix immobiliers.
    """)
    # Vous pouvez ajouter ici des métriques de performance comme MAE, MSE, RMSE (à calculer sur un jeu de test)
    # Exemple :
    # y_pred = ridge_model.predict(X_test)
    # st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    # st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    # st.write("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
