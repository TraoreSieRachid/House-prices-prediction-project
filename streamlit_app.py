import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
#from sklearn.metrics import mean_absolute_error, mean_squared_error
# Configuration de la page
st.set_page_config(page_title="Application de gestion des prix immobiliers", layout="wide")

@st.cache_resource
def load_ridge_model():
    return joblib.load('code/ridge_model.pkl')  # Chemin vers votre fichier modèle Ridge
ridge_model= load_ridge_model()

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("data/train_df.csv")  # Remplacez par le chemin réel de votre fichier
    return data

data = load_data()

# Initialisation de l'état de la page
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# Barre de navigation horizontale
st.title("Application de gestion des prix immobiliers")

# Créer des colonnes pour placer les boutons de navigation horizontalement
col1, col2, col3, col4 = st.columns(4)

# Fonctions pour changer la page active dans st.session_state
def set_page(page_name):
    st.session_state.page = page_name

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
    st.subheader("🏠 Bienvenue dans l'application de gestion des prix immobiliers")
    st.write("Cette application permet de prédire les prix des maisons, d'analyser les données, et d'évaluer les performances des modèles de prédiction.")

# Section Analyse
elif st.session_state.page == "Analyse":
    st.subheader("📊 Analyse des Données")
    st.write("Exploration des données des prix immobiliers.")

    # Affichage des données brutes
    if st.checkbox("Afficher les données brutes"):
        st.subheader("Données des prix immobiliers")
        st.dataframe(data)

    # Statistiques descriptives
    st.write("### Statistiques descriptives")
    st.write(data.describe())

    # Sélection de deux variables pour la visualisation
    st.write("### Visualisation de deux variables")
    variable_x = st.selectbox("Sélectionnez la première variable (axe X)", data.columns)
    variable_y = st.selectbox("Sélectionnez la deuxième variable (axe Y)", data.columns)

    # Génération du graphique en fonction des types des variables
    fig, ax = plt.subplots()
    if data[variable_x].dtype in ['int64', 'float64'] and data[variable_y].dtype in ['int64', 'float64']:
        # Si les deux variables sont numériques, afficher un nuage de points
        sns.scatterplot(data=data, x=variable_x, y=variable_y, ax=ax)
        ax.set_title(f"Nuage de points entre {variable_x} et {variable_y}")
    elif data[variable_x].dtype == 'object' and data[variable_y].dtype == 'object':
        # Si les deux variables sont catégorielles, afficher un graphique en barres empilées
        grouped_data = data.groupby([variable_x, variable_y]).size().unstack()
        grouped_data.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f"Graphique en barres empilées de {variable_x} par {variable_y}")
        ax.set_xlabel(variable_x)
        ax.set_ylabel("Effectifs")
    else:
        # Si une variable est numérique et l'autre catégorielle, afficher un graphique de boîte
        if data[variable_x].dtype == 'object':
            sns.boxplot(data=data, x=variable_x, y=variable_y, ax=ax)
            ax.set_title(f"Graphique de boîte de {variable_y} par {variable_x}")
        else:
            sns.boxplot(data=data, x=variable_y, y=variable_x, ax=ax)
            ax.set_title(f"Graphique de boîte de {variable_x} par {variable_y}")

    st.pyplot(fig)

# Section Prédiction
elif st.session_state.page == "Prédiction":
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
    
    #Bouton pour prédire
    if st.button("Prédire le Prix"):
        st.write("Lancer la prédiction avec les valeurs suivantes :")
        st.write(form_data)
        input_data = pd.DataFrame([form_data])  # Convertir les entrées en DataFrame
        predicted_price = ridge_model.predict(input_data)
        st.write(f"Le prix prédit par le modèle Ridge est : {predicted_price[0]:,.2f}")


# Section Performance
elif st.session_state.page == "Performance":
    st.subheader("📈 Évaluation des Performances du Modèle")
    st.write("Examinez les performances des modèles utilisés pour la prédiction des prix.")
    #y_pred = ridge_model.predict(X_test)
    #st.subheader("📈 Performance du Modèle Ridge")
    #st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    #st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    st.write("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
