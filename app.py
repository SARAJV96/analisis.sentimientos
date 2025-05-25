import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
import os

# Descargar stopwords de NLTK
nltk.download('stopwords', quiet=True)

# --- Configuración esencial para Render ---
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")

# --- Carga de datos optimizada ---
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(
            "data/opiniones_clientes.csv",
            quotechar='"',
            engine='python',
            on_bad_lines='skip',
            delimiter='\n',
            names=['Opinion']
        )
        return df.dropna().sample(min(100, len(df)))  # Limitar a 100 opiniones para pruebas
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

df = cargar_datos()

# --- Carga eficiente del modelo ---
@st.cache_resource(show_spinner="Cargando modelo de análisis...")
def cargar_modelo():
    modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline(
        "sentiment-analysis", 
        model=modelo,
        tokenizer=modelo,
        device_map="auto"
    )

clasificador = cargar_modelo()

# --- Lógica de procesamiento ---
def interpretar(label):
    estrellas = int(label[0])
    return "⭐ Positivo" if estrellas >= 4 else "🔄 Neutro" if estrellas == 3 else "⚠️ Negativo"

# --- Interfaz de usuario ---
st.title("📊 Análisis de Opiniones en Tiempo Real")
with st.spinner("Analizando sentimientos..."):
    df['Sentimiento'] = df['Opinion'].apply(lambda x: interpretar(clasificador(x[:512])[0]['label']))

# Mostrar resultados
st.data_editor(
    df[['Opinion', 'Sentimiento']],
    use_container_width=True,
    height=600,
    column_config={
        "Opinion": "Comentario",
        "Sentimiento": st.column_config.SelectboxColumn(
            "Clasificación",
            options=["⭐ Positivo", "🔄 Neutro", "⚠️ Negativo"]
        )
    }
)

# --- Configuración final para Render ---
if __name__ == "__main__":
    os.system(f"streamlit run {__file__} --server.port {PORT} --server.address 0.0.0.0")