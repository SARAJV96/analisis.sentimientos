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

nltk.download('stopwords', quiet=True)

# Configuración esencial para Render
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")

# Cargar datos CORREGIDO (sin delimiter='\n')
try:
    with open("data/opiniones_clientes.csv", "r", encoding="utf-8") as f:
        lines = [line.strip().strip('"') for line in f.readlines()[1:]]  # Saltar header y limpiar líneas
        
    df = pd.DataFrame(lines, columns=['Opinion'])
    df = df[df['Opinion'].str.len() > 0]  # Filtrar líneas vacías
    
except Exception as e:
    st.error(f"Error crítico: {str(e)}")
    st.stop()

# Cargar modelo de análisis de sentimientos (optimizado)
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device_map="auto"
    )

clasificador = cargar_modelo()

# Lógica de clasificación mejorada
def interpretar(texto):
    try:
        resultado = clasificador(texto[:512], truncation=True)[0]
        estrellas = int(resultado['label'][0])
        return "⭐ Positivo" if estrellas >= 4 else "🔄 Neutro" if estrellas == 3 else "⚠️ Negativo"
    except:
        return "❓ Indeterminado"

# Procesamiento optimizado
with st.spinner("Analizando opiniones..."):
    df['Sentimiento'] = df['Opinion'].progress_apply(interpretar)  # Usar progress_apply si tienes tqdm

# Interfaz de usuario mejorada
st.title("📈 Panel de Análisis de Sentimientos")
st.dataframe(
    df[['Opinion', 'Sentimiento']],
    height=600,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Opinion": "Comentario del cliente",
        "Sentimiento": st.column_config.SelectboxColumn(
            "Clasificación",
            options=["⭐ Positivo", "🔄 Neutro", "⚠️ Negativo", "❓ Indeterminado"]
        )
    }
)

# Configuración final para Render
if __name__ == "__main__":
    os.system(f"streamlit run {__file__} --server.port {PORT} --server.address 0.0.0.0")
