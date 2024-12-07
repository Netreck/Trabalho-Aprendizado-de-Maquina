import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np

# Carregar o modelo salvo
classifier = joblib.load("modelo_treinado.joblib")  


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Carregar o modelo Sentence Transformer
model = SentenceTransformer('all-mpnet-base-v2')

# Função para limpar o texto
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Configurar o layout da página
st.set_page_config(page_title="Classificação de Sentimento", page_icon="🎥", layout="centered")

# Título da aplicação
st.markdown(
    """
    <h1 style="text-align: center; color: #FFF;">🎬 Classificação de Sentimento de Filmes</h1>
    <p style="text-align: center; font-size: 20px;">Digite um review abaixo para descobrir se o sentimento é <b style="color:#4CAF50;">positivo</b> ou <b style="color:#FF4B4B;">negativo</b>!</p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .custom-textarea textarea {
        font-size: 18px;  /* Aumente o tamanho da fonte aqui */
        padding: 10px;    /* Adicione um pouco de espaçamento interno */
        line-height: 1.5; /* Ajuste o espaçamento entre linhas */
    }
    .custom-textarea p {
        font-size: 24px;  /* Aumenta o tamanho do texto */
        font-weight: bold;  /* Negrito (opcional) */
        color: #4CAF50;  /* Cor do texto (opcional) */
        margin-bottom: 10px;  /* Espaçamento inferior (opcional) */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Entrada de texto para o review
review = st.text_area("📝 Escreva sua opinião (apenas em inglês):", placeholder="Escreva aqui a sua opinião sobre o filme...", key="custom-textarea")

# Botão para fazer a previsão
if st.button("🔍 Fazer previsão"):
    if review.strip():  # Verifica se o usuário digitou algo
        # Limpar e processar o texto
        new_reviews = clean_text(review)
        new_embeddings = model.encode(new_reviews)
        new_embeddings = new_embeddings.reshape(1, -1)
        
        # Fazer previsão e obter probabilidades
        prediction = classifier.predict(new_embeddings)
        probabilities = classifier.predict_proba(new_embeddings)  # Retorna as probabilidades
        prob_neg = probabilities[0][0] * 100  # Probabilidade de ser negativo
        prob_pos = probabilities[0][1] * 100  # Probabilidade de ser positivo

        # Exibir o resultado com destaque
        if prediction[0] == 1:  # Sentimento Positivo
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2 style="color: #4CAF50; font-size: 48px;">🎉 Sentimento Positivo! 🎉</h2>
                    <p style="font-size: 20px;">Probabilidade de ser Positivo: <b>{prob_pos:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:  # Sentimento Negativo
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2 style="color: #FF4B4B; font-size: 48px;">💔 Sentimento Negativo 💔</h2>
                    <p style="font-size: 20px;">Probabilidade de ser Negativo: <b>{prob_neg:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Aviso caso o campo de texto esteja vazio
        st.warning("Por favor, insira um review antes de fazer a previsão.")