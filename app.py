import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
import torch.nn.functional as F
import numpy as np
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Duygu analizi modeli
emotion_model_name = "maymuni/bert-base-turkish-cased-emotion-analysis"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(device)

# Nefret söylemi modeli doğrudan Hugging Face'ten yüklenecek
hate_model_name = "Urbanhobbit/turkish-offensive-model"
hate_tokenizer = AutoTokenizer.from_pretrained(hate_model_name)
hate_model = AutoModelForSequenceClassification.from_pretrained(hate_model_name).to(device)

# Etiketler
id2label = {
    0: "anger",
    1: "surprise",
    2: "joy",
    3: "sadness",
    4: "fear",
    5: "disgust"
}

num_labels = emotion_model.config.num_labels

def predict_emotion(sentences, use_none=False, threshold=0.3):
    inputs = emotion_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs_all = F.softmax(logits, dim=1).cpu().tolist()

    results = []
    for sent, probs in zip(sentences, probs_all):
        result = {"sentence": sent}
        max_prob = max(probs)
        if use_none and max_prob < threshold:
            result["none"] = 1.0
        else:
            for i in range(min(len(probs), num_labels)):
                label = id2label.get(i, f"label_{i}")
                result[label] = round(probs[i], 4)
        results.append(result)
    return pd.DataFrame(results)

def predict_offense(sentences):
    inputs = hate_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = hate_model(**inputs).logits
        probs_all = F.softmax(logits, dim=1).cpu().tolist()

    results = []
    for sent, probs in zip(sentences, probs_all):
        results.append({"sentence": sent,
                        "OFFENSIVE": round(float(probs[1]), 4),
                        "NOT OFFENSIVE": round(float(probs[0]), 4)})
    return pd.DataFrame(results)

def extract_sentences_from_docx(docx_file):
    doc = Document(docx_file)
    text = " ".join([para.text for para in doc.paragraphs])
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences

st.title("Türkçe Duygu ve Nefret Söylemi Sınıflandırması")

uploaded_file = st.file_uploader("Word dosyası yükleyin (.docx)", type="docx")
input_text = st.text_area("Ya da metin girin", "")
use_none = st.checkbox("Düşük olasılıklar için 'none' etiketi uygula", value=False)
threshold = st.slider("'None' için eşik (threshold)", 0.0, 1.0, 0.3, 0.05)

sentences = []

if uploaded_file:
    sentences = extract_sentences_from_docx(uploaded_file)
elif input_text:
    sentences = [s.strip() for s in input_text.split(".") if s.strip()]

if sentences:
    df_emotion = predict_emotion(sentences, use_none=use_none, threshold=threshold)
    st.subheader("Duygu Analizi Sonuçları")
    st.dataframe(df_emotion)

    fig, ax = plt.subplots()
    emotion_cols = [col for col in df_emotion.columns if col in id2label.values() or col == "none"]
    df_emotion[emotion_cols].mean().plot(kind="bar", ax=ax)
    plt.title("Ortalama Duygu Olasılıkları")
    st.pyplot(fig)

    csv_emotion = df_emotion.to_csv(index=False).encode("utf-8")
    st.download_button("Duygu CSV indir", data=csv_emotion, file_name="duygu_siniflandirma.csv", mime="text/csv")

    df_offense = predict_offense(sentences)
    st.subheader("Nefret Söylemi Sınıflandırması Sonuçları")
    st.dataframe(df_offense)

    csv_offense = df_offense.to_csv(index=False).encode("utf-8")
    st.download_button("Nefret Söylemi CSV indir", data=csv_offense, file_name="nefret_siniflandirma.csv", mime="text/csv")
