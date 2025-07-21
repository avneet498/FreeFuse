
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import emoji
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Load tokenizer, model, and label encoder
@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    roberta = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = joblib.load("lightgbm_best_model.pkl")

    le = LabelEncoder()
    le.classes_ = np.array(['Negative', 'Neutral', 'Positive'])  # same as training
    return tokenizer, roberta.eval(), model, le

tokenizer, roberta, lgb_model, label_encoder = load_assets()

# Emoji options
emoji_map = {
    1: "😀", 2: "😃", 3: "😄", 4: "😁", 5: "😆", 6: "😅", 7: "😂", 8: "🤣", 9: "❤️", 10: "😍",
    11: "🥰", 12: "😐", 13: "😕", 14: "😟", 15: "🤔", 16: "🙁", 17: "☹️", 18: "😞",
    19: "😢", 20: "😭", 21: "😡", 22: "😠", 23: "🤯", 24: "💔", 25: "🥺", 26: "😨", 27: "😳"
}

def get_roberta_embedding(text):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output = roberta(**encoded)
        return output.last_hidden_state[0, 0, :].numpy()

def predict_sentiment(comment, emoji_list, engagement):
    input_text = f"{comment} {emoji.demojize(' '.join(emoji_list), delimiters=(' ', ' '))}"
    embedding = get_roberta_embedding(input_text)
    engagement_scaled = engagement / 100.0
    features = np.hstack((embedding, [engagement_scaled])).reshape(1, -1)
    prediction = lgb_model.predict(features)[0]
    return label_encoder.inverse_transform([prediction])[0]

# Streamlit UI
st.title("🧠 Sentiment & Emotion Classifier")
st.markdown("Enter a viewer's comment, emoji reactions, and engagement score to get a sentiment prediction.")

comment = st.text_area("💬 Enter viewer comment")

selected_emojis = st.multiselect(
    "🎭 Select emojis used in reaction:",
    options=list(emoji_map.keys()),
    format_func=lambda x: f"{x}: {emoji_map[x]}"
)

engagement = st.slider("📊 Viewer Engagement (0–100%)", 0, 100, 50)

if st.button("Predict Sentiment"):
    if comment.strip() and selected_emojis:
        emojis = [emoji_map[num] for num in selected_emojis]
        sentiment = predict_sentiment(comment, emojis, engagement)
        st.success(f"🧾 Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a comment and select at least one emoji.")
