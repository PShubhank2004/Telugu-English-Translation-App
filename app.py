
# -*- coding: utf-8 -*-
import traceback
import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Telugu ↔ English Translator",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
DEVICE = "cpu"  # Force CPU for Streamlit free tier
BATCH_SIZE = 4

# --- Model Loading ---
@st.cache_resource
def initialize_models():
    """Load Helsinki-NLP MarianMT models (CPU only)."""
    # English → Telugu
    en_to_te_model_name = "Helsinki-NLP/opus-mt-en-te"
    en_to_te_tokenizer = MarianTokenizer.from_pretrained(en_to_te_model_name)
    en_to_te_model = MarianMTModel.from_pretrained(en_to_te_model_name).to(DEVICE)
    en_to_te_model.eval()

    # Telugu → English
    te_to_en_model_name = "Helsinki-NLP/opus-mt-te-en"
    te_to_en_tokenizer = MarianTokenizer.from_pretrained(te_to_en_model_name)
    te_to_en_model = MarianMTModel.from_pretrained(te_to_en_model_name).to(DEVICE)
    te_to_en_model.eval()

    return en_to_te_model, en_to_te_tokenizer, te_to_en_model, te_to_en_tokenizer

# --- Translation Function ---
def batch_translate_marian(input_sentences, model, tokenizer):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translations += decoded
    return translations

# --- UI Rendering ---
st.title("Telugu ↔ English AI Translator")

try:
    with st.spinner("Loading AI models… This may take a few minutes on first run."):
        en_to_te_model, en_to_te_tokenizer, te_to_en_model, te_to_en_tokenizer = initialize_models()
    st.success("Models loaded successfully!")
    st.divider()

    col1, col2 = st.columns(2)

    # English → Telugu
    with col1:
        st.markdown("#### English ➡️ Telugu")
        en_input = st.text_area("Enter English text here:", key="en_input", height=150)
        if st.button("Translate to Telugu", use_container_width=True):
            if en_input.strip():
                with st.spinner("Translating..."):
                    result = batch_translate_marian([en_input], en_to_te_model, en_to_te_tokenizer)
                    st.info("Telugu Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter some text to translate.")

    # Telugu → English
    with col2:
        st.markdown("#### Telugu ➡️ English")
        te_input = st.text_area("Enter Telugu text here:", key="te_input", height=150)
        if st.button("Translate to English", use_container_width=True):
            if te_input.strip():
                with st.spinner("Translating..."):
                    result = batch_translate_marian([te_input], te_to_en_model, te_to_en_tokenizer)
                    st.info("English Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter some text to translate.")

except Exception as e:
    st.error("🚨 Unexpected error occurred while running the app.")
    st.markdown("---")
    st.markdown("**Error Details:**")
    st.text(traceback.format_exc())
