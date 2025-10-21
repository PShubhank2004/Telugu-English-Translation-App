# -*- coding: utf-8 -*-
import traceback
import warnings

import streamlit as st
import torch
# We need AutoModel, AutoTokenizer, and BitsAndBytesConfig for 8-bit loading
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# ------------------------------
# Ignore warnings
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="NLLB Telugu-English Translator (Stable)",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
# CPU device index
DEVICE = -1
# Smaller, stable multilingual model
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def initialize_models():
    """Load the NLLB model in 8-bit quantization for stability on Streamlit Cloud."""

    # 🔑 Define the 8-bit configuration
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)

    # Load Model with 8-bit quantization
    # device_map="auto" ensures the model is placed correctly (likely CPU in this case)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    # We return the tokenizer and model directly
    return tokenizer, model

# --- Translation Function ---
# This function is now generalized for any language pair the model supports
def translate_text(text, src_lang, tgt_lang, model, tokenizer):
    """Translates a single piece of text using the 8-bit NLLB model."""

    # Prepend the target language token, as required by NLLB models
    text = [f"{tgt_lang} {text}"]

    # Tokenize the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    # Move inputs to the model's computed device (CPU or a small GPU)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=4, # Reduced beams slightly for CPU efficiency
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded[0].strip()


# --- Main App ---
st.title("Telugu ↔ English AI Translator (NLLB 8-bit)")

# Load models
try:
    with st.spinner(f"Loading {NLLB_MODEL_NAME} (8-bit)... This may take a few minutes on first run."):
        tokenizer, model = initialize_models()
    st.success("✅ Models loaded successfully! Ready to translate.")
except Exception:
    st.error("🚨 **Failed to load AI models.** This is likely an Out-of-Memory (OOM) error.")
    st.markdown("If the error persists, the model is too large for the Streamlit free tier.")
    st.text(traceback.format_exc())
    st.stop()

# --- UI Layout ---
st.divider()
col1, col2 = st.columns(2)

# English -> Telugu
with col1:
    st.markdown("#### English (eng_Latn) ➡️ Telugu (tel_Telu)")
    en_input = st.text_area("Enter English text:", key="en_input", height=150)
    if st.button("Translate to Telugu", use_container_width=True):
        if en_input.strip():
            with st.spinner("Translating..."):
                try:
                    # NLLB Code: 'eng_Latn' to 'tel_Telu'
                    result = translate_text(en_input, "eng_Latn", "tel_Telu", model, tokenizer)
                    st.info("Telugu Translation:")
                    st.markdown(f"> {result}")
                except Exception:
                    st.error("Translation failed. Check logs.")
                    st.text(traceback.format_exc())
        else:
            st.warning("Please enter some text to translate.")

# Telugu -> English
with col2:
    st.markdown("#### Telugu (tel_Telu) ➡️ English (eng_Latn)")
    te_input = st.text_area("Enter Telugu text:", key="te_input", height=150)
    if st.button("Translate to English", use_container_width=True):
        if te_input.strip():
            with st.spinner("Translating..."):
                try:
                    # NLLB Code: 'tel_Telu' to 'eng_Latn'
                    result = translate_text(te_input, "tel_Telu", "eng_Latn", model, tokenizer)
                    st.info("English Translation:")
                    st.markdown(f"> {result}")
                except Exception:
                    st.error("Translation failed. Check logs.")
                    st.text(traceback.format_exc())
        else:
            st.warning("Please enter some text to translate.")