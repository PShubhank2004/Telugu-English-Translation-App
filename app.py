# -*- coding: utf-8 -*-
import traceback
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
DEVICE = "cpu"  # CPU-only
BATCH_SIZE = 4

# --- Model Loading ---
@st.cache_resource
def initialize_models():
    try:
        HF_ACCESS_TOKEN = st.secrets["HUGGING_FACE_HUB_TOKEN"]
    except KeyError:
        st.error("HUGGING_FACE_HUB_TOKEN not found in Streamlit secrets!")
        raise ValueError("Add HUGGING_FACE_HUB_TOKEN to .streamlit/secrets.toml")

    # IndicProcessor
    ip = IndicProcessor(inference=True)

    # English → Telugu
    en_to_te_model_name = "ai4bharat/indictrans2-en-indic-1B"
    en_to_te_tokenizer = AutoTokenizer.from_pretrained(
        en_to_te_model_name,
        trust_remote_code=True,
        use_auth_token=HF_ACCESS_TOKEN
    )
    en_to_te_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_to_te_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        use_auth_token=HF_ACCESS_TOKEN
    )
    en_to_te_model.eval()

    # Telugu → English
    te_to_en_model_name = "ai4bharat/indictrans2-indic-en-1B"
    te_to_en_tokenizer = AutoTokenizer.from_pretrained(
        te_to_en_model_name,
        trust_remote_code=True,
        use_auth_token=HF_ACCESS_TOKEN
    )
    te_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        te_to_en_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        use_auth_token=HF_ACCESS_TOKEN
    )
    te_to_en_model.eval()

    return ip, en_to_te_model, en_to_te_tokenizer, te_to_en_model, te_to_en_tokenizer

# --- Translation Function ---
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(input_sentences[i:i + BATCH_SIZE], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        processed_sents = ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
        cleaned_sents = [s.replace("Tel_Telu None ", "").replace("ng @Latn ", "").strip() for s in processed_sents]
        translations += cleaned_sents

    return translations

# --- UI Rendering ---
st.title("Telugu ↔ English AI Translator")

try:
    with st.spinner("Loading AI models… This may take a few minutes on first run (CPU-only)."):
        ip, en_to_te_model, en_to_te_tokenizer, te_to_en_model, te_to_en_tokenizer = initialize_models()
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
                    result = batch_translate([en_input], "eng_Latn", "tel_Telu", en_to_te_model, en_to_te_tokenizer, ip)
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
                    result = batch_translate([te_input], "tel_Telu", "eng_Latn", te_to_en_model, te_to_en_tokenizer, ip)
                    st.info("English Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter some text to translate.")

except Exception:
    st.error("🚨 Unexpected error occurred while running the app.")
    st.markdown("---")
    st.markdown("**Error Details:**")
    st.text(traceback.format_exc())
    st.markdown(
        "**Tip:** Make sure your `HUGGING_FACE_HUB_TOKEN` is correct and you have a stable internet connection."
    )
