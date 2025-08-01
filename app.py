# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants and Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
QUANTIZATION = None

# --- Model Loading ---
@st.cache_resource
def initialize_models():
    """Loads and caches all the required models and tokenizers."""
    ip = IndicProcessor(inference=True)
    en_to_ind_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    en_to_ind_tokenizer = AutoTokenizer.from_pretrained(en_to_ind_ckpt_dir, trust_remote_code=True)
    en_to_ind_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_to_ind_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
    )

    ind_to_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
    ind_to_en_tokenizer = AutoTokenizer.from_pretrained(ind_to_en_ckpt_dir, trust_remote_code=True)
    ind_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        ind_to_en_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
    )

    en_to_ind_model = en_to_ind_model.to(DEVICE)
    ind_to_en_model = ind_to_en_model.to(DEVICE)
    if DEVICE == "cuda":
        en_to_ind_model.half()
        ind_to_en_model.half()
    en_to_ind_model.eval()
    ind_to_en_model.eval()
    return ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer

# --- Core Translation Function ---
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Translates a batch of sentences and cleans the output."""
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(input_sentences[i : i + BATCH_SIZE], src_lang=src_lang, tgt_lang=tgt_lang)
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
st.title("🇮🇳 Telugu ↔ English AI Translator")

# Load models and display status
with st.spinner("Loading AI models... This might take a moment on first startup."):
    ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
st.success("✅ Models loaded and ready!")
st.divider()

# --- Main UI Tabs ---
main_tab1, main_tab2 = st.tabs(["🌐 AI Translator", "🔤 Script Converter (Transliteration)"])

with main_tab1:
    col1, col2 = st.columns(2)

    # English to Telugu
    with col1:
        st.markdown("#### English ➡️ Telugu")
        en_input = st.text_area("Enter English text here:", key="en_input", height=150)
        if st.button("Translate to Telugu", use_container_width=True):
            if en_input.strip():
                with st.spinner("Translating..."):
                    result = batch_translate([en_input], "eng_Latn", "tel_Telu", en_to_ind_model, en_to_ind_tokenizer, ip)
                    st.info("Telugu Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter some text to translate.")

    # Telugu to English
    with col2:
        st.markdown("#### Telugu ➡️ English")
        te_input = st.text_area("Enter Telugu text here:", key="te_input", height=150)
        if st.button("Translate to English", use_container_width=True):
            if te_input.strip():
                with st.spinner("Translating..."):
                    result = batch_translate([te_input], "tel_Telu", "eng_Latn", ind_to_en_model, ind_to_en_tokenizer, ip)
                    st.info("English Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter some text to translate.")

with main_tab2:
    st.header("Convert Between ITRANS and Telugu Script")

    itrans_input = st.text_area("Enter text in English (ITRANS format) below:", help="e.g., 'nā pēru ādars'")
    if itrans_input:
        try:
            with st.container(border=True):
                telugu_script = transliterate(itrans_input, ITRANS, TELUGU)
                st.markdown("**Telugu Script Output:**")
                st.success(telugu_script)

                reverse_itrans = transliterate(telugu_script, TELUGU, ITRANS)
                st.markdown("**Converted back to ITRANS:**")
                st.info(reverse_itrans)
        except Exception as e:
            st.error(f"An error occurred during transliteration: {e}")

st.divider()
st.markdown("<p style='text-align: center;'>Made by Adarsh</p>", unsafe_allow_html=True)
