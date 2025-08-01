# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor
import time

# --- Constants and Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
# Set quantization to None as free tier doesn't support GPU-based quantization well
QUANTIZATION = None 

# --- Model Loading ---
# This function will be cached by Streamlit, so models are loaded only once.
@st.cache_resource
def initialize_models():
    """
    Loads and initializes all the required models and tokenizers.
    """
    # Initialize IndicProcessor
    ip = IndicProcessor(inference=True)

    # Determine quantization configuration
    if QUANTIZATION == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif QUANTIZATION == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
    else:
        qconfig = None

    # --- English to Indic Model ---
    en_to_ind_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    en_to_ind_tokenizer = AutoTokenizer.from_pretrained(en_to_ind_ckpt_dir, trust_remote_code=True)
    en_to_ind_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_to_ind_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig
    )

    # --- Indic to English Model ---
    ind_to_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
    ind_to_en_tokenizer = AutoTokenizer.from_pretrained(ind_to_en_ckpt_dir, trust_remote_code=True)
    ind_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        ind_to_en_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig
    )

    if qconfig is None:
        en_to_ind_model = en_to_ind_model.to(DEVICE)
        ind_to_en_model = ind_to_en_model.to(DEVICE)
        # Using .half() can cause issues on CPU, so we might skip it if no CUDA.
        if DEVICE == "cuda":
            en_to_ind_model.half()
            ind_to_en_model.half()
            
    en_to_ind_model.eval()
    ind_to_en_model.eval()

    return ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer

# --- Core Functions ---
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """
    Translates a batch of sentences.
    """
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(input_sentences[i:i+BATCH_SIZE], src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Move inputs to the correct device
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
        
        # Post-process the translations from the model
        processed_sents = ip.postprocess_batch(decoded_tokens, lang=tgt_lang)

        # Add a manual cleaning step to remove the specific leftover tags you found
        cleaned_sents = [
            s.replace("Tel_Telu None ", "").replace("ng @Latn ", "").strip()
            for s in processed_sents
        ]

        translations += cleaned_sents

    return translations

# --- Streamlit UI ---
st.set_page_config(page_title="Telugu-English Translation", layout="wide")

st.title("Telugu ↔ English Translator & Transliterator 🇮🇳")

# Load models and display status
with st.spinner("Loading translation models... This may take a moment."):
    start_time = time.time()
    ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
    end_time = time.time()
st.success(f"✅ Models loaded successfully in {end_time - start_time:.2f} seconds!")


# --- UI Tabs ---
tab1, tab2 = st.tabs(["🔤 Transliteration", "🌐 Translation"])

with tab1:
    st.header("Transliteration (ITRANS ↔ Telugu)")
    st.markdown("Convert Telugu text written in the English alphabet (ITRANS) to the native Telugu script and back.")
    
    itrans_input = st.text_area("Enter text in ITRANS format (e.g., 'namaste')")
    if itrans_input:
        try:
            telugu_script = transliterate(itrans_input, ITRANS, TELUGU)
            st.success(f"**Telugu Script:** {telugu_script}")
            
            reverse_itrans = transliterate(telugu_script, TELUGU, ITRANS)
            st.info(f"**Back to ITRANS:** {reverse_itrans}")
        except Exception as e:
            st.error(f"An error occurred during transliteration: {e}")

with tab2:
    st.header("Translation using IndicTrans2")
    st.markdown("Translate text between English and Telugu using AI4Bharat's state-of-the-art model.")

    direction = st.radio("Select Translation Direction", ["English ➜ Telugu", "Telugu ➜ English"], horizontal=True)
    
    input_text = st.text_area("Enter text to translate:")

    if st.button("Translate"):
        if not input_text.strip():
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                if direction == "English ➜ Telugu":
                    result = batch_translate([input_text], "eng_Latn", "tel_Telu", en_to_ind_model, en_to_ind_tokenizer, ip)
                else:
                    result = batch_translate([input_text], "tel_Telu", "eng_Latn", ind_to_en_model, ind_to_en_tokenizer, ip)
            st.success(f"**Translation:** {result[0]}")
