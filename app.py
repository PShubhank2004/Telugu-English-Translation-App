# -*- coding: utf-8 -*-
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# --- Model Selection ---
# If running on free-tier (CPU only) or low GPU memory, fallback to smaller 100M models
USE_SMALL_MODEL = DEVICE == "cpu"  # Auto fallback for CPU or low GPU memory

if USE_SMALL_MODEL:
    EN_IND_MODEL_ID = "ai4bharat/indictrans2-en-indic-100M"
    IND_EN_MODEL_ID = "ai4bharat/indictrans2-indic-en-100M"
    st.info("⚠️ Using smaller 100M models to avoid memory issues on free-tier / CPU.")
else:
    EN_IND_MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"
    IND_EN_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"


# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def initialize_models():
    """
    Load and cache the IndicTrans2 models and tokenizers.
    Reads Hugging Face token from Streamlit secrets.
    """
    # --- Hugging Face Token ---
    try:
        HF_ACCESS_TOKEN = st.secrets["HUGGING_FACE_HUB_TOKEN"]
    except KeyError:
        raise ValueError(
            "HUGGING_FACE_HUB_TOKEN not found in Streamlit secrets. "
            "Add it via .streamlit/secrets.toml"
        )

    # --- Indic Processor ---
    ip = IndicProcessor(inference=True)

    # --- English to Indic ---
    en_to_ind_tokenizer = AutoTokenizer.from_pretrained(
        EN_IND_MODEL_ID, trust_remote_code=True, token=HF_ACCESS_TOKEN
    )
    en_to_ind_model = AutoModelForSeq2SeqLM.from_pretrained(
        EN_IND_MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
        token=HF_ACCESS_TOKEN,
    )

    # --- Indic to English ---
    ind_to_en_tokenizer = AutoTokenizer.from_pretrained(
        IND_EN_MODEL_ID, trust_remote_code=True, token=HF_ACCESS_TOKEN
    )
    ind_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        IND_EN_MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
        token=HF_ACCESS_TOKEN,
    )

    # --- Move to device ---
    en_to_ind_model = en_to_ind_model.to(DEVICE)
    ind_to_en_model = ind_to_en_model.to(DEVICE)
    if DEVICE == "cuda":
        try:
            en_to_ind_model.half()
            ind_to_en_model.half()
        except Exception:
            st.warning("Half precision not supported on this GPU. Using default precision.")

    en_to_ind_model.eval()
    ind_to_en_model.eval()

    return ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer


# --- Translation Function ---
def batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(sentences[i:i + BATCH_SIZE], src_lang, tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1
            )
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        postprocessed = ip.postprocess_batch(decoded, lang=tgt_lang)
        cleaned = [s.replace("Tel_Telu None ", "").replace("ng @Latn ", "").strip() for s in postprocessed]
        translations.extend(cleaned)
    return translations


# --- UI ---
st.title("Telugu ↔ English AI Translator")

try:
    with st.spinner("Loading AI models... This may take a few minutes on first startup."):
        ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
    st.success("Models loaded successfully!")
    st.divider()

    st.markdown("### 🌐 AI Translator")
    col1, col2 = st.columns(2)

    # English -> Telugu
    with col1:
        st.markdown("#### English ➡️ Telugu")
        en_input = st.text_area("Enter English text:", height=150)
        if st.button("Translate to Telugu", key="en_button"):
            if en_input.strip():
                with st.spinner("Translating..."):
                    result = batch_translate([en_input], "eng_Latn", "tel_Telu", en_to_ind_model, en_to_ind_tokenizer, ip)
                    st.info("Telugu Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter text to translate.")

    # Telugu -> English
    with col2:
        st.markdown("#### Telugu ➡️ English")
        te_input = st.text_area("Enter Telugu text:", height=150)
        if st.button("Translate to English", key="te_button"):
            if te_input.strip():
                with st.spinner("Translating..."):
                    result = batch_translate([te_input], "tel_Telu", "eng_Latn", ind_to_en_model, ind_to_en_tokenizer, ip)
                    st.info("English Translation:")
                    st.markdown(f"> {result[0]}")
            else:
                st.warning("Please enter text to translate.")

except ValueError as ve:
    st.error(f"Configuration Error: {ve}")
except Exception as e:
    st.error(f"Unexpected error occurred: {e}")
    st.markdown("---")
    st.markdown(
        "**Tip:** Ensure your `HUGGING_FACE_HUB_TOKEN` is correct and you have a stable internet connection."
    )
