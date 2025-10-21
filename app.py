'''# -*- coding: utf-8 -*-
import traceback
import warnings

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor

# ------------------------------
# Ignore deprecation and warnings
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
DEVICE = "cpu"
BATCH_SIZE = 4

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def initialize_models():
    """Load models from Hugging Face safely."""
    try:
        HF_ACCESS_TOKEN = st.secrets["HUGGING_FACE_HUB_TOKEN"]
    except KeyError:
        st.error("HUGGING_FACE_HUB_TOKEN missing in Streamlit secrets!")
        raise ValueError("Add HUGGING_FACE_HUB_TOKEN to .streamlit/secrets.toml")

    # Indic Processor
    ip = IndicProcessor(inference=True)

    # Model names (pinned)
    en_to_ind_model_name = "ai4bharat/indictrans2-en-indic-1B"
    ind_to_en_model_name = "ai4bharat/indictrans2-indic-en-1B"

    # Load English -> Telugu
    en_to_ind_tokenizer = AutoTokenizer.from_pretrained(
        en_to_ind_model_name,
        trust_remote_code=True,
        use_auth_token=HF_ACCESS_TOKEN
    )
    en_to_ind_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_to_ind_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        use_auth_token=HF_ACCESS_TOKEN
    )
    en_to_ind_model.eval()

    # Load Telugu -> English
    ind_to_en_tokenizer = AutoTokenizer.from_pretrained(
        ind_to_en_model_name,
        trust_remote_code=True,
        use_auth_token=HF_ACCESS_TOKEN
    )
    ind_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        ind_to_en_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        use_auth_token=HF_ACCESS_TOKEN
    )
    ind_to_en_model.eval()

    return ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer


# --- Translation Function ---
def batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip):
    results = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(sentences[i:i + BATCH_SIZE], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        processed = ip.postprocess_batch(decoded, lang=tgt_lang)
        cleaned = [s.replace("Tel_Telu None ", "").replace("ng @Latn ", "").strip() for s in processed]
        results += cleaned
    return results


# --- Main App ---
st.title("Telugu ↔ English AI Translator")

# Load models
try:
    with st.spinner("Loading AI models… This may take a few minutes on first run."):
        ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
    st.success("✅ Models loaded successfully! Ready to translate.")
except Exception as e:
    st.error("🚨 Failed to load AI models. Check Hugging Face token and internet connection.")
    st.text(traceback.format_exc())
    st.stop()

# Layout
col1, col2 = st.columns(2)

# English -> Telugu
with col1:
    st.markdown("#### English ➡️ Telugu")
    en_input = st.text_area("Enter English text:", key="en_input", height=150)
    if st.button("Translate to Telugu", use_container_width=True):
        if en_input.strip():
            with st.spinner("Translating…"):
                try:
                    result = batch_translate([en_input], "eng_Latn", "tel_Telu", en_to_ind_model, en_to_ind_tokenizer, ip)
                    st.info("Telugu Translation:")
                    st.markdown(f"> {result[0]}")
                except Exception as e:
                    st.error("Translation failed. Check logs.")
                    st.text(traceback.format_exc())
        else:
            st.warning("Please enter some text to translate.")

# Telugu -> English
with col2:
    st.markdown("#### Telugu ➡️ English")
    te_input = st.text_area("Enter Telugu text:", key="te_input", height=150)
    if st.button("Translate to English", use_container_width=True):
        if te_input.strip():
            with st.spinner("Translating…"):
                try:
                    result = batch_translate([te_input], "tel_Telu", "eng_Latn", ind_to_en_model, ind_to_en_tokenizer, ip)
                    st.info("English Translation:")
                    st.markdown(f"> {result[0]}")
                except Exception as e:
                    st.error("Translation failed. Check logs.")
                    st.text(traceback.format_exc())
        else:
            st.warning("Please enter some text to translate.")
'''







# -*- coding: utf-8 -*-
import traceback
import warnings

import streamlit as st
from transformers import pipeline
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor

# ------------------------------
# Ignore warnings
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
DEVICE = -1  # CPU for Hugging Face pipeline

# --- Initialize models ---
@st.cache_resource(show_spinner=False)
def initialize_models():
    """Load smaller translation models (CPU-friendly, public)"""
    ip = IndicProcessor(inference=True)

    # English -> Telugu
    en_to_te_translator = pipeline(
        task="translation",
        model="VijayChandra/english-to-telugu-translator-nllb",
        src_lang="eng_Latn",
        tgt_lang="tel_Telu",
        device=DEVICE
    )

    # Telugu -> English
    te_to_en_translator = pipeline(
        task="translation",
        model="ai4bharat/IndicBARTSS",
        src_lang="tel_Telu",
        tgt_lang="eng_Latn",
        device=DEVICE
    )

    return ip, en_to_te_translator, te_to_en_translator

# --- Translation Function ---
def translate_text(text, direction="en_to_te"):
    """Translate single string using the chosen pipeline"""
    if direction == "en_to_te":
        return en_to_te_translator(text)[0]['translation_text']
    else:
        return te_to_en_translator(text)[0]['translation_text']

# --- Main App ---
st.title("Telugu ↔ English AI Translator")

# Load models
try:
    with st.spinner("Loading AI models… This may take a few seconds on first run."):
        ip, en_to_te_translator, te_to_en_translator = initialize_models()
    st.success("✅ Models loaded successfully! Ready to translate.")
except Exception:
    st.error("🚨 Failed to load AI models.")
    st.text(traceback.format_exc())
    st.stop()

# --- UI Layout ---
col1, col2 = st.columns(2)

# English -> Telugu
with col1:
    st.markdown("#### English ➡️ Telugu")
    en_input = st.text_area("Enter English text:", key="en_input", height=150)
    if st.button("Translate to Telugu", use_container_width=True):
        if en_input.strip():
            try:
                with st.spinner("Translating…"):
                    translated = translate_text(en_input, "en_to_te")
                    st.info("Telugu Translation:")
                    st.markdown(f"> {translated}")
            except Exception:
                st.error("Translation failed. Check logs.")
                st.text(traceback.format_exc())
        else:
            st.warning("Please enter some text to translate.")

# Telugu -> English
with col2:
    st.markdown("#### Telugu ➡️ English")
    te_input = st.text_area("Enter Telugu text:", key="te_input", height=150)
    if st.button("Translate to English", use_container_width=True):
        if te_input.strip():
            try:
                with st.spinner("Translating…"):
                    translated = translate_text(te_input, "te_to_en")
                    st.info("English Translation:")
                    st.markdown(f"> {translated}")
            except Exception:
                st.error("Translation failed. Check logs.")
                st.text(traceback.format_exc())
        else:
            st.warning("Please enter some text to translate.")
