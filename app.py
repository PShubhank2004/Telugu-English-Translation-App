'''# -*- coding: utf-8 -*-
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
            st.warning("Please enter some text to translate.")'''


















'''import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- CONFIGURATION AND SETUP ---

# 1. NEW MODEL: Switched to a significantly smaller model (mT5-small)
# to avoid Out-of-Memory (OOM) errors on Streamlit Community Cloud.
# This model is ~1.2 GB (Float32) compared to the NLLB-600M model's 2.4 GB.
MODEL_NAME = "Helsinki-NLP/opus-mt-en-te"#"google/mt5-small"
SOURCE_LANG_CODE = "te" # Telugu
TARGET_LANG_CODE = "en" # English

# mT5 uses specific prefixes for translation tasks
# The 'en' prefix tells the model the target language should be English.
MT5_ENCODE_PREFIX = f"translate {SOURCE_LANG_CODE} to {TARGET_LANG_CODE}: "

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_model():
    """Loads the model and tokenizer from Hugging Face."""
    try:
        # Load the model and tokenizer in full precision (no bitsandbytes needed)
        #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # New line (Fixes the error by forcing the slow/SentencePiece tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {MODEL_NAME}: {e}")
        st.info("The app will not work until the model loads correctly.")
        return None, None

tokenizer, model = load_model()

# --- TRANSLATION FUNCTION ---

def translate_te_to_en(text_to_translate):
    """Encodes the input, generates the translation, and decodes the output."""
    if not model or not tokenizer:
        return "Model not loaded. Please check the deployment logs."

    # 1. Add the necessary prefix for mT5 translation
    input_text = MT5_ENCODE_PREFIX + text_to_translate

    # 2. Encode the input text
    # We use a context manager to ensure tensors are moved to CPU if necessary
    # (Streamlit free tier is CPU-only)
    with st.spinner("Translating..."):
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # 3. Generate the translation
        # NOTE: Using sensible defaults for generation parameters
        output_ids = model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            do_sample=False,  # Use beam search for higher quality
            # mT5 does not strictly require a decoder_start_token_id for simple translation
        )

        # 4. Decode the generated IDs
        translated_text = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
        return translated_text

# --- STREAMLIT APP LAYOUT ---

st.title("🇮🇳 Telugu (తెలుగు) to English Translator 🇬🇧 (v2 - mT5-small)")

st.markdown("""
This is a test application using the **smaller `google/mt5-small` model** to prevent memory (OOM) crashes on the Streamlit free cloud.
""")

# Input Area
telugu_input = st.text_area(
    "Enter Telugu Text Here:",
    placeholder="ఉదాహరణకు: మీ పేరు ఏమిటి?",
    height=150
)

# Translation Button
if st.button("Translate", type="primary"):
    if not telugu_input.strip():
        st.warning("Please enter some Telugu text to translate.")
    else:
        # Perform Translation
        translation = translate_te_to_en(telugu_input)

        # Output Area
        st.subheader("Translation Result:")
        st.success(translation)

st.markdown("---")
st.caption(f"Model: {MODEL_NAME}")'''










import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# --- Model Configuration ---
# The recommended model for bi-directional English <-> Telugu translation
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# Language codes for mBART-50
LANG_CODE_EN = "en_XX"
LANG_CODE_TE = "te_IN"

@st.cache_resource
def load_model():
    """Loads the mBART-50 model and tokenizer using st.cache_resource."""
    try:
        # Load the tokenizer
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
        # Load the model
        model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        # Set to evaluation mode and move to CPU/GPU if necessary (default is CPU for Streamlit free tier)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

def translate_text(text: str, src_lang: str, tgt_lang: str, tokenizer, model):
    """Performs the translation using the mBART model."""
    if not text or not tokenizer or not model:
        return "Model not loaded or input text is empty."

    try:
        # 1. Set the source language for the tokenizer
        tokenizer.src_lang = src_lang

        # 2. Get the token ID for the target language (to force as the first generated token)
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

        # 3. Tokenize the input
        # Note: 'pt' stands for PyTorch tensors, which is standard for Hugging Face models
        encoded_input = tokenizer(text, return_tensors="pt")

        # 4. Generate the translation
        with torch.no_grad(): # Disable gradient calculations for faster inference
            generated_tokens = model.generate(
                **encoded_input,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,  # Set a reasonable max length for the output
                num_beams=5,     # Use beam search for higher quality translations
            )

        # 5. Decode and return the result
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    except KeyError:
        return f"Error: Language code '{src_lang}' or '{tgt_lang}' not recognized by the tokenizer."
    except Exception as e:
        return f"An error occurred during translation: {e}"

# --- Streamlit App Interface ---
st.title("🔀 Multilingual Translator (English ↔ Telugu)")
st.caption(f"Using **{MODEL_NAME}** ($\approx 610$M parameters) for direct many-to-many translation.")

# Load the model and tokenizer only once
tokenizer, model = load_model()

if tokenizer and model:

    # Dropdown to select translation direction
    direction_options = {
        "English (EN) → Telugu (TE)": (LANG_CODE_EN, LANG_CODE_TE),
        "Telugu (TE) → English (EN)": (LANG_CODE_TE, LANG_CODE_EN),
    }

    selected_direction = st.selectbox(
        "Select Translation Direction:",
        list(direction_options.keys())
    )

    source_lang, target_lang = direction_options[selected_direction]

    # Determine placeholder text for the input box
    if source_lang == LANG_CODE_EN:
        source_label = "English Text"
        placeholder_text = "Enter the English text to translate..."
    else:
        source_label = "Telugu Text"
        placeholder_text = "తెలుగు వచనాన్ని ఇక్కడ నమోదు చేయండి..." # Telugu for "Enter Telugu text here..."

    # Input text area
    input_text = st.text_area(
        source_label,
        placeholder=placeholder_text,
        height=150
    )

    # Translation button
    if st.button("Translate", type="primary"):
        if input_text:
            with st.spinner(f"Translating from {source_lang} to {target_lang}..."):
                # Call the translation function
                translated_text = translate_text(
                    input_text,
                    source_lang,
                    target_lang,
                    tokenizer,
                    model
                )

            # Display the result
            st.subheader("Translation Result")
            st.success(translated_text)
        else:
            st.warning("Please enter some text to translate.")

else:
    # This block displays if model loading failed
    st.error("The translation service is currently unavailable. Please check the model loading logs.")