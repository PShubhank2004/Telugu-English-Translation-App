'''# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor
import time
# trigger redeploy

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🌏",
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
st.title("Telugu ↔ English AI Translator")

# Load models and display status
with st.spinner("Loading AI models... This might take a moment on first startup."):
    ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
st.success("Models loaded.")
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

'''





























'''
# -*- coding: utf-8 -*-
import streamlit as st
import torch
import os # 🌟 CHANGE 1: Import os to access environment variables (Streamlit secrets)
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor
import time
# trigger redeploy

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🌏",
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
    """Loads and caches all the required models and tokenizers, using the HF_TOKEN."""

    # 🌟 CHANGE 2: Read the HF_TOKEN secret
    HF_ACCESS_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_ACCESS_TOKEN:
        # This message appears if you forget to set the secret in Streamlit Cloud or secrets.toml
        st.error("Authentication Error: HF_TOKEN secret is missing. This is required to download the IndicTrans2 models.")
        # Raise an exception to stop deployment until the key is available
        raise ValueError("HF_TOKEN environment variable not set. Please configure your Streamlit secrets.")

    ip = IndicProcessor(inference=True)

    # 1. English to Indic Model (en-indic)
    en_to_ind_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    en_to_ind_tokenizer = AutoTokenizer.from_pretrained(
        en_to_ind_ckpt_dir,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN # 🔑 CHANGE 3: Pass token for authentication
    )
    en_to_ind_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_to_ind_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
        token=HF_ACCESS_TOKEN # 🔑 CHANGE 3: Pass token for authentication
    )

    # 2. Indic to English Model (indic-en)
    ind_to_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
    ind_to_en_tokenizer = AutoTokenizer.from_pretrained(
        ind_to_en_ckpt_dir,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN # 🔑 CHANGE 3: Pass token for authentication
    )
    ind_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        ind_to_en_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
        token=HF_ACCESS_TOKEN # 🔑 CHANGE 3: Pass token for authentication
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
st.title("Telugu ↔ English AI Translator")

# Load models and display status
with st.spinner("Loading AI models... This might take a moment on first startup."):
    # If the token is missing, initialize_models will raise an exception and stop the app.
    ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
st.success("Models loaded.")
st.divider()

# --- Main UI Tab ---
# 🌟 CHANGE 4: Removed the tabs and only kept the main translation section.
st.markdown("### 🌐 AI Translator")
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

# Removed the entire block for 'main_tab2' (transliteration)'''





# -*- coding: utf-8 -*-
import streamlit as st
import torch
# import os # REMOVE: No longer needed, as we use st.secrets
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU
from IndicTransToolkit.processor import IndicProcessor
import time
# trigger redeploy

# --- Page Configuration ---
st.set_page_config(
    page_title="IndicTrans2 | Telugu-English",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants and Configuration ---
# Use a smaller model if crashing due to MemoryError on free tier
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# On Streamlit Cloud, it's often better to force to CPU unless you pay for GPU tier:
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
BATCH_SIZE = 4
# QUANTIZATION is not used, so we can remove it or keep it as None
# QUANTIZATION = None

# --- Model Loading ---
@st.cache_resource
def initialize_models():
    """Loads and caches all the required models and tokenizers, using the HF_TOKEN."""

    # 🌟 CRITICAL CHANGE: Read the HF_TOKEN secret using st.secrets
    # Streamlit Cloud recommends st.secrets for securely accessing the Secrets panel.
    # The key is typically HUGGING_FACE_HUB_TOKEN for Hugging Face
    try:
        HF_ACCESS_TOKEN = st.secrets["HUGGING_FACE_HUB_TOKEN"]
    except KeyError:
        st.error("Authentication Error: HUGGING_FACE_HUB_TOKEN secret is missing. This is required to download the IndicTrans2 models.")
        # Raise an exception to stop deployment until the key is available
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in Streamlit secrets.")

    # 🌟 CRITICAL CHANGE: If you are hitting a MemoryError, you must use a smaller model,
    # or uncomment the quantization/8-bit lines below.

    # 1. IndicProcessor
    ip = IndicProcessor(inference=True)

    # 2. English to Indic Model (en-indic)
    en_to_ind_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    en_to_ind_tokenizer = AutoTokenizer.from_pretrained(
        en_to_ind_ckpt_dir,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN # 🔑 Pass token for authentication
    )
    en_to_ind_model = AutoModelForSeq2SeqLM.from_pretrained(
        en_to_ind_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
        token=HF_ACCESS_TOKEN # 🔑 Pass token for authentication
    )

    # 3. Indic to English Model (indic-en)
    ind_to_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
    ind_to_en_tokenizer = AutoTokenizer.from_pretrained(
        ind_to_en_ckpt_dir,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN # 🔑 Pass token for authentication
    )
    ind_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
        ind_to_en_ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
        token=HF_ACCESS_TOKEN # 🔑 Pass token for authentication
    )

    # Move models to device and set half precision on GPU
    en_to_ind_model = en_to_ind_model.to(DEVICE)
    ind_to_en_model = ind_to_en_model.to(DEVICE)
    if DEVICE == "cuda":
        # Note: half() might not be supported on all free-tier GPUs.
        # You might need to remove this or use quantization if you hit an error.
        en_to_ind_model.half()
        ind_to_en_model.half()
    en_to_ind_model.eval()
    ind_to_en_model.eval()
    return ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer

# --- Core Translation Function ---
# (No changes needed here - it looks great!)
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
st.title("Telugu ↔ English AI Translator")

# Load models and display status
# The exception handling here is crucial for the deployment to show the error message.
try:
    with st.spinner("Loading AI models... This might take a few minutes on first startup if downloading the 1B models."):
        ip, en_to_ind_model, en_to_ind_tokenizer, ind_to_en_model, ind_to_en_tokenizer = initialize_models()
    st.success("Models loaded successfully!")
    st.divider()

    # --- Main UI Tab ---
    st.markdown("### 🌐 AI Translator")
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

except Exception as e:
    st.error(f"Failed to run the application after model loading due to an error: {e}")
    st.markdown("---")
    st.markdown("**Troubleshooting Tip:** If this is a `ValueError` about the token, make sure you have added a `HUGGING_FACE_HUB_TOKEN` in your Streamlit secrets.")

# Removed the entire block for 'main_tab2' (transliteration)