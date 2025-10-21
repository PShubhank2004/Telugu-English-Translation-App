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