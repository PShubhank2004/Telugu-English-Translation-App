# 🇮🇳 Telugu ↔ English AI Translator & Transliterator

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=streamlit)](https://telugu-english-translation-app-r-adarsh-0120.streamlit.app/)

A clean, fast, and intuitive web application for high-quality translation between English and Telugu, and for transliterating between the Telugu script and ITRANS (English keyboard). This app is built with Streamlit and powered by AI4Bharat's state-of-the-art IndicTrans2 models.

## ✨ Features

* **Bilingual Translation:**
    * Translate text from **English to Telugu** with high accuracy.
    * Translate text from **Telugu to English** while preserving context.
* **Script Conversion (Transliteration):**
    * Convert Telugu text written in the English alphabet (ITRANS format) into the native Telugu script.
    * Convert native Telugu script back into the ITRANS format.
* **User-Friendly Interface:**
    * A clean, side-by-side layout for easy translation.
    * Separate, dedicated tabs for translation and transliteration functionalities.
    * Fast, responsive performance thanks to Streamlit's caching.

## 🛠️ Tech Stack

* **Framework:** [Streamlit](https://streamlit.io/)
* **Language:** Python
* **ML/AI Libraries:**
    * [PyTorch](https://pytorch.org/)
    * [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* **Translation Models:**
    * `ai4bharat/indictrans2-en-indic-1B`
    * `ai4bharat/indictrans2-indic-en-1B`
* **Helper Libraries:**
    * [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) for model processing.
    * [indic-transliteration](https://pypi.org/project/indic-transliteration/) for script conversion.

## 🚀 Running the App Locally

To run this application on your local machine, please follow these steps.

### Prerequisites

* Git
* Python (version 3.11 is recommended)
* `pip` and `venv`

### 1. Clone the Repository

```bash
git clone [https://github.com/adarsh-0120/telugu-english-translation-app.git](https://github.com/adarsh-0120/telugu-english-translation-app.git)
cd telugu-english-translation-app
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

* **For macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
* **For Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 3. Install Dependencies

This project requires both system-level and Python-level dependencies.

* **System Dependencies (for Debian/Ubuntu):**
    These are listed in `packages.txt` and are required to build some of the Python libraries.
    ```bash
    sudo apt-get update && sudo apt-get install -y cmake build-essential
    ```
* **Python Dependencies:**
    These are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### 4. Run the Streamlit App

Once all dependencies are installed, you can run the app with a single command:

```bash
streamlit run app.py
```

The application should now be open and running in your web browser!

## Acknowledgements

This project would not be possible without the incredible work done by the **AI4Bharat** team in creating and open-sourcing the **IndicTrans2** models. Their contributions to Indian language AI are foundational to this application.
