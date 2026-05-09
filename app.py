"""
Streamlit web app for Code-Mixed Text Summarization.
Run: streamlit run app.py
"""

import streamlit as st
from src.components.inference.predictor import Predictor

st.set_page_config(
    page_title="Code-Mixed Text Summarizer",
    page_icon="",
    layout="centered"
)

EXAMPLES = [
    "aaj bharat ne cricket match jeeta aur poori team ne bahut achha khela",
    "<2hi> सरकार ने नई policy बनाई है जिससे education sector में बड़ा बदलाव आएगा",
    "<2hi> भारत में flood की situation बेहद serious है, government rescue operation चला रही है",
]


@st.cache_resource
def load_predictor():
    return Predictor()


def main():
    st.title("🌐 Code-Mixed Text Summarizer")
    st.markdown("Summarizes **Hinglish, Hindi, Bengali, and Gujarati** articles using IndicBART.")
    st.divider()

    st.subheader("Input Text")
    example_choice = st.selectbox("Load an example (optional)", ["— Select —"] + EXAMPLES)

    default_text = "" if example_choice == "— Select —" else example_choice
    text_input = st.text_area("Paste your article text here:", value=default_text, height=200)

    if st.button("Summarize", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
            return

        with st.spinner("Generating summary..."):
            try:
                predictor = load_predictor()
                summary = predictor.predict(text_input)

                st.divider()
                st.subheader("Generated Summary")
                st.success(summary)

            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.caption("Model: ai4bharat/IndicBART fine-tuned on ILSUM + HinGE datasets")


if __name__ == "__main__":
    main()
