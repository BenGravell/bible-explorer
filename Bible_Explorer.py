import streamlit as st

from src.streamlit_utils import set_up_streamlit

set_up_streamlit()

st.title("Welcome to the Bible Explorer")

st.markdown("""
This app supports exploration of the text of the Bible in many translations and many languages, as provided by [BibleNLP/ebible](https://github.com/BibleNLP/ebible).

Use the sidebar to navigate to pages dedicated to:

- Reading the Bible.
- Analyzing the relationship between books of the Bible using data-driven / machine-learning techniques.
""")
