import streamlit as st

from src.process import load_book_long_name_df
from src.actions import get_bible_and_books
from src.streamlit_utils import set_up_streamlit

set_up_streamlit()

st.title("Read the Bible")

with st.expander("Choose Translation"):
    bible, bible_books = get_bible_and_books()
book_long_name_df = load_book_long_name_df()


def book_format_func(x):
    return book_long_name_df.loc[x]["English Name"]


book = st.selectbox("Book", options=bible_books.index, format_func=book_format_func)
bible_book_df = bible[bible["book"] == book]


def get_book_display(bible_book_df, book) -> str:
    m = f"## {book_format_func(book)}\n\n"
    last_chapter = 0
    for i in range(len(bible_book_df)):
        book_verse = bible_book_df.iloc[i].name
        book_verse_number = book_verse[4:]
        chapter, verse = tuple(int(s) for s in book_verse_number.split(":"))
        if chapter > last_chapter:
            m += "\n\n"
            m += f"### Chapter {chapter}\n\n"
            last_chapter = chapter
        text = bible_book_df.iloc[i].text
        m += f"<sup>**{verse}**</sup> {text} "
    return m


st.markdown(
    get_book_display(bible_book_df, book),
    unsafe_allow_html=True,
)
