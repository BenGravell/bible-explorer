import streamlit as st

from src.process import (
    load_bible,
    load_translations_df,
    get_bible_books,
)


def get_bible_and_books():
    translations_df = load_translations_df()

    # Choose languages
    language_options = translations_df["languageNameInEnglish"].unique()
    language_selection = st.multiselect(
        "Languages",
        options=language_options,
        default=language_options[0],
        help="Sorted by frequency of language in the source metadata.",
    )
    if not language_selection:
        language_selection = language_options

    # Choose translation
    translation_options = (
        (translations_df[translations_df["languageNameInEnglish"].isin(language_selection)]).sort_values("title").index
    ).to_list()

    def translation_options_format_func(idx):
        row = translations_df.loc[idx]
        return f"{row.title} ({row.FCBHID[3:]})"

    deafult_translation = "eng-engBBE"
    default_index = translation_options.index(deafult_translation) if deafult_translation in translation_options else 0

    translation = st.selectbox(
        "Bible Translation",
        options=translation_options,
        index=default_index,
        format_func=translation_options_format_func,
    )
    translation_row = translations_df.loc[translation]

    include_apocrypha = st.toggle("Include Apocrypha?")
    drop_empty_books = st.toggle("Drop empty books?")

    bible = load_bible(translation)
    bible_books = get_bible_books(bible, translation_row, include_apocrypha, drop_empty_books)
    return bible, bible_books
