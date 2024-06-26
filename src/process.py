from requests import HTTPError

import numpy as np
import pandas as pd
import streamlit as st


from src.web import fetch_text_from_url


def split_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


@st.cache_data
def load_vref_df() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata/vref.txt"
    vref = pd.read_csv(url, sep=" ", header=None)
    vref.columns = ["book", "verse"]
    vref.index = vref["book"] + " " + vref["verse"]
    vref.index.name = "book_verse"
    return vref


@st.cache_data
def load_bible(translation: str) -> pd.DataFrame:
    # Load the vref
    vref_df = load_vref_df()

    try:
        # Load the bible text
        url = f"https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/{translation}.txt"
        bible_text = fetch_text_from_url(url, timeout=10)
    except HTTPError:
        st.error("Selected translation of the Bible not available from the web. Please select a different translation.")

    # Convert to dataframe
    bible = pd.DataFrame(split_lines(bible_text))
    bible.index = vref_df.index
    bible.columns = ["text"]
    bible = bible.join(vref_df)
    return bible


@st.cache_data
def load_translations_df() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata/translations.csv"
    translations_df = pd.read_csv(url)

    # Set the index as the language code + ID, same as the corpus filenames.
    translations_df.index = translations_df["languageCode"] + "-" + translations_df["translationId"]

    # Sort by languages with the most translations
    translations_df["languageCodeCount"] = translations_df.groupby("languageCode")["languageCode"].transform("count")
    translations_df = translations_df.sort_values(
        ["languageCodeCount", "languageNameInEnglish", "title"],
        ascending=[False, True, True],
    ).drop(columns="languageCodeCount")

    return translations_df


@st.cache_data
def load_book_long_name_df():
    book_long_name_df = pd.read_csv("data/books.csv")
    book_long_name_df = book_long_name_df.set_index("Identifier")
    book_long_name_df.index.name = "book"
    return book_long_name_df


def get_bible_books(bible, translation_row, include_apocrypha, drop_empty_books):
    bible_books = bible.groupby("book", sort=False)["text"].apply(lambda x: " ".join(x))

    testaments = ["OT", "NT"]

    if not include_apocrypha:
        # TODO put in a .txt file
        apocrypha_books = [
            "TOB",
            "JDT",
            "ESG",
            "WIS",
            "SIR",
            "BAR",
            "LJE",
            "S3Y",
            "SUS",
            "BEL",
            "1MA",
            "2MA",
            "3MA",
            "4MA",
            "1ES",
            "2ES",
            "MAN",
            "PS2",
            "ODA",
            "PSS",
            "EZA",
            "JUB",
            "ENO",
        ]
        bible_books = bible_books.drop(apocrypha_books)
    else:
        testaments.append("DC")

    if drop_empty_books:
        # If more than 100 verses are blank, or more than 50% of the verses are blank, consider the book empty.
        bible["blank"] = bible["text"].str.strip().replace("", np.nan).isna()
        blank_ratio = (bible.groupby("book")["blank"].sum() / bible.groupby("book")["blank"].count()).rename(
            "blank_ratio"
        )
        blank_count = bible.groupby("book")["blank"].sum().rename("blank_count")
        blank_stats = pd.concat([blank_ratio, blank_count], axis=1)
        blank_stats["empty"] = (blank_stats["blank_count"] > 100) | (blank_stats["blank_ratio"] > 0.5)
        bible_books = bible_books[~blank_stats["empty"]]

    # Warnings if there are missing books
    testament_book_counts = {testament: translation_row[f"{testament}books"] for testament in testaments}
    expected_testament_book_counts = {"OT": 39, "NT": 27, "DC": 23}
    if not drop_empty_books:
        # Only issue warnings if we have not dropped the empty books already
        for testament in testaments:
            actual_count = testament_book_counts[testament]
            expected_count = expected_testament_book_counts[testament]
            if actual_count < expected_count:
                st.warning(
                    f"In the {testament}, found only {actual_count} books but expected {expected_count}. This may lead"
                    " to unexpected results."
                )

    return bible_books
