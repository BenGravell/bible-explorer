import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
import umap
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st


from src.distance import compute_distance_matrix
from src.process import load_book_long_name_df
from src.actions import get_bible_and_books
from src.streamlit_utils import set_up_streamlit


set_up_streamlit()

st.title("Textual Distance Analysis of the Bible")


with st.expander("Choose Translation"):
    bible, bible_books = get_bible_and_books()

book_long_name_df = load_book_long_name_df()


def compute_distance_matrix_callback(df, labels):
    st.session_state["D"] = compute_distance_matrix(df, labels)


def main(bible_books):
    labels = (book_long_name_df.loc[bible_books.index])["English Name"]

    with st.expander("Distance Measure Details"):
        st.markdown(
            "The distance between two books of the bible is determined using the **Normalized Compression Distance** on"
            " the book text, which is a computable approximation to the information-theoretic Kolmogorov complexity."
            ' The concrete compressor used is `gzip`. See the paper ["Less is More: Parameter-Free Text Classification'
            ' with Gzip"](https://arxiv.org/abs/2212.09410) for more details.'
        )

    st.button(
        "Compute Distance Matrix",
        on_click=compute_distance_matrix_callback,
        args=[bible_books, labels],
    )

    D = st.session_state.get("D")
    if D is None:
        st.info("Distance matrix has not been computed yet, no results to show.")
        return

    tab_names = ["Distance Matrix", "Hierarchical Clustering", "Neighbor Graph"]
    tabs = st.tabs(tab_names)

    with tabs[tab_names.index("Distance Matrix")]:
        E = np.copy(D)
        np.fill_diagonal(E, np.nan)
        if st.toggle("Log transform distances"):
            # Convert [0, 1] distance to a [0, inf) distance
            E = -np.log(1.0 - E)
        fig = go.Figure(data=go.Heatmap(z=E, x=labels, y=labels))
        fig.update_layout(height=1200)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[tab_names.index("Hierarchical Clustering")]:

        def dendrogram_distfunc(df):
            return squareform(compute_distance_matrix(df, labels))

        fig = ff.create_dendrogram(
            bible_books,
            distfun=dendrogram_distfunc,
            labels=labels,
            orientation="bottom",
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[tab_names.index("Neighbor Graph")]:
        cols = st.columns(3)
        with cols[0]:
            k = st.number_input("Number of Neighbors", min_value=1, max_value=10, value=3)
        with cols[1]:
            n_neighbors = st.number_input("UMAP `n_neighbors`", min_value=2, max_value=50, value=10)
        with cols[2]:
            min_dist = st.number_input("UMAP `min_dist`", min_value=0.00, max_value=1.00, value=0.1)

        # Nearest neighbors using precomputed distance
        nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed")
        nbrs.fit(D)
        distances, indices = nbrs.kneighbors()

        # UMAP embedding
        umap_model = umap.UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=7,
        )
        X_embedded = umap_model.fit_transform(D)

        def minmax_normalize(x):
            return (x - x.min()) / (x.max() - x.min())

        X_embedded[:, 0] = minmax_normalize(X_embedded[:, 0])
        X_embedded[:, 1] = minmax_normalize(X_embedded[:, 1])

        # Prepare the line segments for the k-nearest neighbor graph
        segments_x = []
        segments_y = []

        for i in range(len(X_embedded)):
            for j in indices[i]:
                segments_x.extend([X_embedded[i][0], X_embedded[j][0], None])
                segments_y.extend([X_embedded[i][1], X_embedded[j][1], None])

        # Create dataframe for plot
        plot_df = pd.DataFrame(X_embedded, columns=["umap_x", "umap_y"])
        plot_df.index = bible_books.index
        plot_df["testament"] = (book_long_name_df.loc[bible_books.index])["Testament"]
        plot_df["name"] = (book_long_name_df.loc[bible_books.index])["English Name"]

        # Scatter plot of points
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color="testament",
            hover_name="name",
            height=800,
        )
        fig.update_traces(marker=dict(size=12))

        # Add trace for the k-nearest neighbor graph edges
        fig.add_trace(
            go.Scatter(
                x=segments_x,
                y=segments_y,
                mode="lines",
                line=dict(color="black"),
                opacity=0.5,
                name="k-nearest neighbors",
            )
        )

        # Plot options
        fig.update_layout(title=f"k-Nearest Neighbor Graph (k={k})")
        # fig.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig, use_container_width=True)


main(bible_books)
