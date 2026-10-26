
import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure the application
st.set_page_config(
    page_title="Patient Case Semantic Search",
    page_icon="🔎",
    layout="wide"
)

# Project paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    PROJECT_PATH, "data", "patient_cases_cleaned.csv"
)

st.title("🔎 Patient Case Semantic Search")
st.write(
    "Search synthetic clinical notes using natural-language queries "
    "and semantic similarity."
)

st.info(
    "Educational information-retrieval prototype only. "
    "This application is not intended for diagnosis or treatment decisions."
)

# Load dataset and model only once
@st.cache_resource
def load_resources():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["note"]).copy()
    df["note"] = df["note"].astype(str).str.strip()
    df = df[df["note"] != ""].reset_index(drop=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(
        df["note"].tolist(),
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return df, model, embeddings


try:
    with st.spinner("Loading clinical notes and search model..."):
        df, model, embeddings = load_resources()

    st.success(f"Search system ready: {len(df)} clinical notes loaded.")

    query = st.text_area(
        "Enter a symptom or clinical description",
        placeholder="Example: Patient with chest pain and shortness of breath"
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Search cases", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            query_embedding = model.encode(
                [query.strip()],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            scores = cosine_similarity(
                query_embedding, embeddings
            )[0]

            top_indices = np.argsort(scores)[::-1][:top_k]

            st.subheader("Most semantically similar cases")

            for rank, idx in enumerate(top_indices, start=1):
                row = df.iloc[idx]

                with st.container(border=True):
                    st.markdown(f"### Result {rank}")

                    if "patient_id" in df.columns:
                        st.caption(f"Record ID: {row['patient_id']}")

                    st.metric(
                        "Text similarity",
                        f"{scores[idx]:.3f}"
                    )

                    st.write(row["note"])

                    if "task" in df.columns and pd.notna(row["task"]):
                        st.caption(f"Task: {row['task']}")

                    if "question" in df.columns and pd.notna(row["question"]):
                        with st.expander("Related question"):
                            st.write(row["question"])

except FileNotFoundError:
    st.error(
        "Cleaned dataset not found. Please check that "
        "data/patient_cases_cleaned.csv exists."
    )

except Exception as exc:
    st.error(f"Unable to load the search system: {exc}")

