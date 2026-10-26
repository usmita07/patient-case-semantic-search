
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

# Hugging Face dataset configuration
DATASET_NAME = "aisc-team-a1/Asclepius-Synthetic-Clinical-Notes"
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
    from datasets import load_dataset

    # Stream a sample instead of downloading the entire dataset
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    records = []

    for i, record in enumerate(dataset):
        if i >= 2000:
            break

        note = record.get("note")

        if note is not None and str(note).strip():
            records.append({
                "patient_id": record.get("patient_id", i),
                "note": str(note).strip(),
                "task": record.get("task", ""),
                "question": record.get("question", "")
            })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No valid clinical notes were loaded.")

    df = df.drop_duplicates(subset=["note"]).reset_index(drop=True)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        df["note"].tolist(),
        batch_size=4,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return df, model, embeddings

except Exception as exc:
    st.error(f"Unable to load the search system: {exc}")


