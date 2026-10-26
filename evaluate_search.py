
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Locate the project and dataset
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    project_path, "data", "patient_cases_cleaned.csv"
)

df = pd.read_csv(data_path)
df = df.dropna(subset=["note"]).copy()
df["note"] = df["note"].astype(str).str.strip()
df = df[df["note"] != ""].reset_index(drop=True)

print("Records available for evaluation:", len(df))

# Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings in small batches...")

notes = df["note"].tolist()
embedding_batches = []

batch_size = 4

for start in range(0, len(notes), batch_size):
    batch_notes = notes[start:start + batch_size]

    batch_embeddings = model.encode(
        batch_notes,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embedding_batches.append(batch_embeddings)

    processed = min(start + batch_size, len(notes))

    if processed % 100 == 0 or processed == len(notes):
        print(f"Processed {processed}/{len(notes)} notes")

embeddings = np.vstack(embedding_batches)

print("Embeddings generated successfully!")
print("Embedding matrix shape:", embeddings.shape)



# Test queries
queries = [
    "Patient experiencing chest pain and shortness of breath",
    "Patient with abdominal pain, nausea, and vomiting",
    "Patient experiencing headache, dizziness, and confusion",
    "Patient with cough, fever, and difficulty breathing",
    "Patient experiencing joint pain and difficulty walking"
]

# Retrieve the top five results for each query
evaluation_results = []

for query in queries:
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:5]

    for rank, idx in enumerate(top_indices, start=1):
        evaluation_results.append({
            "query": query,
            "rank": rank,
            "record_index": int(idx),
            "patient_id": df.iloc[idx].get("patient_id", idx),
            "similarity_score": float(scores[idx]),
            "note": df.iloc[idx]["note"],
            "relevant": ""
        })

# Save results for human relevance judgments
results_df = pd.DataFrame(evaluation_results)

output_path = os.path.join(
    project_path, "data", "evaluation_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\nEvaluation results generated!")
print("Saved to:", output_path)
print("Queries tested:", len(queries))
print("Retrieved results:", len(results_df))
print(
    "\nOpen evaluation_results.csv and review the notes "
    "for each query to judge their relevance."
)

