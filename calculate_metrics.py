
import os
import pandas as pd

# Locate the evaluation results
project_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    project_path, "data", "evaluation_results.csv"
)

results = pd.read_csv(file_path)

# Validate relevance judgments
if results["relevant"].isna().any():
    print("Please label every result in the 'relevant' column first.")
    raise SystemExit

results["relevant"] = pd.to_numeric(
    results["relevant"], errors="coerce"
)

if results["relevant"].isna().any():
    print("Use only 1 or 0 in the 'relevant' column.")
    raise SystemExit

if not results["relevant"].isin([0, 1]).all():
    print("Use only 1 or 0 in the 'relevant' column.")
    raise SystemExit

# Calculate metrics for each query
query_metrics = []

for query, group in results.groupby("query", sort=False):
    group = group.sort_values("rank")
    relevance = group["relevant"].tolist()

    k = min(5, len(relevance))

    # Precision@5
    precision_at_5 = sum(relevance[:k]) / 5

    # Reciprocal rank of the first relevant result
    reciprocal_rank = 0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant == 1:
            reciprocal_rank = 1 / rank
            break

    query_metrics.append({
        "query": query,
        "Precision@5": precision_at_5,
        "Reciprocal Rank": reciprocal_rank
    })

metrics_df = pd.DataFrame(query_metrics)

print("\nMetrics for each query:")
print(metrics_df.to_string(index=False))

print("\nOverall Evaluation")
print("Mean Precision@5:", round(metrics_df["Precision@5"].mean(), 3))
print("Mean Reciprocal Rank (MRR):",
      round(metrics_df["Reciprocal Rank"].mean(), 3))

# Save the metrics
output_path = os.path.join(
    project_path, "data", "evaluation_metrics.csv"
)

metrics_df.to_csv(output_path, index=False)

print("\nMetrics saved to:", output_path)

