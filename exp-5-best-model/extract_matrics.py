import mlflow
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/panchariyarohit486/youtube-sentiment-analysis.mlflow")
client = mlflow.tracking.MlflowClient()

experiment_id = "4"
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"]
)

rows = []
for run in runs:
    m = run.data.metrics
    p = run.data.params
    rows.append({
        "algo_name": p.get("algo_name"),
        "accuracy": m.get("accuracy"),
        "macro_avg_f1": m.get("macro avg_f1-score"),
        "macro_avg_precision": m.get("macro avg_precision"),
        "macro_avg_recall": m.get("macro avg_recall"),
        "weighted_avg_f1": m.get("weighted avg_f1-score"),
        "neutral_f1": m.get("neutral_f1-score"),
        "neutral_precision": m.get("neutral_precision"),
        "neutral_recall": m.get("neutral_recall"),
        "negative_f1": m.get("negative_f1-score"),
        "positive_f1": m.get("positive_f1-score"),
    })

df_results = pd.DataFrame(rows).dropna(subset=["algo_name"])

# Combined score: weight accuracy and macro F1
# macro F1 matters more than accuracy for imbalanced classes
df_results["combined_score"] = (
    0.4 * df_results["accuracy"] + 0.6 * df_results["macro_avg_f1"]
)

df_results = df_results.sort_values("combined_score", ascending=False)

# Save results to a new output file
df_results.to_csv("experiment_4_model_ranking.csv", index=False)
with open("experiment_4_model_ranking.txt", "w") as f:
    f.write(df_results.to_string(index=False))

print(df_results.to_string(index=False))
print("\nSaved to experiment_4_model_ranking.csv and experiment_4_model_ranking.txt")