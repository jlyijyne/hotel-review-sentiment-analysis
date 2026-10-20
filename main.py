
import pandas as pd
import subprocess

# TASK 1: Sentiment Analysis

# Read hotel reviews
df = pd.read_csv(
    "london_hotelrec_datasetti.csv",
    sep=";",
    encoding="utf-8-sig"
)

df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print("Total reviews:", len(df))

positive_scores = []
negative_scores = []
overall_scores = []

# Analyze each review
for i, review in enumerate(df["text"].fillna("")):

    if not review.strip():
        positive_scores.append(float("nan"))
        negative_scores.append(float("nan"))
        overall_scores.append(float("nan"))
        continue

    result = subprocess.run(
        [
            "java", "-jar",
            "sentistrength/SentiStrength.jar",
            "sentidata", "sentistrength/data/",
            "text", review
        ],
        capture_output=True,
        text=True,
        check=True
    )

    positive, negative = map(
        int, result.stdout.strip().split()[:2]
    )

    positive_scores.append(positive)
    negative_scores.append(negative)
    overall_scores.append(positive + negative)

    # Show progress
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(df)} reviews")

# Add sentiment scores
df["positive"] = positive_scores
df["negative"] = negative_scores
df["overall"] = overall_scores

# Save database D1
df.to_csv("D1.csv", index=False)

print("Sentiment analysis completed!")
print("Results saved to D1.csv")

