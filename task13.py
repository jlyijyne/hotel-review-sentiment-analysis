
import pandas as pd

# Read dataset
df = pd.read_csv("D1.csv")

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

# Case A: High rating but very negative sentiment
case_a = df[
    (df["rating"] >= 4) &
    (df["overall"] <= -2)
].copy()

case_a["bias_type"] = "High rating, negative sentiment"

# Case B: Low rating but very positive sentiment
case_b = df[
    (df["rating"] <= 2) &
    (df["overall"] >= 2)
].copy()

case_b["bias_type"] = "Low rating, positive sentiment"

# Combine cases
bias_cases = pd.concat([case_a, case_b])

# Save results
bias_cases.to_csv("task13_bias_cases.csv", index=False)

# Print statistics
print("Total reviews:", len(df))
print("Case A:", len(case_a))
print("Case B:", len(case_b))
print("Total conflicting reviews:", len(bias_cases))

print("\nCase A examples:")
print(case_a[["rating", "overall", "text"]].head(5).to_string(index=False))

print("\nCase B examples:")
print(case_b[["rating", "overall", "text"]].head(5).to_string(index=False))
