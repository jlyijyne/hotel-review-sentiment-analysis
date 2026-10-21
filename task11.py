import pandas as pd
import matplotlib.pyplot as plt
import re
from task10 import ontology

# Read data
df = pd.read_csv("D1.csv")

results = []

# Analyze both subclasses
for subclass in ["Positive", "Negative"]:

    reviews = df[df["subclass"] == subclass]
    text = " ".join(reviews["text"].fillna("").astype(str)).lower()

    # Find words in each category
    for category, words in ontology.items():

        count = 0

        for word in words:
            count += len(re.findall(r"\b" + re.escape(word) + r"\b", text))

        results.append({
            "Subclass": subclass,
            "Category": category,
            "Count": count
        })

# Create results table
results_df = pd.DataFrame(results)

print(results_df)

# Create histogram
chart = results_df.pivot(
    index="Category",
    columns="Subclass",
    values="Count"
)

chart.plot(kind="bar", figsize=(10, 5))

plt.title("Ontology Category Frequencies")
plt.xlabel("Category")
plt.ylabel("Word frequency")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("ontology_histogram.png")
plt.show()