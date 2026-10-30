import pandas as pd
import matplotlib.pyplot as plt

# Read datasets
df = pd.read_csv("D1.csv")
hotels = pd.read_csv("hotel_statistics.csv")

# Classify reviews based on user rating
def classify(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    return "Neutral"

df["rating_class"] = df["rating"].apply(classify)

# Count positive and negative reviews for each hotel
counts = pd.crosstab(df["hotel_id"], df["rating_class"])

counts = counts.reindex(
    index=hotels["hotel_id"],
    columns=["Positive", "Negative"],
    fill_value=0
)

hotels = hotels.set_index("hotel_id")
hotels["ambiguous"] = (
    hotels["ambiguous"].astype(str).str.lower() == "true"
)

# Determine subclass for ambiguous hotels
hotels["subclass"] = "Not Ambiguous"

mask = hotels["ambiguous"]

hotels.loc[
    mask & (counts["Positive"] > counts["Negative"]),
    "subclass"
] = "Positive"

hotels.loc[
    mask & (counts["Negative"] > counts["Positive"]),
    "subclass"
] = "Negative"

hotels.loc[
    mask & (counts["Positive"] == counts["Negative"]),
    "subclass"
] = "Undetermined"

# Add attributes to D1
df["ambiguous"] = df["hotel_id"].map(hotels["ambiguous"])
df["subclass"] = df["hotel_id"].map(hotels["subclass"])

df.to_csv("D1_task4.csv", index=False)
hotels.to_csv("hotel_statistics_task4.csv")

# Calculate subclass proportions
subclasses = hotels[hotels["ambiguous"]]["subclass"]
proportions = (
    subclasses.value_counts(normalize=True) * 100
).reindex(["Positive", "Negative"], fill_value=0)

print("Ambiguous hotels:", len(subclasses))
print("\nSubclass proportions:")
print(proportions)
print("Undetermined:", (subclasses == "Undetermined").sum())

# Plot histogram
proportions.plot(kind="bar", color=["green", "red"])

plt.ylabel("Percentage of Ambiguous Hotels (%)")
plt.xlabel("Subclass")
plt.title("Positive and Negative Ambiguous Hotels")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("ambiguous_subclasses.png")
plt.show()
