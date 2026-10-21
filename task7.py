import pandas as pd
from empath import Empath

df = pd.read_csv("D1.csv")

lexicon = Empath()

results = []

for subclass in ["Positive", "Negative"]:

    reviews = df[df["subclass"] == subclass]

    text = " ".join(reviews["text"].dropna().astype(str))

    if not text.strip():
        print("No reviews found for:", subclass)
        continue

    scores = lexicon.analyze(text, normalize=True)

    for category, score in scores.items():
        if score > 0:
            results.append({
                "subclass": subclass,
                "category": category,
                "score": score
            })

results_df = pd.DataFrame(results)

results_df.to_csv("empath_results.csv", index=False)

print(results_df)
print("Results saved to empath_results.csv")