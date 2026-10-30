import pandas as pd
from empath import Empath

# Read results
empath = pd.read_csv("empath_results.csv")
lda = pd.read_csv("lda_topics.csv")
distribution = pd.read_csv("lda_subclass_distribution.csv")

lexicon = Empath()

results = []

for subclass in ["Positive", "Negative"]:

    print("\nAnalyzing:", subclass)

    # Get Empath categories with non-zero scores
    categories = set(
        empath[
            (empath["subclass"] == subclass) &
            (empath["score"] > 0)
        ]["category"]
    )

    # Select two most common LDA topics
    row = distribution[
        distribution["subclass"] == subclass
    ].iloc[0]

    # Convert topic scores to numeric values
    topic_scores = pd.to_numeric(row.drop("subclass"))

    # Select the two highest scoring topics
    top_topics = topic_scores.nlargest(2).index

    print("Selected LDA topics:", list(top_topics))

    # Collect the words from these topics
    lda_words = set()

    for topic in top_topics:
        topic_number = int(topic.split("_")[-1])

        words = lda[
            lda["topic"] == topic_number
        ]["words"].iloc[0]

        lda_words.update(
            word.strip().lower()
            for word in words.split(",")
        )

    # Compare LDA words with Empath category vocabularies
    matching = []

    for category in categories:
        category_words = set(lexicon.cats[category])

        if lda_words.intersection(category_words):
            matching.append(category)

    # Calculate agreement
    ratio = len(matching) / len(categories) if categories else 0

    print("LDA words:", lda_words)
    print("Empath categories:", len(categories))
    print("Matching categories:", len(matching))
    print("Matching category names:", sorted(matching))
    print("Agreement ratio:", ratio)
    print("Agreement percentage:", ratio * 100)

    results.append({
        "subclass": subclass,
        "empath_categories": len(categories),
        "matching_categories": len(matching),
        "agreement_ratio": ratio
    })

# Save results
pd.DataFrame(results).to_csv(
    "task9_agreement.csv", index=False
)

print("\nResults saved to task9_agreement.csv")
