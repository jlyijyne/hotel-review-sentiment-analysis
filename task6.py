import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Read dataset
df = pd.read_csv("D1.csv")

# Select positive and negative subclasses
reviews = df[
    df["subclass"].isin(["Positive", "Negative"])
    & df["text"].fillna("").str.strip().ne("")
].copy()

stopwords = list(ENGLISH_STOP_WORDS) + [
    "hotel", "hotels",
    "room", "rooms",
    "stay", "stayed",
    "just", "like"
]

# Convert reviews into word counts
vectorizer = CountVectorizer(
    stop_words=list(stopwords),
    max_features=3000,
    min_df=2
)

X = vectorizer.fit_transform(reviews["text"])

# Train LDA with five topics
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)

topic_scores = lda.fit_transform(X)

# Show five most important words for each topic
words = vectorizer.get_feature_names_out()
topic_words = []

for i, topic in enumerate(lda.components_):
    top_words = words[topic.argsort()[-5:][::-1]].tolist()
    topic_words.append(top_words)
    print(f"Topic {i + 1}: {', '.join(top_words)}")

# Save topics
pd.DataFrame({
    "topic": range(1, 6),
    "words": [", ".join(w) for w in topic_words]
}).to_csv("lda_topics.csv", index=False)

# Add topic probabilities to D1
columns = [f"lda_topic_{i}" for i in range(1, 6)]

df[columns] = pd.DataFrame(
    topic_scores,
    index=reviews.index,
    columns=columns
)

df["lda_main_topic"] = pd.Series(
    topic_scores.argmax(axis=1) + 1,
    index=reviews.index
)

df.to_csv("D1_task6.csv", index=False)

# Average topic distribution for both subclasses
distribution = (
    df.loc[reviews.index]
    .groupby("subclass")[columns]
    .mean()
)

print("\nTopic distribution:")
print(distribution.round(3))

distribution.to_csv("lda_subclass_distribution.csv")

# Plot topic distribution
distribution.T.plot(kind="bar", figsize=(9, 5))
plt.xlabel("Topic")
plt.ylabel("Average topic proportion")
plt.title("LDA Topic Distribution by Subclass")
plt.xticks(range(5), [f"Topic {i}" for i in range(1, 6)],
           rotation=0)
plt.tight_layout()
plt.savefig("lda_distribution.png")
plt.show()


# Compare all LDA topics with WordCloud top 20 words
for subclass in ["Positive", "Negative"]:

    text = " ".join(
        reviews[reviews["subclass"] == subclass]["text"]
    )

    wc = WordCloud(
        stopwords=set(ENGLISH_STOP_WORDS),
        max_words=200,
        collocations=False,
        normalize_plurals=False
    ).generate(text)

    wc_words = set(wc.words_)

    # Use words from ALL five LDA topics
    lda_words = set()

    for topic in topic_words:
        lda_words.update(topic)

    overlap = lda_words & wc_words

    print(f"\n{subclass} subclass:")
    print("Top WordCloud words:", ", ".join(wc.words_.keys()))
    print("All LDA words:", ", ".join(sorted(lda_words)))
    print("Overlapping words:", ", ".join(sorted(overlap)))

    print(
        "Overlap with WordCloud:",
        f"{len(overlap)}/{len(wc_words)}",
        f"({len(overlap)/len(wc_words)*100:.1f}%)"
    )
