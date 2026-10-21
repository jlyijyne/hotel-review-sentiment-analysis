import pandas as pd
from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv("D1.csv")
empath = pd.read_csv("empath_results.csv")

stopwords = set(STOPWORDS)

extra_stopwords = {
    "hotel", "room", "stay", "stayed", "one", "would",
    "could", "also", "really", "us", "get", "got"
}

stopwords.update(extra_stopwords)

# Analyze both subclasses
for subclass in ["Positive", "Negative"]:

    print("\nAnalyzing:", subclass)

    # Select reviews
    reviews = df[df["subclass"] == subclass]
    text = " ".join(reviews["text"].fillna("").astype(str))

    # Generate WordCloud with Task 5 settings
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords
    ).generate(text)

    # Get WordCloud words
    words = set(cloud.words_.keys())

    # Get Empath categories
    categories = set(
        empath[
            (empath["subclass"] == subclass) &
            (empath["score"] > 0)
        ]["category"].str.lower()
    )

    # Calculate overlap
    overlap = words.intersection(categories)

    ratio = len(overlap) / len(categories)

    # Print results
    print("Empath categories:", len(categories))
    print("Overlapping categories:", len(overlap))
    print("Matching words:", overlap)
    print("Overlap ratio:", ratio)
    print("Overlap percentage:", ratio * 100)