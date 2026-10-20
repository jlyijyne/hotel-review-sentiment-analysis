import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Read data
df = pd.read_csv("D1_task4.csv")

# Keep only positive and negative subclasses
positive_reviews = df[df["subclass"] == "Positive"]["text"].fillna("")
negative_reviews = df[df["subclass"] == "Negative"]["text"].fillna("")

# Concatenate all reviews into one long text
positive_text = " ".join(positive_reviews.astype(str))
negative_text = " ".join(negative_reviews.astype(str))

# Stopwords
stopwords = set(STOPWORDS)
extra_stopwords = {
    "hotel", "room", "stay", "stayed", "one", "would",
    "could", "also", "really", "us", "get", "got"
}
stopwords.update(extra_stopwords)

# Generate word clouds
positive_wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=stopwords
).generate(positive_text)

negative_wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=stopwords
).generate(negative_text)

# Plot positive word cloud
plt.figure(figsize=(10, 5))
plt.imshow(positive_wc, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Positive Subclass")
plt.tight_layout()
plt.savefig("wordcloud_positive.png")
plt.show()

# Plot negative word cloud
plt.figure(figsize=(10, 5))
plt.imshow(negative_wc, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Negative Subclass")
plt.tight_layout()
plt.savefig("wordcloud_negative.png")
plt.show()