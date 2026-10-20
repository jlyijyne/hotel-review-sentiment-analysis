
import pandas as pd
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("D1_with_stars.csv")
df.columns = df.columns.str.strip().str.strip("*")

df["hotel_stars"] = pd.to_numeric(
    df["hotel_stars"], errors="coerce"
)

# Calculate statistics for each hotel
hotels = df.groupby("hotel_id").agg(
    average=("rating", "mean"),
    std=("rating", "std"),
    kurtosis=("rating", "kurt"),
    reviews=("rating", "count"),
    stars=("hotel_stars", "first")
)

# Show hotels with lowest and highest standard deviation
print("Lowest standard deviation:")
print(hotels.nsmallest(5, "std"))

print("\nHighest standard deviation:")
print(hotels.nlargest(5, "std"))

print("\nStandard deviation statistics:")
print(hotels["std"].describe())

# Define Ambiguous Class threshold
epsilon = hotels["std"].quantile(0.75)
hotels["ambiguous"] = hotels["std"] > epsilon

print("\nThreshold epsilon:", epsilon)
print("Ambiguous hotels:", hotels["ambiguous"].sum())

# Compare standard deviation between hotel star categories
print("\nAverage standard deviation by hotel stars:")
print(hotels.groupby("stars")["std"].mean())

# Percentage of ambiguous hotels in each star category
proportions = (
    hotels.groupby("stars")["ambiguous"].mean() * 100
)

print("\nAmbiguous hotel percentages:")
print(proportions)

# Plot histogram
proportions.plot(kind="bar")

plt.xlabel("Hotel stars")
plt.ylabel("Ambiguous hotels (%)")
plt.title("Proportion of Ambiguous Hotels by Star Category")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# Save hotel statistics
hotels.to_csv("hotel_statistics.csv")
