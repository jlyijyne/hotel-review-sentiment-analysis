import pandas as pd

df = pd.read_csv("D1.csv")

correlation = df["overall"].corr(df["rating"], method="pearson")

print("Pearson correlation:", correlation)