import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("movies_metadata_cleaned (2).csv")

# -------------------------
# 1. Remove duplicates
# -------------------------
df = df.drop_duplicates(subset="id", keep="first")

# -------------------------
# 2. Clean dates
# -------------------------
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

df.loc[
    (df["release_year"] < 1888) | (df["release_year"] > 2026),
    "release_year"
] = np.nan

# -------------------------
# 3. Clean text
# -------------------------
text_columns = [
    "title", "original_title", "status",
    "tagline", "overview", "genres",
    "collection_name", "original_language",
    "homepage", "imdb_id"
]

for col in text_columns:
    df[col] = df[col].str.strip()

df["original_language"] = df["original_language"].str.lower()
df["status"] = df["status"].str.strip().str.title()

# -------------------------
# 4. Handle missing text
# -------------------------
df["overview"] = df["overview"].fillna("No overview")
df["tagline"] = df["tagline"].fillna("No tagline")
df["status"] = df["status"].fillna("Unknown")
df["original_language"] = df["original_language"].fillna("Unknown")

# -------------------------
# 5. Clean numerical data
# -------------------------
df.loc[df["budget"] < 0, "budget"] = np.nan
df.loc[df["revenue"] < 0, "revenue"] = np.nan
df.loc[df["vote_count"] < 0, "vote_count"] = np.nan

df.loc[
    (df["runtime"] <= 0) | (df["runtime"] > 600),
    "runtime"
] = np.nan

df.loc[
    (df["vote_average"] < 0) |
    (df["vote_average"] > 10),
    "vote_average"
] = np.nan

# -------------------------
# 6. Derived columns (Profit and ROI)
# -------------------------
df["profit"] = np.where(
    df["budget"].notna() & df["revenue"].notna(),
    df["revenue"] - df["budget"],
    np.nan
)

df["roi"] = np.where(
    df["budget"] > 0,
    df["revenue"] / df["budget"],
    np.nan
)

# -------------------------
# 7. Remove impossible values
# -------------------------
numeric_positive = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_count"
]

for col in numeric_positive:
    df.loc[df[col] < 0, col] = np.nan

df.loc[
    (df["runtime"] <= 0) | (df["runtime"] > 600),
    "runtime"
] = np.nan

# -------------------------
# 8. Remove unusable rows
# -------------------------
df = df.dropna(subset=["title", "release_date"])

# -------------------------
# 9. Fix data types
# -------------------------
df["release_year"] = df["release_year"].astype("Int64")
df["vote_count"] = df["vote_count"].astype("Int64")
df["budget"] = df["budget"].astype("Float64")
df["revenue"] = df["revenue"].astype("Float64")

# -------------------------
# 10. Save
# -------------------------
df.to_csv("movies_metadata_final.csv", index=False)

print("Final shape:", df.shape)
print(df.isnull().sum())