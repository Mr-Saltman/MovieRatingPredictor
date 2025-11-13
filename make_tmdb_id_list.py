import pandas as pd

links = pd.read_csv("dataset/links.csv")
ratings = pd.read_csv("dataset/ratings.csv")
df = (ratings.groupby("movieId").size().rename("n").reset_index()
            .merge(links[["movieId","tmdbId"]], on="movieId", how="left"))
df = df.dropna(subset=["tmdbId"])
df["tmdbId"] = df["tmdbId"].astype(int)

top = df.sort_values("n", ascending=False).head(5000)
top["tmdbId"].to_csv("tmdb_ids_to_scrape.txt", index=False, header=False)
print(f"wrote {len(top)} tmdb IDs to tmdb_ids_to_scrape.txt")