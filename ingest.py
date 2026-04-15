"""
Run this once before launching the Streamlit app to populate the vector database.

    uv run python ingest.py
"""

from vector import ingest_movies, get_collection, PAGES, MOVIE_ENDPOINTS, TV_ENDPOINTS

if __name__ == "__main__":
    total_endpoints = len(MOVIE_ENDPOINTS) + len(TV_ENDPOINTS)
    print(f"Fetching from {total_endpoints} TMDB endpoints ({len(MOVIE_ENDPOINTS)} movie + {len(TV_ENDPOINTS)} TV), {PAGES} pages each...")
    print("This may take a minute while embeddings are computed.\n")

    count = ingest_movies()

    total_in_db = get_collection().count()
    print(f"\nDone! Ingested {count} movies this run.")
    print(f"Total movies in vector DB: {total_in_db}")
    print("\nYou can now launch the app:")
    print("  uv run streamlit run app.py")
