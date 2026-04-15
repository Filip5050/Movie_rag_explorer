"""
Run this once before launching the Streamlit app to populate the vector database.

    uv run python ingest.py

This will:
1. Drop the old collection (if any)
2. Fetch ~4,000 titles from 8 TMDB list endpoints
3. Enrich each with cast, crew, keywords, trailer, streaming providers (parallel API calls)
4. Embed every title and store it in ChromaDB

Expect ~3-5 minutes total depending on network speed.
"""

from vector import (
    DISCOVER_ENDPOINTS,
    MOVIE_ENDPOINTS,
    PAGES,
    TV_ENDPOINTS,
    get_collection,
    ingest_movies,
    reset_collection,
)

if __name__ == "__main__":
    print("Resetting old collection for a clean rebuild...")
    reset_collection()

    total_endpoints = len(MOVIE_ENDPOINTS) + len(TV_ENDPOINTS) + len(DISCOVER_ENDPOINTS)
    print(
        f"Fetching from {total_endpoints} TMDB endpoints "
        f"({len(MOVIE_ENDPOINTS)} movie + {len(TV_ENDPOINTS)} TV "
        f"+ {len(DISCOVER_ENDPOINTS)} discover), {PAGES} pages each..."
    )
    print("Embeddings will be computed once all titles are enriched.\n")

    count = ingest_movies()

    total_in_db = get_collection().count()
    print(f"\nDone! Ingested {count} titles this run.")
    print(f"Total titles in vector DB: {total_in_db}")
    print("\nYou can now launch the app:")
    print("  uv run streamlit run app.py")
