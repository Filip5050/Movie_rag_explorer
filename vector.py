import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import chromadb
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
CHROMA_PATH = "chroma_db"
COLLECTION = "movies"
EMBED_MODEL = "all-MiniLM-L6-v2"
PAGES = 50
MOVIE_ENDPOINTS = [
    "/movie/popular",
    "/movie/top_rated",
    "/movie/now_playing",
    "/movie/upcoming",
]
TV_ENDPOINTS = [
    "/tv/popular",
    "/tv/top_rated",
    "/tv/on_the_air",
    "/tv/airing_today",
]

GENRE_MAP = {
    # Movie genres
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
    # TV-specific genres
    10759: "Action & Adventure",
    10762: "Kids",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
}


def _normalize(raw: dict, media_type: str) -> dict:
    """Normalize a raw TMDB result into a consistent shape regardless of movie vs TV."""
    if media_type == "tv":
        return {
            "id": f"tv_{raw['id']}",
            "title": raw.get("name", "Unknown"),
            "overview": raw.get("overview", ""),
            "release_date": raw.get("first_air_date", ""),
            "vote_average": raw.get("vote_average", 0),
            "vote_count": raw.get("vote_count", 0),
            "genre_ids": raw.get("genre_ids", []),
            "media_type": "TV Series",
            "poster_path": raw.get("poster_path", ""),
        }
    return {
        "id": f"movie_{raw['id']}",
        "title": raw.get("title", "Unknown"),
        "overview": raw.get("overview", ""),
        "release_date": raw.get("release_date", ""),
        "vote_average": raw.get("vote_average", 0),
        "vote_count": raw.get("vote_count", 0),
        "genre_ids": raw.get("genre_ids", []),
        "media_type": "Movie",
        "poster_path": raw.get("poster_path", ""),
    }


def fetch_all(pages: int = PAGES) -> list[dict]:
    """Fetch movies and TV shows from TMDB, deduplicated by prefixed ID."""
    seen: set[str] = set()
    items: list[dict] = []

    sources = (
        [(ep, "movie") for ep in MOVIE_ENDPOINTS]
        + [(ep, "tv") for ep in TV_ENDPOINTS]
    )

    for endpoint, media_type in sources:
        for page in range(1, pages + 1):
            resp = requests.get(
                f"{TMDB_BASE}{endpoint}",
                params={"api_key": TMDB_API_KEY, "language": "en-US", "page": page},
                timeout=10,
            )
            resp.raise_for_status()
            for raw in resp.json().get("results", []):
                item = _normalize(raw, media_type)
                if item["id"] in seen:
                    continue
                if not item["overview"]:
                    continue
                if item["vote_count"] < 50:
                    continue
                seen.add(item["id"])
                items.append(item)

    return items


def build_document(item: dict) -> str:
    """Build the text string that will be embedded."""
    year = item.get("release_date", "")[:4] or "Unknown"
    genres = ", ".join(
        GENRE_MAP[gid] for gid in item.get("genre_ids", []) if gid in GENRE_MAP
    ) or "Unknown"
    return (
        f"Title: {item['title']}\n"
        f"Type: {item['media_type']}\n"
        f"Year: {year}\n"
        f"Genres: {genres}\n"
        f"Rating: {item['vote_average']}/10\n"
        f"Overview: {item['overview']}"
    )


def get_collection() -> chromadb.Collection:
    """Return (creating if needed) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_movies(pages: int = PAGES) -> int:
    """Fetch movies + TV shows, embed, and upsert into ChromaDB. Returns count upserted."""
    items = fetch_all(pages)
    collection = get_collection()

    ids = [item["id"] for item in items]
    documents = [build_document(item) for item in items]
    metadatas = [
        {
            "title": item["title"],
            "type": item["media_type"],
            "year": item.get("release_date", "")[:4] or "Unknown",
            "genres": ", ".join(
                GENRE_MAP[gid] for gid in item.get("genre_ids", []) if gid in GENRE_MAP
            ) or "Unknown",
            "rating": item["vote_average"],
            "poster_path": item.get("poster_path", ""),
        }
        for item in items
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def query_movies(query_text: str, n_results: int = 5) -> list[dict]:
    """Semantic search the ChromaDB collection. Returns list of {document, metadata, distance}."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]
    return [
        {
            "document": documents[0][i],
            "metadata": metadatas[0][i],
            "distance": distances[0][i],
        }
        for i in range(len(documents[0]))
    ]
