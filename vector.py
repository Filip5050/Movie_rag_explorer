import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from concurrent.futures import ThreadPoolExecutor, as_completed

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
ENRICH_WORKERS = 12
TOP_CAST = 6
TOP_KEYWORDS = 15

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
# Broader catalog: most-voted and most-popular across all time, not just "currently trending".
# Fills gaps like Dallas Buyers Club, Goodfellas, Heat, etc. that miss the trending lists.
DISCOVER_ENDPOINTS = [
    ("/discover/movie", "movie", {"sort_by": "vote_count.desc"}),
    ("/discover/movie", "movie", {"sort_by": "popularity.desc"}),
    ("/discover/tv", "tv", {"sort_by": "vote_count.desc"}),
    ("/discover/tv", "tv", {"sort_by": "popularity.desc"}),
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
            "tmdb_id": raw["id"],
            "media": "tv",
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
        "tmdb_id": raw["id"],
        "media": "movie",
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

    sources: list[tuple[str, str, dict]] = (
        [(ep, "movie", {}) for ep in MOVIE_ENDPOINTS]
        + [(ep, "tv", {}) for ep in TV_ENDPOINTS]
        + list(DISCOVER_ENDPOINTS)
    )

    for endpoint, media_type, extra_params in sources:
        for page in range(1, pages + 1):
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page,
                **extra_params,
            }
            resp = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            for raw in resp.json().get("results", []):
                item = _normalize(raw, media_type)
                if item["id"] in seen:
                    continue
                if not item["overview"]:
                    continue
                if item["vote_count"] < 30:
                    continue
                seen.add(item["id"])
                items.append(item)

    return items


def _fetch_details(item: dict) -> dict:
    """
    Fetch credits, keywords, videos, and watch providers for a single title in one
    API call using TMDB's append_to_response. Mutates and returns the item.
    """
    item["cast"] = []
    item["directors"] = []
    item["keywords"] = []
    item["trailer_key"] = ""
    item["providers"] = []

    try:
        resp = requests.get(
            f"{TMDB_BASE}/{item['media']}/{item['tmdb_id']}",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "append_to_response": "credits,keywords,videos,watch/providers",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return item

    credits = data.get("credits", {}) or {}
    item["cast"] = [c["name"] for c in credits.get("cast", [])[:TOP_CAST] if c.get("name")]

    if item["media"] == "movie":
        crew = credits.get("crew", []) or []
        item["directors"] = [c["name"] for c in crew if c.get("job") == "Director" and c.get("name")]
    else:
        item["directors"] = [c["name"] for c in data.get("created_by", []) or [] if c.get("name")]

    kw_block = data.get("keywords", {}) or {}
    kw_list = kw_block.get("keywords") or kw_block.get("results") or []
    item["keywords"] = [k["name"] for k in kw_list if k.get("name")]

    videos = (data.get("videos", {}) or {}).get("results", []) or []
    trailer = None
    for v in videos:
        if v.get("site") != "YouTube":
            continue
        if v.get("type") != "Trailer":
            continue
        if v.get("official") and not trailer:
            trailer = v
            break
        if trailer is None:
            trailer = v
    item["trailer_key"] = trailer.get("key", "") if trailer else ""

    providers_block = (data.get("watch/providers", {}) or {}).get("results", {}) or {}
    us = providers_block.get("US", {}) or {}
    item["providers"] = [p["provider_name"] for p in us.get("flatrate", []) or [] if p.get("provider_name")]

    return item


def enrich_items(items: list[dict], max_workers: int = ENRICH_WORKERS) -> list[dict]:
    """Enrich every item with cast/crew/keywords/trailer/providers in parallel."""
    total = len(items)
    enriched: list[dict] = []
    print(f"Enriching {total} titles with cast, crew, keywords, trailers, and providers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_details, item) for item in items]
        for i, future in enumerate(as_completed(futures), 1):
            enriched.append(future.result())
            if i % 200 == 0 or i == total:
                print(f"  {i}/{total}")

    return enriched


def build_document(item: dict) -> str:
    """Build the text string that will be embedded. Richer = better retrieval."""
    year = item.get("release_date", "")[:4] or "Unknown"
    genres = ", ".join(
        GENRE_MAP[gid] for gid in item.get("genre_ids", []) if gid in GENRE_MAP
    ) or "Unknown"

    parts = [
        f"Title: {item['title']}",
        f"Type: {item['media_type']}",
        f"Year: {year}",
        f"Genres: {genres}",
        f"Rating: {item['vote_average']}/10",
    ]

    directors = item.get("directors", [])
    if directors:
        label = "Director" if item["media_type"] == "Movie" else "Created by"
        parts.append(f"{label}: {', '.join(directors)}")

    cast = item.get("cast", [])
    if cast:
        parts.append(f"Cast: {', '.join(cast)}")

    keywords = item.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords[:TOP_KEYWORDS])}")

    providers = item.get("providers", [])
    if providers:
        parts.append(f"Streaming on: {', '.join(providers)}")

    parts.append(f"Overview: {item['overview']}")

    return "\n".join(parts)


def get_collection() -> chromadb.Collection:
    """Return (creating if needed) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Delete the existing collection so we can rebuild it from scratch."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name=COLLECTION)
    except Exception:
        pass


def ingest_movies(pages: int = PAGES) -> int:
    """Fetch, enrich, embed, and upsert into ChromaDB. Returns count upserted."""
    items = fetch_all(pages)
    print(f"Fetched {len(items)} unique titles.")
    items = enrich_items(items)

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
            "cast": ", ".join(item.get("cast", [])[:3]),
            "directors": ", ".join(item.get("directors", [])),
            "trailer_key": item.get("trailer_key", ""),
            "providers": ", ".join(item.get("providers", [])),
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
