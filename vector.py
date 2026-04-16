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
TOP_CAST = 8
TOP_CREW = 3

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
DISCOVER_ENDPOINTS = [
    ("/discover/movie", "movie", {"sort_by": "vote_count.desc"}),
    ("/discover/movie", "movie", {"sort_by": "popularity.desc"}),
    ("/discover/tv", "tv", {"sort_by": "vote_count.desc"}),
    ("/discover/tv", "tv", {"sort_by": "popularity.desc"}),
]

GENRE_MAP = {
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
    """Normalize a raw TMDB list result into a consistent shape."""
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
            "original_language": raw.get("original_language", "en"),
            "popularity": raw.get("popularity", 0.0),
            "origin_country": raw.get("origin_country", []),
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
        "original_language": raw.get("original_language", "en"),
        "popularity": raw.get("popularity", 0.0),
        "origin_country": raw.get("origin_country", []),
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
    Fetch full details for a single title using TMDB's append_to_response.
    Extracts cast with roles, crew, keywords, trailers, all provider types,
    tagline, runtime, status, franchise, production companies, and TV-specific fields.
    """
    item["cast_credits"] = []   # list of {"name": str, "character": str}
    item["directors"] = []
    item["writers"] = []
    item["keywords"] = []
    item["trailer_key"] = ""
    item["providers_stream"] = []
    item["providers_buy"] = []
    item["providers_rent"] = []
    item["tagline"] = ""
    item["runtime"] = 0
    item["status"] = ""
    item["collection"] = ""
    item["production_companies"] = []
    item["networks"] = []
    item["number_of_seasons"] = 0
    item["number_of_episodes"] = 0
    item["budget"] = 0
    item["revenue"] = 0

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

    # Cast with character names
    credits = data.get("credits", {}) or {}
    cast_raw = credits.get("cast", []) or []
    item["cast_credits"] = [
        {"name": c["name"], "character": c.get("character", "")}
        for c in cast_raw[:TOP_CAST]
        if c.get("name")
    ]

    # Directors and writers
    crew = credits.get("crew", []) or []
    if item["media"] == "movie":
        item["directors"] = [
            c["name"] for c in crew if c.get("job") == "Director" and c.get("name")
        ]
        item["writers"] = [
            c["name"] for c in crew
            if c.get("job") in ("Screenplay", "Writer", "Story") and c.get("name")
        ][:TOP_CREW]
    else:
        item["directors"] = [
            c["name"] for c in (data.get("created_by", []) or []) if c.get("name")
        ]

    # Keywords (all of them — more = better retrieval)
    kw_block = data.get("keywords", {}) or {}
    kw_list = kw_block.get("keywords") or kw_block.get("results") or []
    item["keywords"] = [k["name"] for k in kw_list if k.get("name")]

    # Official trailer
    videos = (data.get("videos", {}) or {}).get("results", []) or []
    trailer = None
    for v in videos:
        if v.get("site") != "YouTube" or v.get("type") != "Trailer":
            continue
        if v.get("official") and not trailer:
            trailer = v
            break
        if trailer is None:
            trailer = v
    item["trailer_key"] = trailer.get("key", "") if trailer else ""

    # All provider types (streaming, buy, rent) — US market
    providers_block = (data.get("watch/providers", {}) or {}).get("results", {}) or {}
    us = providers_block.get("US", {}) or {}
    item["providers_stream"] = [
        p["provider_name"] for p in (us.get("flatrate", []) or []) if p.get("provider_name")
    ]
    item["providers_buy"] = [
        p["provider_name"] for p in (us.get("buy", []) or []) if p.get("provider_name")
    ]
    item["providers_rent"] = [
        p["provider_name"] for p in (us.get("rent", []) or []) if p.get("provider_name")
    ]

    # Core detail fields
    item["tagline"] = data.get("tagline", "") or ""
    item["status"] = data.get("status", "") or ""
    item["production_companies"] = [
        c["name"] for c in (data.get("production_companies", []) or []) if c.get("name")
    ]
    # origin_country may already be set from list endpoint; prefer detail
    item["origin_country"] = data.get("origin_country", item.get("origin_country", [])) or []

    if item["media"] == "movie":
        item["runtime"] = data.get("runtime", 0) or 0
        item["budget"] = data.get("budget", 0) or 0
        item["revenue"] = data.get("revenue", 0) or 0
        col = data.get("belongs_to_collection")
        item["collection"] = col["name"] if col and col.get("name") else ""
    else:
        run_times = data.get("episode_run_time", []) or []
        item["runtime"] = run_times[0] if run_times else 0
        item["networks"] = [
            n["name"] for n in (data.get("networks", []) or []) if n.get("name")
        ]
        item["number_of_seasons"] = data.get("number_of_seasons", 0) or 0
        item["number_of_episodes"] = data.get("number_of_episodes", 0) or 0

    return item


def enrich_items(items: list[dict], max_workers: int = ENRICH_WORKERS) -> list[dict]:
    """Enrich every item with full details in parallel."""
    total = len(items)
    enriched: list[dict] = []
    print(f"Enriching {total} titles with full details...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_details, item) for item in items]
        for i, future in enumerate(as_completed(futures), 1):
            enriched.append(future.result())
            if i % 200 == 0 or i == total:
                print(f"  {i}/{total}")

    return enriched


def build_document(item: dict) -> str:
    """Build the text string that gets embedded. Every field improves retrieval."""
    year = item.get("release_date", "")[:4] or "Unknown"
    genres = ", ".join(
        GENRE_MAP[gid] for gid in item.get("genre_ids", []) if gid in GENRE_MAP
    ) or "Unknown"

    parts = [
        f"Title: {item['title']}",
        f"Type: {item['media_type']}",
        f"Year: {year}",
        f"Genres: {genres}",
        f"Rating: {item['vote_average']}/10 ({item.get('vote_count', 0):,} votes)",
    ]

    if item.get("tagline"):
        parts.append(f"Tagline: {item['tagline']}")

    if item.get("status"):
        parts.append(f"Status: {item['status']}")

    lang = item.get("original_language", "en")
    if lang and lang != "en":
        parts.append(f"Original language: {lang}")

    if item.get("origin_country"):
        parts.append(f"Country: {', '.join(item['origin_country'])}")

    runtime = item.get("runtime", 0)
    if runtime:
        label = "Runtime" if item["media"] == "movie" else "Episode runtime"
        parts.append(f"{label}: {runtime} min")

    if item["media"] == "movie":
        if item.get("collection"):
            parts.append(f"Part of franchise: {item['collection']}")
        budget = item.get("budget", 0)
        revenue = item.get("revenue", 0)
        if budget:
            parts.append(f"Budget: ${budget:,}")
        if revenue:
            parts.append(f"Box office: ${revenue:,}")
    else:
        seasons = item.get("number_of_seasons", 0)
        episodes = item.get("number_of_episodes", 0)
        if seasons:
            ep_str = f" ({episodes} episodes)" if episodes else ""
            parts.append(f"Seasons: {seasons}{ep_str}")
        if item.get("networks"):
            parts.append(f"Networks: {', '.join(item['networks'])}")

    directors = item.get("directors", [])
    if directors:
        label = "Director" if item["media_type"] == "Movie" else "Created by"
        parts.append(f"{label}: {', '.join(directors)}")

    writers = item.get("writers", [])
    if writers:
        parts.append(f"Writer(s): {', '.join(writers)}")

    cast_credits = item.get("cast_credits", [])
    if cast_credits:
        cast_parts = [
            f"{c['name']} as {c['character']}" if c.get("character") else c["name"]
            for c in cast_credits
        ]
        parts.append(f"Cast: {', '.join(cast_parts)}")

    if item.get("production_companies"):
        parts.append(f"Production: {', '.join(item['production_companies'][:5])}")

    keywords = item.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    if item.get("providers_stream"):
        parts.append(f"Streaming on: {', '.join(item['providers_stream'])}")
    if item.get("providers_buy"):
        parts.append(f"Available to buy: {', '.join(item['providers_buy'])}")
    if item.get("providers_rent"):
        parts.append(f"Available to rent: {', '.join(item['providers_rent'])}")

    parts.append(f"Overview: {item['overview']}")

    return "\n".join(parts)


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
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
            "vote_count": item.get("vote_count", 0),
            "poster_path": item.get("poster_path", ""),
            # Cast: plain names for display, top 3
            "cast": ", ".join(c["name"] for c in item.get("cast_credits", [])[:3]),
            "directors": ", ".join(item.get("directors", [])),
            "writers": ", ".join(item.get("writers", [])),
            "trailer_key": item.get("trailer_key", ""),
            "providers": ", ".join(item.get("providers_stream", [])),
            "providers_buy": ", ".join(item.get("providers_buy", [])),
            "providers_rent": ", ".join(item.get("providers_rent", [])),
            "tagline": item.get("tagline", ""),
            "runtime": item.get("runtime", 0),
            "status": item.get("status", ""),
            "collection": item.get("collection", ""),
            "original_language": item.get("original_language", ""),
            "origin_country": ", ".join(item.get("origin_country", [])),
            "production_companies": ", ".join(item.get("production_companies", [])[:4]),
            "networks": ", ".join(item.get("networks", [])),
            "number_of_seasons": item.get("number_of_seasons", 0),
            "number_of_episodes": item.get("number_of_episodes", 0),
            "budget": item.get("budget", 0),
            "revenue": item.get("revenue", 0),
        }
        for item in items
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def query_movies(query_text: str, n_results: int = 5) -> list[dict]:
    """Semantic search the ChromaDB collection."""
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
