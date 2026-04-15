# Movie RAG Explorer

A local Retrieval-Augmented Generation (RAG) application that lets you search and get recommendations for movies and TV series using natural language. Movie data is fetched from the TMDB API, embedded into a local vector database, and answered by a locally running LLM via Ollama — all through a Streamlit chat interface.

---

## How it works

1. **Ingest** — Fetches ~3,800+ movies and TV series from TMDB across 8 endpoints (popular, top-rated, now playing, upcoming — for both movies and TV). Each title is embedded using `all-MiniLM-L6-v2` and stored in a local ChromaDB vector database.
2. **Query** — Your question is embedded and semantically matched against the database. The top 5 most relevant titles are retrieved.
3. **Generate** — The retrieved titles (with overviews, ratings, genres) are passed as context to `gemma4:latest` running locally via Ollama, which generates a conversational recommendation.

---

## Stack

| Component | Tool |
|---|---|
| Movie/TV data | [TMDB API](https://www.themoviedb.org/documentation/api) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| LLM | `gemma4:latest` via [Ollama](https://ollama.com/) |
| UI | [Streamlit](https://streamlit.io/) |
| Package manager | [uv](https://github.com/astral-sh/uv) |

---

## Requirements

- Python 3.13+
- [Ollama](https://ollama.com/) installed and running
- A TMDB API key (free — see below)

---

## Setup

**1. Clone and install dependencies**
```bash
git clone <your-repo-url>
cd Movie_rag_explorer
uv sync
```

**2. Add your TMDB API key**

Create a `.env` file in the project root:
```
TMDB_API_KEY=your_api_key_here
```

**3. Pull the LLM**
```bash
ollama pull gemma4:latest
```

**4. Ingest movies and TV shows into the vector DB**
```bash
uv run python ingest.py
```
This fetches ~3,800 titles from TMDB, embeds them, and saves them locally. Run once — or re-run whenever you want fresh data.

**5. Launch the app**
```bash
ollama serve          # if Ollama isn't already running
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```
Movie_rag_explorer/
├── .env                  # Your TMDB API key (never committed)
├── .streamlit/
│   └── config.toml       # Streamlit config (disables noisy file watcher)
├── app.py                # Streamlit chat UI
├── ingest.py             # Run this first to populate the vector DB
├── vector.py             # TMDB fetching, ChromaDB, embeddings, querying
├── main.py               # CLI for testing ingest/query without the UI
├── pyproject.toml        # Dependencies
└── chroma_db/            # Local vector DB (auto-created, not committed)
```

---

## CLI usage (for testing)

```bash
uv run python main.py ingest
uv run python main.py query "scary horror movies"
```

---

## TMDB API

This product uses the TMDB API but is not endorsed or certified by TMDB.

[![TMDB Logo](https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg)](https://www.themoviedb.org)

### About the TMDB API

The TMDB API provides access to a large collection of movie, TV show, and actor data including images. It is a free service for non-commercial use.

**Applying for an API key**
Apply at [themoviedb.org](https://www.themoviedb.org) by navigating to Settings → API in your account.

**SSL**
The TMDB API supports SSL across all endpoints and CDN assets. This project uses HTTPS for all requests.

**Commercial use**
This project uses the developer (non-commercial) API tier. If you intend to use this project commercially, contact [sales@themoviedb.org](mailto:sales@themoviedb.org). Commercial use requires a separate license.

> *A project is considered commercial if its primary purpose is to generate revenue for the owner.*

**Cost**
The TMDB API is free for non-commercial use, provided TMDB is attributed as the source of the data and images.

**Attribution requirements**

- Use the TMDB logo to identify your use of the API.
- Display the notice: *"This product uses the TMDB API but is not endorsed or certified by TMDB."*
- Attribution must appear in an "About" or "Credits" section of the application.
- The TMDB logo must be less prominent than your own application's primary branding.
- When linking to TMDB, use: [https://www.themoviedb.org](https://www.themoviedb.org)
- Refer to the company as either **"TMDB"** or **"The Movie Database"** — no other names are acceptable.
- Do not modify the TMDB logo (no color changes, flipping, rotating, or altering aspect ratio).

**Branding**
The TMDB logo may be used in white, black, or approved brand colors. See the [logos & attribution page](https://www.themoviedb.org/about/logos-attribution) for approved assets.

**Legal notice**
TMDB does not claim ownership of any images or data in the API. TMDB complies with the DMCA and removes infringing content when properly notified. Any data or images you upload grant TMDB a license to use. You are prohibited from using TMDB images or data in connection with libelous, defamatory, obscene, pornographic, abusive, or otherwise offensive content.

---

## License

This project is for non-commercial, educational use in accordance with the [TMDB API terms](https://www.themoviedb.org/documentation/api/terms-of-use).
