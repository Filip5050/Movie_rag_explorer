import streamlit as st
import ollama

from vector import query_movies, get_collection

st.set_page_config(page_title="Movie RAG Explorer", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.movie-title {
    font-size: 0.9rem;
    font-weight: 700;
    margin: 6px 0 2px 0;
    line-height: 1.3;
}
.movie-meta {
    font-size: 0.75rem;
    color: #888;
    margin: 2px 0;
}
.star-row {
    font-size: 0.95rem;
    margin: 4px 0;
    line-height: 1;
}
.movie-overview {
    font-size: 0.75rem;
    color: #bbb;
    margin-top: 6px;
    line-height: 1.4;
}
.movie-cast {
    font-size: 0.72rem;
    color: #9aa0a6;
    margin-top: 4px;
    font-style: italic;
}
.movie-providers {
    font-size: 0.72rem;
    color: #6cf;
    margin-top: 4px;
}
.trailer-link a {
    font-size: 0.78rem;
    color: #ff6464 !important;
    text-decoration: none;
    font-weight: 600;
}
.no-poster {
    background: #1e1e2e;
    height: 210px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    font-size: 3rem;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_stars(rating: float) -> str:
    n = min(10, max(0, round(float(rating))))
    return "★" * n + "☆" * (10 - n)


def expand_query(user_query: str) -> str:
    """Use Ollama to enrich the search query with likely titles, actors, and keywords."""
    try:
        response = ollama.chat(
            model="gemma4:latest",
            messages=[{
                "role": "user",
                "content": (
                    f"A user is searching for movies or TV series with this query: \"{user_query}\"\n\n"
                    "Expand it into a richer set of search keywords: include likely movie or show titles, "
                    "full actor/director names, genres, themes, and synonyms.\n"
                    "Return ONLY the expanded keywords as a single line. No explanation."
                ),
            }],
            stream=False,
        )
        expanded = response["message"]["content"].strip()
        return f"{user_query} {expanded}"
    except Exception:
        return user_query


def show_movie_cards(results: list[dict]) -> None:
    """Display a row of movie/series cards: poster, stars, cast, providers, trailer, overview."""
    if not results:
        return
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        m = r["metadata"]
        raw_overview = r["document"].split("Overview:")[-1].strip()
        overview = raw_overview[:130] + "…" if len(raw_overview) > 130 else raw_overview

        with col:
            poster = m.get("poster_path", "")
            if poster:
                st.image(
                    f"https://image.tmdb.org/t/p/w342{poster}",
                    use_container_width=True,
                )
            else:
                st.markdown("<div class='no-poster'>🎬</div>", unsafe_allow_html=True)

            badge = "🎬" if m.get("type") == "Movie" else "📺"
            st.markdown(
                f"<div class='movie-title'>{m['title']}</div>"
                f"<div class='movie-meta'>{badge} {m.get('type', 'Movie')} · {m['year']}</div>",
                unsafe_allow_html=True,
            )
            stars = render_stars(float(m["rating"]))
            st.markdown(
                f"<div class='star-row'>"
                f"<span style='color:#FFD700'>{stars}</span>"
                f"<span style='color:#888; font-size:0.8rem'> {m['rating']}/10</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='movie-meta'>{m['genres']}</div>",
                unsafe_allow_html=True,
            )

            directors = m.get("directors", "")
            cast = m.get("cast", "")
            if directors:
                st.markdown(
                    f"<div class='movie-cast'>Dir: {directors}</div>",
                    unsafe_allow_html=True,
                )
            if cast:
                st.markdown(
                    f"<div class='movie-cast'>★ {cast}</div>",
                    unsafe_allow_html=True,
                )

            providers = m.get("providers", "")
            if providers:
                st.markdown(
                    f"<div class='movie-providers'>📡 {providers}</div>",
                    unsafe_allow_html=True,
                )

            trailer_key = m.get("trailer_key", "")
            if trailer_key:
                st.markdown(
                    f"<div class='trailer-link'>"
                    f"<a href='https://www.youtube.com/watch?v={trailer_key}' target='_blank'>"
                    f"▶ Watch trailer</a></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div class='movie-overview'>{overview}</div>",
                unsafe_allow_html=True,
            )


def build_rag_prompt(user_query: str, context: str) -> str:
    return f"""You are a knowledgeable movie and TV series recommendation assistant. Use ONLY the titles listed in the context below. Do not hallucinate or invent titles not in the context.

If the context does not contain relevant titles, say so honestly and suggest the user try rephrasing.

CONTEXT:
{context}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Recommend 2-4 titles from the context that best match the request
- For each, briefly explain why it fits and mention the title, year, type (Movie/TV), and rating
- Reference the director and lead cast when relevant
- Keep your tone conversational and enthusiastic
- If asked for a specific number of recommendations, respect that"""


# --- Sidebar ---
with st.sidebar:
    st.title("🎬 Movie RAG Explorer")
    st.caption("TMDB · ChromaDB · Ollama gemma4")
    st.divider()

    try:
        col = get_collection()
        count = col.count()
        st.metric("Titles in DB", count)
        if count == 0:
            st.warning("DB is empty.\nRun: `uv run python ingest.py`")
    except Exception:
        st.error("DB not found.\nRun: `uv run python ingest.py`")

    st.divider()
    st.markdown("**Example questions:**")
    st.caption("• Horror Movies from the 80s")
    st.caption("• Scorsese crime films")
    st.caption("• Shows available on Netflix")
    st.caption("• Something like Breaking Bad")
    st.caption("• Psychological thrillers with plot twists")


# --- Main chat ---
st.title("Write a question?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("results"):
            show_movie_cards(message["results"])
            st.divider()
        st.markdown(message["content"])

if prompt := st.chat_input("Describe what kind of movie or series you're looking for..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking…"):
        expanded = expand_query(prompt)

    try:
        results = query_movies(expanded, n_results=5)
    except Exception as e:
        st.error(f"Vector DB error: {e}\n\nRun `uv run python ingest.py` first.")
        st.stop()

    if not results:
        st.warning("No titles in DB. Run `uv run python ingest.py` first.")
        st.stop()

    context_lines = []
    for r in results:
        m = r["metadata"]
        overview = r["document"].split("Overview:")[-1].strip()
        extras = []
        if m.get("directors"):
            extras.append(f"Dir: {m['directors']}")
        if m.get("cast"):
            extras.append(f"Cast: {m['cast']}")
        if m.get("providers"):
            extras.append(f"On: {m['providers']}")
        extras_line = " | ".join(extras)
        context_lines.append(
            f"- {m['title']} ({m['year']}) | {m.get('type', 'Movie')} | "
            f"Genres: {m['genres']} | Rating: {m['rating']}/10"
            + (f"\n  {extras_line}" if extras_line else "")
            + f"\n  {overview}"
        )
    context = "\n".join(context_lines)
    rag_prompt = build_rag_prompt(prompt, context)

    st.session_state.messages.append({
        "role": "assistant",
        "content": "",
        "results": results,
    })

    with st.chat_message("assistant"):
        show_movie_cards(results)
        st.divider()

        response_container = st.empty()
        full_response = ""
        try:
            stream = ollama.chat(
                model="gemma4:latest",
                messages=[{"role": "user", "content": rag_prompt}],
                stream=True,
            )
            for chunk in stream:
                full_response += chunk["message"]["content"]
                response_container.markdown(full_response + "▌")
                st.session_state.messages[-1]["content"] = full_response
            response_container.markdown(full_response)
            st.session_state.messages[-1]["content"] = full_response
        except ollama.ResponseError as e:
            st.error(
                f"Ollama error: {e}\n\n"
                "Make sure Ollama is running (`ollama serve`) "
                "and the model is pulled (`ollama pull gemma4:latest`)."
            )
            st.stop()
