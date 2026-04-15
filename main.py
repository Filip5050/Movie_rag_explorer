import sys

from vector import ingest_movies, query_movies


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest")
        print("  python main.py query '<question>'")
        return

    command = sys.argv[1]

    if command == "ingest":
        print("Fetching and embedding movies...")
        count = ingest_movies()
        print(f"Ingested {count} movies.")

    elif command == "query" and len(sys.argv) == 3:
        results = query_movies(sys.argv[2])
        for r in results:
            m = r["metadata"]
            print(f"{m['title']} ({m['year']}) | {m['genres']} | Rating: {m['rating']}/10")

    else:
        print("Unknown command. Use 'ingest' or 'query <text>'.")


if __name__ == "__main__":
    main()
