from ingest import chunk_file


def pretty_print_chunks(chunks):
    for chunk in chunks:
        page_content = chunk.get("page_content", "")
        metadata = chunk.get("metadata", {})

        print("=" * 80)
        print(f"Chunk #{metadata.get('chunk_number', 'N/A')} - {metadata.get('file_name', 'N/A')}")
        print(f"Path: {metadata.get('file_path', 'N/A')}")
        print("-" * 80)
        print(page_content)
        print()

def main():
    path = "/home/anish-goenka/Documents/obsidian-vault-current/cp-algorithms.com/graph/bellman_ford.md"
    chunks = chunk_file(path)
    pretty_print_chunks(chunks)
    

if __name__ == "__main__":
    main()