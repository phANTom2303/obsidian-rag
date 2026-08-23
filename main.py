from chunking import chunk_file
from vectorize import Vectorizer

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

def pretty_print_vectorized(vectorized_data):
    for item in vectorized_data:
        print("=" * 80)
        print(f"ID: {item.get('id')}")
        print(f"Metadata: {item.get('metadata')}")
        print(f"Embedding Dimension: {len(item.get('embedding', []))}")
        print(f"Embedding Prefix (first 5 floats): {item.get('embedding', [])[:5]}")
        print("-" * 80)
        # print(item.get("document", ""))
        print()

def main():
    path = "/home/anish-goenka/Documents/obsidian-vault-current/cp-algorithms.com/graph/bellman_ford.md"
    
    print("Chunking document...")
    try:
        chunks = chunk_file(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        print("Please provide a valid markdown file path to test.")
        return
        
    print(f"Generated {len(chunks)} chunks.")
    
    print("Initializing Vectorizer (this may download the model on first run)...")
    vectorizer = Vectorizer()
    
    print(f"Vectorizing chunks...")
    vectorized_data = vectorizer.vectorize_chunks(chunks)
    
    print("\nVectorization complete. Output:")
    pretty_print_vectorized(vectorized_data)

if __name__ == "__main__":
    main()