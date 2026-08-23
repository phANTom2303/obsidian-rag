import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Reads a markdown file, splits it into chunks using a hierarchy of separators, 
    and returns a list of chunks with metadata.
    
    Args:
        file_path (str): The path to the markdown file.
        chunk_size (int): The target size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        
    Returns:
        List[Dict[str, Any]]: List of dictionary chunks containing 'page_content' and 'metadata'.
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    file_name = os.path.basename(file_path)
    
    # Hierarchy list of separators designed to respect markdown structure.
    # Placing the code block delimiter `\n```\n` before smaller elements like `\n\n`
    # or `\n` instructs the RecursiveCharacterTextSplitter to split around code blocks 
    # instead of inside them, keeping a single codeblock completely present in a 
    # single chunk (provided the codeblock is within the chunk_size limit).
    separators = [
        "\n# ",       # Heading 1
        "\n## ",      # Heading 2
        "\n### ",     # Heading 3
        "\n#### ",    # Heading 4
        "\n##### ",   # Heading 5
        "\n###### ",  # Heading 6
        "\n---",      # Horizontal rules
        "\n```\n",    # Codeblocks
        "\n\n",       # Paragraphs
        "\n- ",       # Unordered lists
        "\n* ",       # Unordered lists
        "\n> ",       # Blockquotes
        "\n1. ",      # Ordered lists (best effort for start of lists)
        "\n",         # Newlines
        " ",          # Words
        ""            # Characters
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
        is_separator_regex=False
    )
    
    # Generate the chunks
    raw_chunks = splitter.split_text(text)
    
    chunks = []
    # Inject metadata into each chunk
    for i, content in enumerate(raw_chunks):
        chunks.append({
            "page_content": content.strip(),
            "metadata": {
                "file_path": file_path,
                "file_name": file_name,
                "chunk_number": i + 1
            }
        })
        
    return chunks

if __name__ == "__main__":
    # Example usage / basic test
    pass
