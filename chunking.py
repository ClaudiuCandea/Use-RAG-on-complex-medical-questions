from typing import List
import re

def word_splitter(source_text: str) -> List[str]:
    import re
    source_text = re.sub(r"\s+", " ", source_text)
    return re.split(r"\s", source_text)

def get_chunks_fixed_size(text: str, chunk_size: int) -> List[str]:
    text_words = word_splitter(text)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[i: i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks

def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk = " ".join(text_words[max(i - overlap_int, 0): i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_chunks_by_paragraph(source_text: str) -> List[str]:
    chunks = source_text.split(".\n")
    return chunks

def get_chunks_by_paragraph_and_min_length(source_text: str) -> List[str]:
    text = re.sub(r'\n+', '\n', source_text)
    chunks = text.split(".\n")

    # Chunking
    new_chunks = list()
    chunk_buffer = ""
    min_length = 50

    for chunk in chunks:
        new_buffer = chunk_buffer + chunk  # Create new buffer
        new_buffer_words = new_buffer.split(" ")  # Split into words
        if len(new_buffer_words) < min_length:  # Check whether buffer length too small
            chunk_buffer = new_buffer  # Carry over to the next chunk
        else:
            new_chunks.append(new_buffer)  # Add to chunks
            chunk_buffer = ""

    if len(chunk_buffer) > 0:
        new_chunks.append(chunk_buffer)  # Add last chunk, if necessary
    return new_chunks

