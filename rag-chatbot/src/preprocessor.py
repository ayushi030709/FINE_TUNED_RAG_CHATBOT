# src/preprocessor.py
import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Read all text from PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def clean_text(text: str) -> str:
    """Remove extra spaces, fix broken lines."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 newlines in a row
    text = re.sub(r'[ \t]+', ' ', text)        # collapse spaces
    text = re.sub(r'\n ', '\n', text)          # remove leading spaces
    return text.strip()

def sentence_aware_chunk(text: str, max_words: int = 200, overlap_words: int = 30) -> list:
    """
    Split text into chunks of ~200 words.
    overlap_words = last 30 words of previous chunk are repeated
    at start of next chunk so context is not lost.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "chunk_id": chunk_index,
                "text": chunk_text,
                "word_count": current_word_count
            })
            chunk_index += 1

            # Keep last 30 words for overlap
            overlap_text = ' '.join(' '.join(current_chunk).split()[-overlap_words:])
            current_chunk = [overlap_text, sentence]
            current_word_count = len(overlap_text.split()) + word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    # Save the last remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            "chunk_id": chunk_index,
            "text": chunk_text,
            "word_count": len(chunk_text.split())
        })

    return chunks

def process_document(pdf_path: str, output_dir: str = "chunks") -> list:
    """Main function: PDF → clean → chunk → save JSON."""
    Path(output_dir).mkdir(exist_ok=True)

    print(f" Reading PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    print(" Cleaning text...")
    clean = clean_text(raw_text)

    print(" Chunking into ~200 word segments...")
    chunks = sentence_aware_chunk(clean, max_words=200, overlap_words=30)

    # Save to JSON
    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f" Done! {len(chunks)} chunks saved to {output_path}")
    return chunks

if __name__ == "__main__":
    process_document("data/AI_Training_Document.pdf")