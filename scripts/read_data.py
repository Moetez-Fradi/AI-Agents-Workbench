from pathlib import Path
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text.strip()

def extract_text_from_txt(file_path):
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# change this based on the data...
def split_text_into_chunks(text, chunk_size=40, overlap=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_cheats_from_folder(folder="docs"):
    cheats = []
    for file in Path(folder).rglob("*"):
        if file.suffix.lower() == ".pdf":
            full_text = extract_text_from_pdf(file)
        elif file.suffix.lower() == ".txt":
            full_text = extract_text_from_txt(file)
        else:
            continue

        chunks = split_text_into_chunks(full_text)
        cheats.extend(chunks)
    
    return [c for c in cheats if c.strip()]
