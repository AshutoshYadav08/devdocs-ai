from pathlib import Path


def load_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found :{file_path}")

    if path.suffix != '.txt':
        raise ValueError("Only text files are supported")

    text =  path.read_text(encoding="utf-8")
    text = preprocess_text(text)

    return {
        "text": text,
        "metadata": {
            "source": path.name
        }
    }


def preprocess_text(text):
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)