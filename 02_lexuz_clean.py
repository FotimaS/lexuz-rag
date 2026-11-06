import json
import re
from pathlib import Path


INPUT = Path("lexuz_chunks.json")
OUTPUT = Path("lexuz_chunks_clean.json")

OKOZ_BLOCK = re.compile(r"\[OKOZ\s*:[^\]]*\]", re.IGNORECASE)
SITE_CONTROLS = re.compile(
    r"Ҳужжатга таклиф юбориш\s*"
    r"Аудиони тинглаш\s*"
    r"Ҳужжат элементидан ҳавола олиш"
)


def clean_text(text):
    text = OKOZ_BLOCK.sub("", text)
    text = SITE_CONTROLS.sub("", text)
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


def main():
    chunks = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    changed = 0

    for chunk in chunks:
        original = chunk["text"]
        cleaned = clean_text(original)
        if not cleaned:
            raise ValueError(f"Tozalashdan keyin matn bo‘sh: {chunk['chunk_id']}")
        changed += original != cleaned
        chunk["text"] = cleaned

    OUTPUT.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{OUTPUT}: {len(chunks)} chunk, {changed} tasi tozalandi.")


if __name__ == "__main__":
    main()