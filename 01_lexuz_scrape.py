import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup


DOC_PATH = re.compile(r"/(?:uz/)?docs/(-?\d+)/?")
ARTICLE = re.compile(
    r"^(?P<number>\d+(?:[-‑–]\d+|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)?)"
    r"\s*[-‑–.]?\s*(?:modda|модда)\b",
    re.IGNORECASE,
)
DEFAULT_SELECTORS = (
    "#divDocument",
    "#document",
    ".document-content",
    ".docview",
    "main",
)


def document_id(url):
    parts = urlsplit(url)
    match = DOC_PATH.fullmatch(parts.path)
    if (
        parts.scheme != "https"
        or parts.hostname not in {"lex.uz", "www.lex.uz"}
        or match is None
    ):
        raise ValueError(f"Lex.uz hujjat havolasi kutilgan: {url}")
    return match.group(1)


def normalize(text):
    return " ".join(text.split())


def extract_articles(container):
    # Inline teglar ichidagi so‘zlar bitta sarlavhada qoladi.
    # Blok elementlari orasida yangi qator qo‘shiladi.
    for block in container.select("p, div, li, h1, h2, h3, h4, tr, br"):
        block.insert_before("\n")
        if block.name != "br":
            block.append("\n")

    articles = []
    current = None

    for raw_line in container.get_text().splitlines():
        line = normalize(raw_line)
        if not line:
            continue

        heading = ARTICLE.match(line)
        if heading:
            current = {
                "article": heading.group("number"),
                "lines": [line],
            }
            articles.append(current)
        elif current is not None:
            current["lines"].append(line)

    return [
        article for article in articles
        if len(article["lines"]) > 1
    ]


def parse_document(html, selector=None):
    selectors = (selector,) if selector else DEFAULT_SELECTORS

    for candidate in selectors:
        soup = BeautifulSoup(html, "html.parser")
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        container = soup.select_one(candidate)
        if container is None:
            continue

        for element in container.select(
            "script, style, nav, footer, button, form, [hidden]"
        ):
            element.decompose()

        articles = extract_articles(container)
        if articles:
            return page_title, articles, candidate

    raise ValueError(
        "Moddalar ajratilmadi. raw_html faylini oching va "
        "hujjat matni konteynerini --selector orqali ko‘rsating."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=Path, default=Path("urls.txt"))
    parser.add_argument("--output", type=Path, default=Path("lexuz_chunks.json"))
    parser.add_argument("--selector", help="Hujjat matni konteynerining CSS selektori")
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    if args.delay < 1:
        parser.error("--delay kamida 1 soniya bo‘lsin.")

    urls = list(dict.fromkeys(
        line.strip()
        for line in args.urls.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))
    if not 1 <= len(urls) <= 10:
        parser.error("Pilot sinov uchun urls.txt ichiga 1–10 ta URL kiriting.")

    raw_dir = Path("raw_html")
    raw_dir.mkdir(exist_ok=True)
    chunks = []
    failures = 0

    with httpx.Client(
        timeout=25,
        follow_redirects=True,
        headers={"User-Agent": "LexUzRagPilot/0.1"},
    ) as client:
        for position, url in enumerate(urls):
            if position:
                time.sleep(args.delay)

            try:
                doc_id = document_id(url)
                response = client.get(url)

                if response.status_code in {403, 429}:
                    print(
                        f"STOP {response.status_code}: {url}. "
                        f"Retry-After: {response.headers.get('Retry-After', '—')}"
                    )
                    failures += 1
                    break

                response.raise_for_status()
                source_url = str(response.url)
                document_id(source_url)

                raw_path = raw_dir / f"{position + 1:02d}_{doc_id}.html"
                raw_path.write_bytes(response.content)
                title, articles, used_selector = parse_document(
                    response.text, args.selector
                )
                fetched_at = datetime.now(timezone.utc).isoformat()

                for index, article in enumerate(articles, start=1):
                    chunks.append({
                        "id": doc_id,
                        "chunk_id": f"article-{article['article']}-{index}",
                        "title": title,
                        "text": "\n".join(article["lines"]),
                        "source": "lex.uz",
                        "source_url": source_url,
                        "article": article["article"],
                        "date": None,
                        "version_id": None,
                        "lang": None,
                        "fetched_at": fetched_at,
                    })

                print(f"OK {doc_id}: {len(articles)} modda [{used_selector}]")

            except (httpx.HTTPError, ValueError) as error:
                failures += 1
                print(f"ERROR {url}: {error}")

    if not chunks:
        raise SystemExit("Hech qanday chunk yozilmadi. Yuqoridagi xatoni tekshiring.")

    args.output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{args.output}: {len(chunks)} chunk, {failures} xato.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()