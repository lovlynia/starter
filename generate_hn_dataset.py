import csv
import html
import json
import re
import subprocess
from pathlib import Path


API_URL = "https://hn.algolia.com/api/v1/search_by_date?tags=comment&hitsPerPage=100&page={page}"
OUTPUT_PATH = Path("hn_hn_comments_labeled.csv")
TARGET_PER_LABEL = 70
MAX_PAGES = 20

LABELS = {
    "analysis": "The comment mainly explains, argues, or evaluates with concrete reasoning, evidence, mechanism, or comparison.",
    "experience": "The comment mainly relies on the writer's own experience, project, job, or personal example to make its point.",
    "reaction": "The comment is primarily a quick reaction, joke, praise, dismissal, or unsupported opinion rather than developed reasoning.",
}

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|me|my|mine|we|we're|we've|our|ours|personally)\b", re.IGNORECASE)
EXPERIENCE_MARKERS = (
    "built",
    "made",
    "used",
    "worked",
    "tried",
    "saw",
    "remember",
    "deployed",
    "shipped",
    "running",
    "maintain",
    "maintained",
    "found",
    "personally",
    "in my",
    "at my",
    "on my team",
    "at work",
    "for work",
    "i have",
    "i had",
    "i take",
    "i noticed",
    "i use",
    "i used",
    "my site",
    "my team",
    "my job",
    "my experience",
)
ANALYSIS_MARKERS = (
    "because",
    "therefore",
    "however",
    "although",
    "if",
    "since",
    "which means",
    "for example",
    "for instance",
    "the reason",
    "in practice",
    "in theory",
    "compared",
    "comparison",
    "evidence",
    "regulation",
    "data",
    "stat",
    "benchmark",
    "probability",
    "distribution",
    "tradeoff",
    "cost",
    "benefit",
    "impact",
    "model",
    "system",
)


def fetch_page(page: int) -> list[dict]:
    result = subprocess.run(
        ["curl", "-s", API_URL.format(page=page)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return payload["hits"]


def clean_text(raw_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(raw_text or ""))
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def classify_comment(text: str) -> tuple[str | None, str]:
    if len(text) < 40:
        return None, "too_short"

    sentence_count = len(re.findall(r"[.!?]", text))
    has_first_person = bool(FIRST_PERSON.search(text))
    has_experience = contains_any(text, EXPERIENCE_MARKERS)
    has_analysis = contains_any(text, ANALYSIS_MARKERS)
    has_link = "http://" in text or "https://" in text
    has_number = bool(re.search(r"\b\d+[\d.,%]*\b", text))

    if has_first_person and has_experience and len(text) >= 80:
        return "experience", "first_person_experience"

    if len(text) < 120 and not has_analysis and not has_link:
        return "reaction", "short_reaction"

    if has_analysis or has_link or (has_number and len(text) >= 140) or (sentence_count >= 2 and len(text) >= 160):
        return "analysis", "reasoned_structure"

    return "reaction", "default_reaction"


def is_balanced(label_counts: dict[str, int]) -> bool:
    return all(count >= TARGET_PER_LABEL for count in label_counts.values())


def write_output(rows: list[dict[str, str]], label_counts: dict[str, int]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "notes", "story_title", "author", "hn_object_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    for label_name, count in label_counts.items():
        print(f"{label_name}: {count}")


def add_row(hit: dict, label: str, note: str, text: str, rows: list[dict[str, str]], label_counts: dict[str, int], seen_ids: set[str]) -> None:
    object_id = hit["objectID"]
    seen_ids.add(object_id)
    label_counts[label] += 1
    rows.append(
        {
            "text": text,
            "label": label,
            "notes": note,
            "story_title": (hit.get("story_title") or "").strip(),
            "author": (hit.get("author") or "").strip(),
            "hn_object_id": object_id,
        }
    )


def main() -> None:
    label_counts = dict.fromkeys(LABELS, 0)
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for page in range(MAX_PAGES):
        for hit in fetch_page(page):
            object_id = hit.get("objectID")
            if not object_id or object_id in seen_ids:
                continue

            text = clean_text(hit.get("comment_text") or "")
            label, note = classify_comment(text)
            if not label or label_counts[label] >= TARGET_PER_LABEL:
                continue

            add_row(hit, label, note, text, rows, label_counts, seen_ids)

            if is_balanced(label_counts):
                write_output(rows, label_counts)
                return

    raise RuntimeError(f"Unable to gather enough comments with balance {label_counts}")


if __name__ == "__main__":
    main()