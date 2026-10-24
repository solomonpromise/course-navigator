"""Freeze real course prose, section-group splits, provenance and authored probes.

Run once before training. The app's retrieval library contains all published
sections, but classifier fitting sees train rows only. Retrieval and prediction
are separate operations and have separate evaluation claims.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coursenav.data import LABELS, words
from sklearn.model_selection import train_test_split

PLATFORM = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
BASE_URL = "https://solomonpromise.github.io/deep-learning-lab/"


def clean(markdown):
    """Remove formatting/links/code while retaining the explanatory prose."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|attachment:\S+", " ", text)
    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s*(?:#{1,6}|>)\s*", "", text)
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip()


def main():
    revision = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=PLATFORM).decode().strip()
    sections, source_manifest, seen = [], [], set()
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "notes"], cwd=PLATFORM).decode().splitlines()
    for path in paths:
        match = re.fullmatch(r"notes/module-(\d+)/lesson-(\d+\.\d+)\.ipynb", path)
        if not match or int(match[1]) > 8:
            continue
        module, lesson = int(match[1]), match[2]
        label = 0 if module <= 2 else 1 if module <= 4 else module - 3
        raw = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=PLATFORM)
        source_manifest.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest()))
        notebook = json.loads(raw)
        current = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            source = "".join(cell["source"])
            # Keep numbered teaching sections; omit setup/objectives/navigation.
            for part in re.split(r"(?m)(?=^##\s+)", source):
                heading = re.match(r"##\s+(\d+)[.)]\s+([^\n]+)", part)
                if part.startswith("## "):
                    if heading:
                        number, title = int(heading[1]), re.sub(r"<[^>]+>", "", heading[2]).strip()
                        current = dict(id=f"{lesson}-s{number}", lesson=lesson, section=number,
                                       title=title, label=label, topic=LABELS[label], paragraphs=[],
                                       url=f"{BASE_URL}module-{module}/lesson-{lesson.replace('.', '-')}.html#s{number}")
                        sections.append(current)
                    else:
                        current = None
                if current:
                    # Question/answer scaffolding is not a new real learner question.
                    text = clean(re.sub(r"^##[^\n]+", "", part))
                    if len(words(text)) >= 20:
                        current["paragraphs"].append(text)
    rows = []
    for section in sections:
        text = " ".join(section.pop("paragraphs"))
        tokens = text.split()
        section["text"] = text
        # Non-overlapping windows are grouped by their entire source section.
        for start in range(0, len(tokens), 90):
            chunk = " ".join(tokens[start:start + 90])
            if len(words(chunk)) < 25:
                continue
            digest = hashlib.sha256(" ".join(words(chunk)).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(dict(id=f"{section['id']}-p{start//90}", group=section["id"],
                             lesson=section["lesson"], title=section["title"], url=section["url"],
                             text=chunk, label=section["label"], text_sha256=digest, origin="course_prose"))
    group_labels = {row["group"]: row["label"] for row in rows}
    groups = sorted(group_labels)
    train, remaining = train_test_split(groups, test_size=.30, stratify=[group_labels[g] for g in groups], random_state=42)
    valid, test = train_test_split(remaining, test_size=.5, stratify=[group_labels[g] for g in remaining], random_state=43)
    assignment = {g: split for split, gs in [("train", train), ("validation", valid), ("test", test)] for g in gs}
    for row in rows:
        row["split"] = assignment[row["group"]]
    (ROOT / "data/passages.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    # Excerpts carry precise section attribution; the index is not training data.
    (ROOT / "data/sections.json").write_text(json.dumps([s for s in sections if len(words(s['text'])) >= 25], ensure_ascii=False, indent=2))
    manifest = dict(source_revision=revision, source_repository="solomonpromise/deep-learning-lab",
                    seed=42, split_unit="complete source section; non-overlapping passage windows",
                    limitation="Sections of one lesson can appear in different splits. This measures new passages within this course, not new-course generalization.",
                    text_origin="Real prose from the published course notebooks; code cells and notebook outputs excluded.",
                    labels=LABELS, sources=source_manifest, rows=len(rows), groups=len(groups),
                    split_counts=dict(Counter(row["split"] for row in rows)),
                    per_label=dict(Counter(LABELS[row["label"]] for row in rows)))
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'}, indent=2))


if __name__ == "__main__":
    main()
