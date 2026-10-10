"""Turn a fact sheet into a short plain-language explanation.

The model is used as a formatter. It receives only the fact sheet, and its output is checked
against that fact sheet before anyone sees it: every number must trace to a fact, no clinical or
mechanistic language the facts did not supply, no disorder the facts do not name, and 2 to 8
sentences. A rejected output gets one retry with the complaint attached; a second rejection, or
any failure to reach the model, falls back to the deterministic template. Rejections are logged
to results/tables/explain_failures.tsv, which is evidence, not noise.

Nothing here can change a band, a score or a verdict: the layer runs strictly after them and
only reads them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from epigrade import paths
from epigrade.explain import groq_client, template

SOURCE_MODEL = "model"
SOURCE_TEMPLATE = "template"
SMALL_COUNT = 13  # integers below this are ordinary prose counts ("two studies")
CORPUS_DISORDERS = (
    "Silver-Russell", "Sotos", "Kabuki", "Down", "Coffin-Siris", "Williams", "CHARGE",
    "Nicolaides-Baraitser", "ICF", "7q11.23", "Weaver", "Claes-Jensen",
)
FORBIDDEN = [
    (r"\bdiagnos(?:is|es|ed|e|ing|tic)\b", "diagnostic language"),
    (r"\bpatients? should\b|\byou should\b|\bconsult\b|\brecommend", "advice"),
    (r"\btreat(?:ment|ed|ing)?\b|\btherap(?:y|ies|eutic)\b|\bprognos", "clinical advice"),
    (r"\bmutation", "mutation language"),
    (r"\bprotein\b|\benzyme\b|\bhistone\b|\bchromatin\b|\bmethyltransferase\b|"
     r"\btranscription factor\b|\bgene function\b|\bencodes?\b", "gene-function claim"),
    (r"\bmechanism|\bpathway\b|\bcaused by\b|\bbecause the gene\b|\bsuggests that\b",
     "mechanistic speculation"),
    (r"\blikely pathogenic\b|\bpathogenic\b|\blikely benign\b|\bbenign\b",
     "pathogenicity language"),
    (r"\binterpret(?:s|ed)? the biology\b|\bunderstands?\b", "claims of understanding"),
]
SYSTEM_PROMPT = (
    "You convert a precomputed analysis result into plain language for a reader who is not a "
    "geneticist. You are given a JSON fact sheet. Restate only what is in it. Do not add "
    "biological mechanism, gene function, clinical interpretation, diagnosis, advice, or any "
    "number that is not in the fact sheet. Do not speculate about why a result occurred. If the "
    "fact sheet says a result was refused or scoped, say so plainly. Write 3 to 5 sentences, no "
    "bullet points, no headings, no hedging phrases, no restating the JSON keys. British "
    "spelling."
)
INSTRUCTION = "Write the plain-language summary of this fact sheet now."


@dataclass
class Explanation:
    text: str
    source: str                 # "model" or "template"
    model: str | None
    fact_hash: str
    created_utc: str
    attempts: int = 0           # model calls made for this explanation
    rejections: list[str] = field(default_factory=list)
    fallback_reason: str = ""   # why the template was used, if it was


def fact_hash(facts: dict) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- validation


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def _numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text)]


def _as_float(token: str) -> float:
    return float(token.rstrip("%").replace(",", ""))


def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def allowed_numbers(facts: dict) -> set[float]:
    """Every number the facts contain - numeric values and numbers inside strings such as
    "5/18" or an accession - in each form the prose may legitimately use: as given, as a
    percentage of a fraction or a fraction of a percentage, and rounded to 0, 1 or 2 places."""
    raw = set()
    for v in _walk(facts):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            raw.add(float(v))
        elif isinstance(v, str):
            raw.update(_as_float(t) for t in _numbers_in(v))
    allowed = set()
    for x in raw:
        for y in (x, x * 100, x / 100):
            allowed.update({round(y, 0), round(y, 1), round(y, 2), round(y, 4)})
    return allowed


def validate(text: str, facts: dict) -> list[str]:
    """Reasons to reject `text`; empty if it may be shown."""
    problems = []
    allowed = allowed_numbers(facts)
    facts_text = json.dumps(facts, sort_keys=True).lower()
    for token in _numbers_in(text):
        value = _as_float(token)
        if value.is_integer() and 0 <= value < SMALL_COUNT and not token.endswith("%"):
            continue
        if not any(abs(value - a) < 1e-9 for a in allowed):
            problems.append(f"number not in the fact sheet: {token}")
    lowered = text.lower()
    for pattern, label in FORBIDDEN:
        for m in re.finditer(pattern, lowered):
            if m.group(0) not in facts_text:
                problems.append(f"{label}: '{m.group(0)}'")
    for name in CORPUS_DISORDERS:
        if name.lower() in lowered and name.lower() not in facts_text:
            problems.append(f"disorder not in the fact sheet: {name}")
    for m in re.finditer(r"\b([A-Z][\w-]+) syndrome\b", text):
        if m.group(1).lower() not in facts_text:
            problems.append(f"disorder not in the fact sheet: {m.group(0)}")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip()) if s]
    if not 2 <= len(sentences) <= 8:
        problems.append(f"{len(sentences)} sentences (allowed 2 to 8)")
    if re.search(r"^\s*([-*•]|#)", text, re.M):
        problems.append("bullet points or headings")
    return list(dict.fromkeys(problems))


# --------------------------------------------------------------------------- cache + log


def cache_dir():
    path = paths.repo_root() / "results" / "cache" / "explanations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached(h: str) -> Explanation | None:
    path = cache_dir() / f"{h}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Explanation(**data)


def _store(explanation: Explanation) -> None:
    with open(cache_dir() / f"{explanation.fact_hash}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(explanation), f, indent=2, sort_keys=True)
        f.write("\n")


def failures_path():
    return paths.tables_dir() / "explain_failures.tsv"


FAILURE_COLUMNS = ["timestamp_utc", "fact_hash", "kind", "subject", "attempt", "model",
                   "reason", "rejected_text"]


def _log_failure(facts: dict, h: str, attempt: int, model: str, reasons: list[str],
                 text: str) -> None:
    path = failures_path()
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if new:
            writer.writerow(FAILURE_COLUMNS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"), h, facts.get("kind"),
            facts.get("disorder") or facts.get("top_match") or facts.get("verdict"), attempt,
            model, "; ".join(reasons), " ".join(text.split()),
        ])


# --------------------------------------------------------------------------- main entry


def explain(facts: dict, use_cache: bool = True, allow_model: bool = True) -> Explanation:
    """Explanation for a fact sheet: from the cache if present, else from the model if it is
    reachable and passes validation, else from the template."""
    h = fact_hash(facts)
    if use_cache:
        hit = _cached(h)
        if hit is not None:
            return hit

    model = groq_client.config()["model"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rejections: list[str] = []
    attempts = 0
    fallback = "model not requested"
    if allow_model:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts, sort_keys=True, indent=1)},
            {"role": "user", "content": INSTRUCTION},
        ]
        for attempt in range(1 + int(groq_client.config()["validation_retries"])):
            text, status = groq_client.chat(messages)
            if text is None:
                fallback = status
                break
            attempts += 1
            problems = validate(text, facts)
            if not problems:
                result = Explanation(text=text, source=SOURCE_MODEL, model=model, fact_hash=h,
                                     created_utc=now, attempts=attempts, rejections=rejections)
                _store(result)
                return result
            rejections.append("; ".join(problems))
            _log_failure(facts, h, attempt + 1, model, problems, text)
            fallback = "validator rejected the model output twice"
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "That summary was rejected by an automatic check: "
                    + "; ".join(problems)
                    + ". Rewrite it using only the facts given, and fix each problem."
                )},
            ]
    text = template.explain(facts)
    result = Explanation(text=text, source=SOURCE_TEMPLATE, model=None, fact_hash=h,
                         created_utc=now, attempts=attempts, rejections=rejections,
                         fallback_reason=fallback)
    # Cache a template only when the model was reached and its output rejected twice. A
    # template produced because the key is missing or the network failed is not cached, so a
    # later run with a working key is not stuck with it.
    if rejections:
        _store(result)
    return result
