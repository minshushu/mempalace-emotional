"""
emotion_llm.py — Tag drawers with emotional metadata via opt-in LLM post-processing.

Parallel to closet_llm.py: regex-based mining stays API-free, but users can
optionally run this pass to enrich drawer metadata with three fields:

    intensity      — int 0-10, emotional charge (0 = neutral / informational)
    emotion_label  — short emotion words ("惆怅,无力" / "释然" / "" if neutral)
    reflection     — first-person 1-2 sentence reflection ("" if neutral)

Why a separate pass:
- Core mining (convo_miner.py) is regex-only by design — see CLAUDE.md
  "Local-first, zero API". This module honors that boundary.
- Closet generation (closet_llm.py) writes the *closet* index collection.
  Emotional metadata belongs on the *drawer* itself, since the same topic
  can appear in drawers with completely different emotional weights.

Bring-your-own-LLM. Same OpenAI-compatible config as closet_llm:
    LLM_ENDPOINT  — base URL (required)
    LLM_KEY       — bearer token (optional)
    LLM_MODEL     — model name (required)

Defaults to processing only ``emotional`` and ``milestone`` rooms — drawers
in ``decision`` / ``preference`` rooms are typically informational and would
get noisy emotion labels.

CLI:
    python -m mempalace.emotion_llm \\
        --palace ~/.mempalace/palace \\
        --rooms emotional,milestone \\
        --sample 10
"""

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Optional

from .closet_llm import LLMConfig, HTTP_TIMEOUT_S
from .palace import get_collection

# Bump when prompt or schema changes meaningfully — old tags then become
# stale and a re-run (without --force) will pick them up.
EMOTION_VERSION = 1

DEFAULT_ROOMS = ("emotional", "milestone")

# Drawers are per-chunk; content is much smaller than the multi-drawer
# concatenation closet_llm sends, so we cap lower.
MAX_CONTENT_CHARS = 8000
MAX_OUTPUT_TOKENS = 400

PROMPT_TEMPLATE = """You are reading a memory drawer. Tag it with emotional metadata.

CONTENT:
{content}

---

Output a JSON object with EXACTLY these fields:

{{
  "intensity": <integer 0-10>,
  "emotion_label": "<comma-separated short emotion words, or empty string>",
  "reflection": "<one to two sentences first-person reflection, or empty string>"
}}

RULES:
- intensity: 0 = no emotional charge (purely informational, technical, factual);
  10 = overwhelming feeling. Most everyday content sits between 2-5.
- emotion_label: 1-3 short words in the SAME language as the content above
  (e.g. Chinese: "惆怅,无力" / "释然" / "兴奋,紧张"). Empty string if intensity=0.
- reflection: write in FIRST PERSON ("I"/"我"), as if the speaker themselves
  is reflecting. Aim for the felt sense — what left weight here? what
  touched something fragile? what is still unresolved? what can be let go?
  Use everyday spoken voice in the SAME language as the content. NOT clinical
  or third-person ("the user expressed..."). Empty string if intensity=0
  or content is purely informational.
- If the content is mostly direct quotes, narrative excerpts, or
  third-party records (not the speaker's own words — e.g. blockquote
  lines starting with ">", quoted novel passages, transcripts), tag the
  SPEAKER's reaction to the material — NOT the emotions inside the
  quotes. If the speaker barely reacts (records or quotes without
  commentary), set intensity=0.
- Output valid JSON only. No code fences. No commentary.
"""


def _call_llm_for_emotion(cfg: LLMConfig, content: str):
    """Single LLM call returning (parsed_dict, usage_dict) or (None, None).

    Inline copy of closet_llm._call_llm's HTTP/retry core — kept separate
    so the two modules can evolve their prompts independently. Refactor
    to a shared helper if both prompts stabilize.
    """
    prompt = PROMPT_TEMPLATE.format(content=content[:MAX_CONTENT_CHARS])

    body = json.dumps(
        {
            "model": cfg.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if cfg.key:
        headers["Authorization"] = f"Bearer {cfg.key}"

    url = f"{cfg.endpoint}/chat/completions"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            text = payload["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)
            return parsed, payload.get("usage")
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            return None, None
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 2:
                time.sleep(2**attempt)
                continue
            return None, None
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 2:
                time.sleep(2**attempt)
                continue
            return None, None
    return None, None


_QUOTE_LINE = re.compile(r"^\s*>")


def _estimate_quote_ratio(text: str) -> float:
    """Fraction of non-blank lines that are blockquote markers.

    1.0 = pure quoted material; 0.0 = no markdown blockquote at all.
    Used as a coarse signal that a source file (when aggregated) is
    dominated by third-party text rather than the speaker's own voice.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    quote_lines = sum(1 for ln in lines if _QUOTE_LINE.match(ln))
    return quote_lines / len(lines)


def _normalize_parsed(parsed: dict) -> dict:
    """Coerce LLM output into the metadata schema. ChromaDB metadata can't
    hold None/list, so we flatten and cap."""
    try:
        intensity = int(parsed.get("intensity", 0))
    except (TypeError, ValueError):
        intensity = 0
    intensity = max(0, min(10, intensity))

    raw_label = parsed.get("emotion_label", "")
    if isinstance(raw_label, list):
        emotion_label = ",".join(str(x).strip() for x in raw_label if str(x).strip())
    elif isinstance(raw_label, str):
        emotion_label = raw_label.strip()
    else:
        emotion_label = ""
    emotion_label = emotion_label[:200]

    raw_refl = parsed.get("reflection", "")
    reflection = raw_refl.strip() if isinstance(raw_refl, str) else ""
    reflection = reflection[:1000]

    # If LLM judged this neutral, blank text fields so they don't leak into
    # the emotion-aware retrieval path as faint noise.
    if intensity == 0:
        emotion_label = ""
        reflection = ""

    return {
        "intensity": intensity,
        "emotion_label": emotion_label,
        "reflection": reflection,
        "emotion_version": EMOTION_VERSION,
    }


def tag_emotions(
    palace_path: str,
    rooms: tuple = DEFAULT_ROOMS,
    wing: Optional[str] = None,
    sample: int = 0,
    dry_run: bool = False,
    force: bool = False,
    max_quote_ratio: float = 1.0,
    cfg: Optional[LLMConfig] = None,
):
    """Tag drawers with emotional metadata.

    Reads drawers from the palace, calls the configured LLM for each one,
    writes (intensity, emotion_label, reflection, emotion_version) back to
    drawer metadata via ChromaDB ``update``.

    Skips drawers whose ``emotion_version`` already matches EMOTION_VERSION
    unless ``force=True``.
    """
    if cfg is None:
        cfg = LLMConfig()
    missing = cfg.missing()
    if missing:
        print("Error: missing configuration: " + ", ".join(missing))
        print("Set env vars LLM_ENDPOINT / LLM_MODEL (and optionally LLM_KEY),")
        print("or pass --endpoint / --model / --key on the CLI.")
        return {"error": "missing-config", "missing": missing}

    drawers_col = get_collection(palace_path, create=False)
    total = drawers_col.count()
    if total == 0:
        print("No drawers in palace.")
        return {"processed": 0}

    rooms_set = set(rooms)
    eligible: list = []  # (id, doc, meta)

    # Paginate the fetch to avoid SQLITE_MAX_VARIABLE_NUMBER on large palaces
    # — same pattern as closet_llm.regenerate_closets.
    batch_size = 5000
    offset = 0
    while offset < total:
        batch = drawers_col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = batch["ids"]
        if not ids:
            break
        for doc_id, doc, meta in zip(ids, batch["documents"], batch["metadatas"]):
            meta = meta or {}
            if wing and meta.get("wing") != wing:
                continue
            if meta.get("room") not in rooms_set:
                continue
            if not force and meta.get("emotion_version") == EMOTION_VERSION:
                continue
            eligible.append((doc_id, doc, meta))
        offset += len(ids)

    # Source-level filter: when an entire source file is dominated by
    # blockquoted material (e.g. reading-companion sessions where the
    # user pastes long novel passages), skip it. The prompt also has
    # a per-drawer guard, but coarse source-level skipping saves LLM
    # tokens and avoids edge cases where local quote density misleads.
    if max_quote_ratio < 1.0 and eligible:
        src_groups: dict = {}
        for doc_id, doc, meta in eligible:
            src = meta.get("source_file", "")
            src_groups.setdefault(src, []).append(doc)
        high_quote_sources = set()
        for src, docs in src_groups.items():
            ratio = _estimate_quote_ratio("\n".join(docs))
            if ratio > max_quote_ratio:
                high_quote_sources.add(src)
        if high_quote_sources:
            before = len(eligible)
            eligible = [
                (i, d, m) for (i, d, m) in eligible
                if m.get("source_file", "") not in high_quote_sources
            ]
            print(
                f"Skipped {before - len(eligible)} drawers from "
                f"{len(high_quote_sources)} source files with quote ratio "
                f"> {max_quote_ratio}"
            )

    if sample > 0:
        eligible = eligible[:sample]

    if not eligible:
        print(f"No eligible drawers in rooms {sorted(rooms_set)} (force={force}).")
        return {"processed": 0}

    print(
        f"Tagging {len(eligible)} drawers in rooms {sorted(rooms_set)} "
        f"via {cfg.endpoint} ({cfg.model})..."
    )
    if dry_run:
        print("DRY RUN — no changes will be written")

    processed = 0
    failed = 0
    neutral_count = 0
    total_input = 0
    total_output = 0

    for i, (doc_id, doc, meta) in enumerate(eligible, 1):
        if dry_run:
            preview = doc[:60].replace("\n", " ")
            print(
                f"  [{i}/{len(eligible)}] "
                f"{meta.get('wing')}/{meta.get('room')} — {preview}..."
            )
            continue

        parsed, usage = _call_llm_for_emotion(cfg, doc)
        if not parsed:
            failed += 1
            print(f"  [{i}/{len(eligible)}] ✗ {doc_id[:30]} — LLM failed")
            continue

        if usage:
            total_input += usage.get("prompt_tokens", 0)
            total_output += usage.get("completion_tokens", 0)

        emotion_meta = _normalize_parsed(parsed)
        # ChromaDB ``update`` overwrites the whole metadata dict — preserve
        # existing fields and layer the new ones on top.
        new_meta = {**meta, **emotion_meta}

        try:
            drawers_col.update(ids=[doc_id], metadatas=[new_meta])
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(eligible)}] ✗ {doc_id[:30]} — write failed: {e}")
            continue

        processed += 1
        if emotion_meta["intensity"] == 0:
            neutral_count += 1
            marker = "·"
            label_disp = "neutral"
        else:
            marker = "✓"
            label_disp = emotion_meta["emotion_label"] or "(no label)"
        print(
            f"  [{i}/{len(eligible)}] {marker} "
            f"{meta.get('wing')}/{meta.get('room')} "
            f"— intensity={emotion_meta['intensity']} label={label_disp!r}"
        )

    print(f"\nDone. {processed} tagged ({neutral_count} neutral), {failed} failed.")
    if total_input or total_output:
        print(f"Tokens: {total_input:,} in + {total_output:,} out (cost depends on provider)")

    return {
        "processed": processed,
        "failed": failed,
        "neutral": neutral_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
    }


def _parse_rooms(s: str) -> tuple:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Tag drawers with emotional metadata via a user-configured LLM "
        "(OpenAI-compatible API). Parallel to closet_llm.py."
    )
    parser.add_argument(
        "--palace",
        default=os.path.expanduser("~/.mempalace/palace"),
        help="Path to the palace",
    )
    parser.add_argument("--wing", default=None, help="Limit to one wing")
    parser.add_argument(
        "--rooms",
        type=_parse_rooms,
        default=DEFAULT_ROOMS,
        help=f"Comma-separated rooms to process (default: {','.join(DEFAULT_ROOMS)})",
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Only process first N drawers"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List work without calling the LLM"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Re-tag drawers even if they already match emotion_version={EMOTION_VERSION}",
    )
    parser.add_argument(
        "--max-quote-ratio",
        type=float,
        default=1.0,
        help="Skip source files whose blockquote-line ratio exceeds this "
        "threshold (e.g. 0.5 = drop files where >50%% of non-blank lines "
        "are blockquotes). Default 1.0 disables the filter. Useful for "
        "backfill of palaces with reading-companion data.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="LLM base URL (overrides $LLM_ENDPOINT)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="LLM bearer token (overrides $LLM_KEY)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (overrides $LLM_MODEL)",
    )
    args = parser.parse_args()

    cfg = LLMConfig(endpoint=args.endpoint, key=args.key, model=args.model)
    tag_emotions(
        args.palace,
        rooms=args.rooms,
        wing=args.wing,
        sample=args.sample,
        dry_run=args.dry_run,
        force=args.force,
        max_quote_ratio=args.max_quote_ratio,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
