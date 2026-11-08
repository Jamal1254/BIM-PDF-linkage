"""spec_matching.py

Modules M3 and M4 of the BIM compliance auditor.

M3  Requirement mapping (the ONLY step that uses an LLM)
    The LLM reads the specification text and, for each building-element
    category found in the IFC model, reports the requirement it finds
    (required thickness in mm, required materials, a verbatim evidence quote).
    It never judges compliance, and it is never shown the model's own values.
    The reply is forced into a fixed JSON schema and validated in Python.

M4  Matching and severity classification (pure Python, deterministic)
    Every element is compared numerically / by material tokens against the
    requirement for its category and assigned one severity class. Given the
    same requirement table, the output is identical on every run.

This file has no Streamlit dependency, so it can be unit-tested on its own.
"""

import json
import re
import unicodedata
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration (kept in one place so it can be reported for reproducibility)
# ---------------------------------------------------------------------------
LLM_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")  # stable; both free-tier listed
TEMPERATURE = None  # Gemini 3.x: do not override its recommended default
SEED = None         # no reproducibility guarantee; test empirically
THINKING_LEVEL = "low"
BATCH_SIZE = 25                      # categories per LLM request
MAX_REQUIRED_THICKNESS_MM = 2000.0   # anything above is treated as an LLM error

# rank -> (severity, issue type). Lower rank = more severe.
ISSUE_RANKS = {
    1: ("Critical", "Geometry & material missing"),
    2: ("High", "Geometry missing"),
    3: ("High", "Material missing"),
    4: ("Medium", "Geometry & material mismatch"),
    5: ("Medium", "Geometry mismatch"),
    6: ("Low", "Material mismatch"),
    7: ("Info", "Spec requirement not found"),
    8: ("Pass", "Compliant"),
}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Info", "Pass"]
SEVERITY_COLOURS = {
    "Critical": "#F28B82",
    "High": "#F6B26B",
    "Medium": "#FFD966",
    "Low": "#FFF2CC",
    "Info": "#D9D9D9",
    "Pass": "#B6E3B6",
}

# ---------------------------------------------------------------------------
# Prompts. Two wordings are kept so prompt sensitivity can be measured.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You extract construction-specification requirements from text. "
    "You never judge compliance and you never use information that is not "
    "written in the specification text provided. Reply with JSON only."
)

OUTPUT_FORMAT = (
    "OUTPUT: a single JSON object, exactly in this shape, with one entry per "
    "category in the same order as listed:\n"
    '{"requirements": [{"category_id": <int>, "spec_found": <true|false>, '
    '"required_thickness_mm": <number or null>, "required_materials": '
    '[<string>, ...], "clause_reference": <string>, "evidence_quote": <string>, '
    '"source_id": <string>}]}'
)


# Explanatory translations of IFC labels, NOT evidence that a requirement applies.
# Model widths/material values are not passed to the LLM.
IFC_TERM_HINTS = (
    ("mauerwerk", "masonry"),
    ("ziegel", "brickwork"),
    ("stein bestand", "existing stone masonry"),
    ("ortbeton", "cast-in-place concrete"),
    ("stb", "reinforced concrete"),
    ("trockenbau", "drywall / dry lining"),
    ("gk", "gypsum plasterboard"),
    ("daemm", "insulation"),
    ("dämm", "insulation"),
    ("wd", "thermal insulation (designation uncertain)"),
    ("beschichtung", "surface coating"),
    ("liftwand", "lift shaft wall"),
    ("maschendrahtzaun", "chain-link fence"),
    ("vorhang", "curtain"),
)


def _category_hints(key):
    name = unicodedata.normalize("NFKC", str(key)).casefold()
    return ", ".join(dict.fromkeys(hint for term, hint in IFC_TERM_HINTS if term in name)) or "not available"

PROMPT_VARIANTS = {
    "A": (
        "TASK: For each numbered building-element category below, find the "
        "requirement in the SPECIFICATION TEXT that applies to it and report "
        "it. Do NOT judge compliance. Do NOT use any value that is not written "
        "in the specification text.\n\n"
        "RULES:\n"
        "1. required_thickness_mm: the overall thickness (or width) in "
        "millimetres that the specification requires for that category. "
        "Convert cm or m to mm. Use null if no thickness is given for it.\n"
        "2. required_materials: material names the specification requires for "
        "that category, as named in the specification. Empty list if none.\n"
        "3. spec_found: true only if at least one clause applies to this "
        "category.\n"
        "4. evidence_quote: one short verbatim quote (max 200 characters) "
        "copied from the specification that supports your answer. Empty string "
        "if spec_found is false.\n"
        "5. clause_reference: clause number, heading or page marker if visible, "
        "otherwise an empty string.\n"
        "6. source_id: copy the exact [P14-C3]-style clause ID that "
        "contains the cited requirement. Never invent a source_id. "
        "If no source clause applies, source_id must be empty. "
        "7. Copy evidence_quote as a SHORT, CONTIGUOUS VERBATIM PASSAGE "
        "(ideally 10-25 words) from ONE [Page N] block, including the actual "
        "thickness/material words whenever possible. Do not paraphrase, "
        "translate, merge clauses, or rewrite units in evidence_quote. "
        "8. IFC categories may be single layers, coatings or standalone walls, "
        "while the specification may describe a COMPLETE multi-layer assembly. "
        "Never assign an assembly thickness to a single leaf merely because "
        "its name contains brick/concrete/plasterboard. "
        "9. English hints are translations of category labels ONLY; they do "
        "not prove applicability. External status 'internal/unspecified' is "
        "not confirmation of an internal or external use. "
        "10. If element function, construction assembly, or applicability is "
        "uncertain, set spec_found to false. Never guess."
    ),
    "B": (
        "Read the specification text and, for every category listed, record "
        "what the specification demands: the overall thickness in mm (convert "
        "from cm or m if needed) and the required materials, plus a short "
        "verbatim quote (under 200 characters) that proves it. Do not assess "
        "compliance. If the specification says nothing about a category, set "
        "spec_found to false, thickness to null and materials to an empty "
        "list. Only use facts stated in the text. Give the clause or page "
        "reference when it is visible. Also return the exact [P14-C3]-style "
        "source_id of the supporting paragraph. The evidence quote MUST be copied "
        "word-for-word as a contiguous passage from a single [Page N] block, "
        "not paraphrased. Do not apply complete wall-assembly requirements "
        "to isolated single-layer objects. If application is uncertain, "
        "report spec_found false."
    ),
}


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def _build_config(model, system_instruction, json_mode, max_tokens):
    """Build a GenerateContentConfig, degrading gracefully on older SDKs."""
    from google.genai import types

    kwargs = {
        "max_output_tokens": max_tokens,
        "system_instruction": system_instruction,
    }
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if model.startswith("gemini-3."):
        # Current Gemini 3 guidance: don't set temperature=0 or thinking_budget.
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    elif TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE
    if SEED is not None:
        kwargs["seed"] = SEED
    return types.GenerateContentConfig(**kwargs)


def _response_text(response):
    try:
        return response.text
    except Exception:
        return None


def call_llm(client, prompt, system_instruction=SYSTEM_INSTRUCTION,
             json_mode=True, max_tokens=8192, models=LLM_MODELS):
    """Call supported Gemini models, retrying only transient API failures.

    Invalid model names, bad API versions and authentication errors should
    fail fast instead of wasting time on identical unsuccessful requests.
    """
    errors = []
    for model in models:
        for attempt in range(2):
            try:
                config = _build_config(model, system_instruction, json_mode, max_tokens)
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                output = _response_text(response)
                if output and output.strip():
                    return output, model
                errors.append(f"{model}: empty response")
                break
            except Exception as exc:
                error = str(exc)
                errors.append(f"{model}: {error}")
                low = error.lower()
                fatal = any(code in low for code in
                    ("400", "401", "403", "404", "invalid_argument", "not_found", "unauthenticated"))
                retryable = not fatal and any(code in low for code in
                    ("429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "timeout"))
                if not retryable or attempt == 1:
                    break
                time.sleep(2)
    raise RuntimeError("All Gemini calls failed. " + " | ".join(errors[-4:]))


def parse_json_response(text):
    """Return the 'requirements' list from the model's JSON reply."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in the LLM response")
        data = json.loads(cleaned[start:end + 1])
    if isinstance(data, list):
        data = {"requirements": data}
    if not isinstance(data, dict) or not isinstance(data.get("requirements"), list):
        raise ValueError("JSON reply has no 'requirements' list")
    return data["requirements"]


# ---------------------------------------------------------------------------
# M3: requirement mapping
# ---------------------------------------------------------------------------
def _alnum(text):
    """Canonically compare PDF quotations despite spacing / ligatures.

    This is still a strict text check, NOT a fuzzy semantic match: extra or
    invented words never pass. NFKC handles PDF superscripts and ligatures.
    """
    normal = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^a-z0-9]+", "", normal)


def verify_source_quote(quote, context):
    """True only for a substantive continuous quotation found in source text.

    Page headers occur only between pages after app.py's retrieval fix.
    Requiring >= 15 canonical characters rejects meaningless placeholders.
    """
    needle = _alnum(quote)
    return len(needle) >= 15 and needle in _alnum(context or "")


def _source_blocks(context):
    """Resolve source IDs to exact PDF excerpts annotated by app.py.

    IDs are unique per input PDF page and bullet clause. If an LLM gives an
    invalid ID we do not accept its claim of evidence.
    """
    raw = context or ""
    pattern = list(re.finditer(r"\[(P[0-9]+-C[0-9]+)\]\s*", raw))
    blocks = {}
    for i, found in enumerate(pattern):
        end = pattern[i + 1].start() if i + 1 < len(pattern) else len(raw)
        value = raw[found.end():end].strip()
        # Do not treat the next page header as part of the last clause.
        value = re.split(r"\s*\[Page\s+[0-9]+\]", value, maxsplit=1)[0].strip()
        if value:
            blocks[found.group(1)] = value
    return blocks


def _source_has_thickness(text, thickness_mm):
    """Verify a stated overall thickness exists in the selected clause."""
    if thickness_mm is None:
        return False
    normal = unicodedata.normalize("NFKC", text).lower()
    for number, unit in re.findall(r"(?<![0-9])([0-9]+(?:[.,][0-9]+)?)\s*(mm|cm|metres|meters|m)\b", normal):
        factor = 1 if unit == "mm" else 10 if unit == "cm" else 1000
        if abs(float(number.replace(",", ".")) * factor - thickness_mm) < 0.05:
            return True
    return False


def _source_supported_materials(materials, clause_text):
    """Keep material names explicitly evidenced by the selected excerpt."""
    valid = []
    for material in materials:
        words = [w for w in re.findall(r"[a-z]{4,}", material.lower())
                 if w not in {"with", "type", "wall", "layer", "material"}]
        if words and all(w in clause_text.lower() for w in words):
            valid.append(material)
    return valid


def _source_quote_from_clause(source, thickness_mm):
    """Return real PDF wording, centred on the cited dimension when possible."""
    pos = 0
    if thickness_mm is not None:
        matches = list(re.finditer(r"\b[0-9]+(?:[.,][0-9]+)?\s*(?:mm|cm|m)\b", source, flags=re.I))
        for match in matches:
            if _source_has_thickness(match.group(0), thickness_mm):
                pos = max(0, match.start() - 25)
                break
    return source[pos:pos + 200].strip()


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _to_thickness(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0 or number > MAX_REQUIRED_THICKNESS_MM:
        return None
    return round(number, 2)


def empty_requirement(category_id, note=""):
    return {
        "category_id": category_id,
        "spec_found": False,
        "required_thickness_mm": None,
        "required_materials": [],
        "clause_reference": "",
        "evidence_quote": "",
        "source_id": "",
        "evidence_verified": False,
        "note": note,
    }


def empty_requirements(categories, note=""):
    return [empty_requirement(c["id"], note) for c in categories]


def normalise_requirements(items, batch_ids, spec_context):
    """Validate the LLM output. Returns {category_id: requirement_dict}."""
    out = {}
    available_sources = _source_blocks(spec_context)
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("category_id"))
        except (TypeError, ValueError):
            continue
        if cid not in batch_ids or cid in out:
            continue

        spec_found = _to_bool(item.get("spec_found"))
        materials = item.get("required_materials") or []
        if isinstance(materials, str):
            materials = [materials]
        clean_materials = []
        for m in materials if isinstance(materials, list) else []:
            m = str(m).strip()[:80]
            if m and m not in clean_materials:
                clean_materials.append(m)

        quote = str(item.get("evidence_quote") or "").strip()[:300]
        source_id = str(item.get("source_id") or "").strip().strip("[]")
        clause_text = available_sources.get(source_id)
        required_t = _to_thickness(item.get("required_thickness_mm"))
        note = ""
        if clause_text:
            # Use a verbatim excerpt recovered from our source text, not the
            # model's paraphrase. This removes the root cause of false quote
            # rejections while preserving source traceability.
            quote = _source_quote_from_clause(clause_text, required_t)
            verified = True
            if required_t is not None and not _source_has_thickness(clause_text, required_t):
                note = "Reported thickness not present in cited clause; thickness discarded"
                required_t = None
            # Materials not present in the cited clause cannot form a tested
            # requirement. Do not silently trust model-generated material lists.
            valid_materials = _source_supported_materials(clean_materials, clause_text)
            if len(valid_materials) != len(clean_materials):
                note = (note + "; " if note else "") + "Unverified materials excluded"
            clean_materials = valid_materials
        else:
            verified = verify_source_quote(quote, spec_context)
            if source_id and not verified:
                note = "Source clause ID was invalid and the quotation was not verified"
            if verified:
                # Legacy model responses that omit source_id are accepted only
                # when the quoted excerpt itself proves the reported details.
                if required_t is not None and not _source_has_thickness(quote, required_t):
                    note = (note + "; " if note else "") + "Thickness not supported by evidence quote"
                    required_t = None
                valid_materials = _source_supported_materials(clean_materials, quote)
                if len(valid_materials) != len(clean_materials):
                    note = (note + "; " if note else "") + "Unverified materials excluded"
                clean_materials = valid_materials

        req = {
            "category_id": cid,
            "spec_found": spec_found,
            "required_thickness_mm": required_t,
            "required_materials": clean_materials[:10],
            "clause_reference": str(item.get("clause_reference") or "").strip()[:120],
            "evidence_quote": quote,
            "source_id": source_id if clause_text else "",
            "evidence_verified": verified,
            "note": note,
        }
        if spec_found and not verified:
            # A model-generated requirement without traceable source evidence
            # must never be used to assert compliance or non-compliance.
            req["spec_found"] = False
            req["note"] = "Unsupported requirement rejected: evidence quote not found in supplied text"
        if spec_found and verified and required_t is None and not clean_materials:
            req["spec_found"] = False
            req["note"] = (req["note"] + "; " if req["note"] else "") + "No checkable requirement supported by cited clause"
        if not req["spec_found"]:
            req["required_thickness_mm"] = None
            req["required_materials"] = []
        out[cid] = req

    for cid in batch_ids:
        if cid not in out:
            out[cid] = empty_requirement(cid, "Not returned by the LLM")
    return out


def build_prompt(spec_context, batch, variant="A"):
    lines = []
    for c in batch:
        external = "Yes" if c.get("is_external") else "No"
        lines.append(
            f'ID {c["id"]}: name="{c["key"]}" | ifc_class={c["ifc_class"]} | '
            f"external={external} | english_label_hint={_category_hints(c['key'])}"
        )
    return (
        f"{PROMPT_VARIANTS[variant]}\n\n"
        "--- SPECIFICATION TEXT ---\n"
        f"{spec_context}\n\n"
        "--- CATEGORIES ---\n"
        + "\n".join(lines)
        + f"\n\n{OUTPUT_FORMAT}"
    )


def extract_requirements(client, spec_context, categories, variant="A"):
    """Map every category to a requirement using the LLM.

    Returns (requirements_in_category_order, metadata).
    The model's own thickness and material values are deliberately NOT sent,
    so the LLM cannot simply agree with the model.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")

    ordered, raw_responses, model_used, n_batches = [], [], None, 0
    for start in range(0, len(categories), BATCH_SIZE):
        batch = categories[start:start + BATCH_SIZE]
        batch_ids = {c["id"] for c in batch}
        prompt = build_prompt(spec_context, batch, variant)

        items, last_error = None, None
        for _attempt in range(2):
            text, model_used = call_llm(client, prompt)
            raw_responses.append(text)
            try:
                items = parse_json_response(text)
                break
            except ValueError as exc:
                last_error = exc
        if items is None:
            raise RuntimeError(
                f"Could not parse the LLM's JSON reply (batch {n_batches + 1}): {last_error}"
            )

        by_id = normalise_requirements(items, batch_ids, spec_context)
        ordered.extend(by_id[c["id"]] for c in batch)
        n_batches += 1

    meta = {
        "model": model_used,
        "prompt_variant": variant,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "thinking_level": THINKING_LEVEL,
        "batches": n_batches,
        "raw_responses": raw_responses,
    }
    return ordered, meta


# ---------------------------------------------------------------------------
# M4: deterministic matching and severity
# ---------------------------------------------------------------------------
_STOP_WORDS = {"and", "the", "with", "for", "type", "grade", "class",
               "standard", "min", "max", "minimum", "maximum"}
_SYNONYMS = {"blockwork": "block", "cmu": "block", "drywall": "plasterboard",
             "gypsum": "plasterboard", "masonry": "brick"}


def _tokens(text):
    out = set()
    for token in re.findall(r"[a-z0-9]+", str(text).lower()):
        token = _SYNONYMS.get(token, token)
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        token = _SYNONYMS.get(token, token)
        if len(token) >= 3 and token not in _STOP_WORDS and not token.isdigit():
            out.add(token)
    return out


def match_materials(required, model_names):
    """Return (matched, unmatched) required materials.

    A required material counts as found when every significant word of it
    appears somewhere in the model's material names (all layers pooled).
    """
    model_tokens = set()
    for name in model_names:
        model_tokens |= _tokens(name)
    matched, unmatched = [], []
    for req in required:
        req_tokens = _tokens(req)
        if not req_tokens:
            continue  # nothing checkable in this phrase
        (matched if req_tokens <= model_tokens else unmatched).append(req)
    return matched, unmatched


def _fmt(number):
    return f"{number:g}"


def evaluate_element(element, requirement, tolerance_mm):
    """Classify one IFC element. Pure function: same input -> same output."""
    requirement = requirement or {}
    spec_found = bool(requirement.get("spec_found"))
    req_t = requirement.get("required_thickness_mm") if spec_found else None
    req_m = list(requirement.get("required_materials") or []) if spec_found else []

    width = element.get("WidthMm")
    names = [n for n in (element.get("MaterialNames") or []) if n]
    notes = []
    delta = None
    matched, unmatched = [], []

    # --- geometry state --------------------------------------------------
    if width is None:
        geometry = "missing"
        if element.get("WidthStatus") == "unreliable":
            notes.append(
                "Geometry missing: model thickness is unreliable "
                f"({element.get('WidthRaw')} mm is implausible)"
            )
        else:
            notes.append("Geometry missing: no usable thickness in the model")
    elif req_t is None:
        geometry = "unassessable"
    else:
        delta = round(width - req_t, 2)
        if abs(delta) <= tolerance_mm + 1e-9:
            geometry = "ok"
        else:
            geometry = "mismatch"
            notes.append(
                f"Thickness {_fmt(width)} mm vs required {_fmt(req_t)} mm "
                f"(diff {delta:+g} mm, tolerance +/-{_fmt(tolerance_mm)} mm)"
            )

    # --- material state --------------------------------------------------
    if not names:
        material = "missing"
        notes.append("Material missing: no named material associated in the model")
    elif not req_m:
        material = "unassessable"
    else:
        matched, unmatched = match_materials(req_m, names)
        if not matched and not unmatched:
            material = "unassessable"
        elif unmatched:
            material = "mismatch"
            notes.append("Required material(s) not found in model: " + ", ".join(unmatched))
        else:
            material = "ok"

    # --- severity ----------------------------------------------------------
    if geometry == "missing" and material == "missing":
        rank = 1
    elif geometry == "missing":
        rank = 2
    elif material == "missing":
        rank = 3
    elif geometry == "mismatch" and material == "mismatch":
        rank = 4
    elif geometry == "mismatch":
        rank = 5
    elif material == "mismatch":
        rank = 6
    elif geometry == "ok" and material == "ok":
        rank = 8
    else:
        rank = 7

    severity, issue_type = ISSUE_RANKS[rank]
    if rank == 7:
        if not spec_found:
            notes.append("No applicable requirement found in the specification text")
        else:
            issue_type = "Partially assessable (spec silent)"
            notes.append("Specification does not state every checkable value for this category")
    if rank == 8:
        notes.append("Thickness within tolerance and required materials found")

    return {
        "SeverityRank": rank,
        "Severity": severity,
        "IssueType": issue_type,
        "Issue": "; ".join(notes),
        "GeometryState": geometry,
        "MaterialState": material,
        "DeltaMm": delta,
        "RequiredThicknessMm": req_t,
        "RequiredMaterials": req_m,
        "MatchedMaterials": matched,
        "UnmatchedMaterials": unmatched,
    }


# ---------------------------------------------------------------------------
# Reliability tests (repeatability and prompt sensitivity)
# ---------------------------------------------------------------------------
def _signature(req):
    materials = tuple(sorted({m.strip().lower() for m in req.get("required_materials", []) if m.strip()}))
    thickness = req.get("required_thickness_mm")
    thickness = None if thickness is None else round(float(thickness), 1)
    return (bool(req.get("spec_found")), thickness, materials)


def _describe(signature):
    found, thickness, materials = signature
    if not found:
        return "spec not found"
    thickness_text = "n/a" if thickness is None else f"{thickness:g} mm"
    return f"{thickness_text} | {', '.join(materials) or 'no materials'}"


def run_repeatability(client, spec_context, categories, n_runs=5, variant="A",
                      progress_cb=None):
    """Repeat the LLM mapping n_runs times on identical input.

    Agreement per category = share of runs that returned the most common
    answer (thickness, materials, spec_found).
    """
    runs = []
    for i in range(n_runs):
        reqs, _meta = extract_requirements(client, spec_context, categories, variant)
        runs.append([_signature(r) for r in reqs])
        if progress_cb:
            progress_cb(i + 1, n_runs)

    rows = []
    for idx, cat in enumerate(categories):
        counts = Counter(run[idx] for run in runs)
        modal, modal_count = counts.most_common(1)[0]
        rows.append({
            "Category": cat["key"],
            "Elements": cat.get("count", ""),
            "Most common answer": _describe(modal),
            "Agreement %": round(100.0 * modal_count / n_runs, 1),
            "Distinct answers": len(counts),
        })
    stable = sum(1 for r in rows if r["Distinct answers"] == 1)
    summary = {
        "runs": n_runs,
        "categories": len(rows),
        "fully_stable_pct": round(100.0 * stable / len(rows), 1) if rows else 100.0,
        "mean_agreement_pct": round(sum(r["Agreement %"] for r in rows) / len(rows), 1) if rows else 100.0,
        "prompt_variant": variant,
    }
    return {"table": rows, "summary": summary}


def compare_prompt_variants(client, spec_context, categories, variants=("A", "B"),
                            progress_cb=None):
    """Run each prompt wording once and compare the extracted requirements."""
    results = {}
    for i, variant in enumerate(variants):
        reqs, _meta = extract_requirements(client, spec_context, categories, variant)
        results[variant] = [_signature(r) for r in reqs]
        if progress_cb:
            progress_cb(i + 1, len(variants))

    first, second = variants[0], variants[1]
    rows, same = [], 0
    for idx, cat in enumerate(categories):
        a, b = results[first][idx], results[second][idx]
        is_same = a == b
        same += int(is_same)
        rows.append({
            "Category": cat["key"],
            f"Wording {first}": _describe(a),
            f"Wording {second}": _describe(b),
            "Same?": "Yes" if is_same else "NO",
        })
    summary = {
        "categories": len(rows),
        "identical_pct": round(100.0 * same / len(rows), 1) if rows else 100.0,
        "variants": list(variants),
    }
    return {"table": rows, "summary": summary}


# ---------------------------------------------------------------------------
# Optional narrative (the LLM only explains results that Python already decided)
# ---------------------------------------------------------------------------
NARRATIVE_SYSTEM = (
    "You are a senior BIM compliance auditor writing for a BIM coordinator. "
    "The severity classes below were computed by a deterministic program. "
    "Do not change, re-judge or add classes, and use only the numbers given."
)


def generate_narrative(client, results_text):
    prompt = (
        "Write a concise audit summary in Markdown with three short sections: "
        "1) Overall picture, 2) Main issues in order of severity, "
        "3) Recommended actions for the BIM coordinator (refer to category "
        "names).\n\n--- COMPUTED RESULTS ---\n" + results_text
    )
    text, _model = call_llm(
        client, prompt, system_instruction=NARRATIVE_SYSTEM,
        json_mode=False, max_tokens=4096,
    )
    return text
