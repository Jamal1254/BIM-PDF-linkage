import hashlib
import io
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from pypdf import PdfReader

import spec_matching as sm
from check_ifc_compliance import extract_all_ifc_wall_dimensions

st.set_page_config(page_title="BIM Hybrid Auditor", page_icon="🏗️", layout="wide")
st.title("🏗️ Automated BIM Model-vs-Specification Compliance Engine")
st.markdown(
    "Checks **IFC (3D BIM) elements** against requirements found in a **technical "
    "specification PDF**. An LLM is used only to *read the specification*; the "
    "comparison and the severity classification are done by deterministic Python "
    "code, and results are stored in a **Neo4j** graph. See the "
    "**Architecture & Method** tab for the module flow."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_secret(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# Specific wall-construction terms. Very common words ("wall", "mm") are left
# out on purpose: they match almost every paragraph and defeat the fallback.
SPEC_SEARCH_TERMS_BASE = [
    "thickness", "width", "cavity", "insulation", "external", "structural",
    "cladding", "u-value", "party wall", "separating wall",
    "building regulations", "fire rating", "blockwork", "masonry",
]
MIN_CHUNK_LEN = 40
MAX_CONTEXT_CHARS = 120000
MIN_MATCHED_CHUNKS = 3
BUILD_VERSION = "2026-11-08-compliance-diagnostics-v3"


def chunk_page_text(text):
    """Split page text into paragraphs (blank lines first, single lines as fallback)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def _fit_budget(parts, separator="\n\n"):
    """Join parts until the character budget is reached. Returns (text, truncated)."""
    kept, total = [], 0
    for part in parts:
        if total + len(part) > MAX_CONTEXT_CHARS and kept:
            return separator.join(kept), True
        kept.append(part)
        total += len(part) + len(separator)
    return separator.join(kept), False


def category_key(element):
    """Group repeated IFC instance labels under their shared construction type.

    Revit-style names often end in :<numeric instance ID>; that ID must not
    make every wall into a separate specification category.
    """
    name = str(element.get("Name") or "Unnamed")
    name = re.sub(r":\s*\d+\s*$", "", name).strip()
    exposure = "external" if element.get("IsExternal") else "internal/unspecified"
    return f"{name} ({element['Type']}; {exposure})"


def make_genai_client(api_key):
    """Use the beta endpoint for structured content/system instructions."""
    return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})


def diagnose_mapping(spec_context, truncated, has_client, llm_error, requirements):
    """Explain WHY a specification audit is incomplete, without guessing.

    This diagnostic does not determine compliance and never changes severity.
    It only summarizes observable extraction/mapping conditions for the user.
    """
    requirements = requirements or []
    stats = {
        "mapped_categories": len(requirements),
        "spec_found": sum(bool(r.get("spec_found")) for r in requirements),
        "evidence_verified": sum(bool(r.get("evidence_verified")) for r in requirements),
        "verified_checkable": sum(
            bool(r.get("spec_found") and r.get("evidence_verified")) and (
                r.get("required_thickness_mm") is not None
                or bool(r.get("required_materials"))
            ) for r in requirements
        ),
        "unsupported_rejected": sum(
            "unsupported requirement rejected" in str(r.get("note", "")).lower()
            for r in requirements
        ),
        "no_clause_found": sum(not bool(r.get("spec_found")) for r in requirements),
    }
    reasons = []
    actions = []
    if not spec_context:
        reasons.append("No usable specification text was extracted from the uploaded PDF.")
        actions.append("Check that the PDF contains selectable text and relevant clauses; scanned PDFs require OCR.")
    if not has_client:
        reasons.append("Gemini client was not available (API key missing).")
        actions.append("Configure GEMINI_API_KEY and run the API connection test.")
    if llm_error:
        reasons.append("Gemini requirement mapping failed: " + str(llm_error)[:900])
        actions.append("Run the 'Test Gemini API connection' button and check model access, quota, and the error message.")
    if truncated:
        reasons.append("Some matching specification paragraphs were omitted because the text exceeded the context budget.")
        actions.append("Use a focused specification PDF or raise the text budget after checking your free-tier token quota.")
    if spec_context and has_client and not llm_error and stats["verified_checkable"] == 0:
        if stats["unsupported_rejected"]:
            reasons.append(
                f"No verifiable thickness/material requirement was retained; "
                f"{stats['unsupported_rejected']} category requirement(s) were rejected because their evidence quotes were absent from the PDF text."
            )
            actions.append("Open 'Requirements the LLM extracted' and inspect Note, EvidenceQuote and EvidenceVerifiedInPDF.")
        elif stats["spec_found"] == 0:
            reasons.append("Gemini returned no applicable wall requirements for any category; this does not establish that the model is compliant.")
            actions.append("Check whether the specification actually states thickness/material requirements for these wall types and whether the type names match its terminology.")
        else:
            reasons.append("Some clauses were mapped but none includes both a verified quote and a usable thickness or material requirement.")
            actions.append("Inspect the extracted requirements table and compare the quoted clauses with the PDF.")
    return {"stats": stats, "reasons": reasons, "actions": list(dict.fromkeys(actions))}


def extract_spec_context(pdf_file, search_keyword):
    """Collect *complete pages* containing relevant construction language.

    The previous line/paragraph retrieval inserted `[Page N]:` inside clauses.
    Even a genuine quotation spanning two PDF lines then failed the exact
    evidence check. We now normalise line wrapping and add a page label only
    BEFORE each entire page. This preserves traceable contiguous quotations.

    Returns (context, number_of_relevant_pages, truncated). A truncation is
    reported rather than silently declaring a complete specification check.
    """
    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    keyword = (search_keyword or "").strip().lower()
    is_wall = "wall" in keyword
    wall_terms = (
        "external wall", "internal wall", "separating wall", "loadbearing wall",
        "masonry", "brickwork", "blockwork", "cavity", "plasterboard", "cladding",
    )
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if not raw.strip():
            continue
        # Join hard-wrapped PDF lines so source words remain adjacent during
        # evidence validation. Bullet characters still delimit clauses.
        clean = " ".join(raw.split())
        clean = re.sub(r"\s*[\uf0b7\u2022]\s*", "\n• ", clean)
        low = clean.lower()
        if is_wall:
            hits = sum(low.count(term) for term in wall_terms)
            if not hits:
                continue
            score = (
                10 * low.count("external wall")
                + 10 * low.count("internal wall")
                + 10 * low.count("separating wall")
                + 6 * low.count("loadbearing wall")
                + 3 * low.count("brickwork")
                + 3 * low.count("blockwork")
                + 2 * low.count("cavity")
                + low.count("wall")
                + hits
            )
        else:
            # Preserve the app's other element-keyword searches.
            hits = low.count(keyword) if keyword else 0
            if not hits:
                continue
            score = hits
        # Make the LLM select a real source-clause ID instead of inventing
        # evidence quotations. Each ID resolves to literal PDF-extracted text.
        clauses = [c.strip().lstrip("• ").strip() for c in clean.split("\n• ")]
        clauses = [c for c in clauses if c]
        annotated = "\n".join(
            f"[P{page_num}-C{i}] {clause}"
            for i, clause in enumerate(clauses, start=1)
        )
        pages.append((page_num, score, f"[Page {page_num}]\n{annotated}"))

    if not pages:
        return None, 0, False

    # Prioritise wall-specific pages rather than simply dropping the later
    # chapters when a larger PDF exceeds the model context budget.
    prioritized = sorted(pages, key=lambda entry: (-entry[1], entry[0]))
    selected, used = [], 0
    for page_num, score, text in prioritized:
        cost = len(text) + 2
        if used + cost <= MAX_CONTEXT_CHARS:
            selected.append((page_num, text))
            used += cost
    selected.sort(key=lambda entry: entry[0])
    context = "\n\n".join(text for _, text in selected)
    return context or None, len(selected), len(selected) < len(pages)


def build_results_dataframe(categories, cat_elements, req_by_id, tolerance_mm):
    rows = []
    for cat in categories:
        req = req_by_id[cat["id"]]
        for el in cat_elements[cat["key"]]:
            ev = sm.evaluate_element(el, req, tolerance_mm)
            rows.append({
                "Severity": ev["Severity"],
                "IssueType": ev["IssueType"],
                "Issue": ev["Issue"],
                "Category": cat["key"],
                "Name": el["Name"],
                "Type": el["Type"],
                "GlobalId": el["GlobalId"],
                "Storey": el["Storey"],
                "IsExternal": el["IsExternal"],
                "ModelThicknessMm": el.get("WidthMm"),
                "RequiredThicknessMm": ev["RequiredThicknessMm"],
                "DeltaMm": ev["DeltaMm"],
                "ThicknessSource": el.get("WidthSource", ""),
                "ModelMaterials": "; ".join(el.get("Materials", [])),
                "RequiredMaterials": "; ".join(ev["RequiredMaterials"]),
                "UnmatchedMaterials": "; ".join(ev["UnmatchedMaterials"]),
                "SpecClause": req.get("clause_reference", ""),
                "EvidenceVerified": bool(req.get("evidence_verified")),
                "SeverityRank": ev["SeverityRank"],
            })
    df = pd.DataFrame(rows)
    for col in ("ModelThicknessMm", "RequiredThicknessMm", "DeltaMm"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["SeverityRank", "Category", "GlobalId"]).reset_index(drop=True)


def build_requirements_dataframe(categories, req_by_id):
    rows = []
    for cat in categories:
        req = req_by_id[cat["id"]]
        rows.append({
            "Category": cat["key"],
            "Elements": cat["count"],
            "SpecFound": req["spec_found"],
            "RequiredThicknessMm": req["required_thickness_mm"],
            "RequiredMaterials": "; ".join(req["required_materials"]),
            "Clause": req["clause_reference"],
            "EvidenceQuote": req["evidence_quote"],
            "SourceID": req.get("source_id", ""),
            "EvidenceVerifiedInPDF": req["evidence_verified"],
            "Note": req.get("note", ""),
        })
    df = pd.DataFrame(rows)
    df["RequiredThicknessMm"] = pd.to_numeric(df["RequiredThicknessMm"], errors="coerce")
    return df


def style_severity(df):
    """Colour the Severity and IssueType cells by severity class."""
    def _row_style(row):
        colour = sm.SEVERITY_COLOURS.get(row["Severity"], "#FFFFFF")
        css = f"background-color: {colour}; color: #000000; font-weight: bold"
        return [css if col in ("Severity", "IssueType") else "" for col in row.index]
    return df.style.apply(_row_style, axis=1)


def build_excel(df, summary_df, req_df):
    """Colour-coded Excel workbook (needs openpyxl). Returns bytes or None."""
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            style_severity(df).to_excel(writer, sheet_name="Elements", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            req_df.to_excel(writer, sheet_name="SpecRequirements", index=False)
        return buffer.getvalue()
    except Exception:
        return None


def build_narrative_input(df, tolerance_mm):
    counts = df["Severity"].value_counts()
    lines = [
        f"Total elements: {len(df)}",
        f"Thickness tolerance: +/-{tolerance_mm:g} mm",
        "Severity counts: " + ", ".join(f"{s}={int(counts.get(s, 0))}" for s in sm.SEVERITY_ORDER),
        "",
        "Category | IssueType | Severity | Elements | Model thickness (mm) | Required thickness (mm)",
    ]
    grouped = (
        df.groupby(["SeverityRank", "Category", "IssueType", "Severity"], sort=True)
        .agg(n=("GlobalId", "count"), model=("ModelThicknessMm", "first"),
             required=("RequiredThicknessMm", "first"))
        .reset_index()
        .head(60)
    )

    def num(x):
        return "n/a" if pd.isna(x) else f"{x:g}"

    for _, r in grouped.iterrows():
        lines.append(
            f"{r['Category']} | {r['IssueType']} | {r['Severity']} | {r['n']} | "
            f"{num(r['model'])} | {num(r['required'])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit pipeline (runs once per button click; results go to session_state)
# ---------------------------------------------------------------------------
def run_audit_pipeline(ifc_file, pdf_file, keyword, tolerance_mm, gemini_key,
                       do_graph, clear_graph, do_narrative):
    audit = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "build_version": BUILD_VERSION,
        "ifc_name": ifc_file.name,
        "pdf_name": pdf_file.name,
        "keyword": keyword,
        "tolerance_mm": tolerance_mm,
        "warnings": [],
    }
    tmp_path = None
    try:
        with st.status("Running compliance audit...", expanded=True) as status:
            # M2: specification retrieval
            st.write("📄 Step 1/5: extracting specification clauses")
            try:
                spec_context, matched, truncated = extract_spec_context(pdf_file, keyword)
            except Exception as exc:
                spec_context, matched, truncated = None, 0, False
                audit["warnings"].append(f"Could not read the PDF text: {exc}")
            if spec_context is None:
                audit["warnings"].append(
                    "No specification clauses matching the search terms were found in the PDF "
                    "(scanned/image-only PDFs have no extractable text)."
                )
            if truncated:
                audit["warnings"].append(
                    "Specification text was longer than the context budget and was cut at a "
                    "paragraph boundary; later clauses were not sent to the LLM."
                )
            audit.update(spec_context=spec_context, matched_chunks=matched)

            # M1: IFC extraction
            st.write(f"📦 Step 2/5: extracting '{keyword}' elements from the IFC model")
            ifc_bytes = ifc_file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc") as tmp:
                tmp.write(ifc_bytes)
                tmp_path = tmp.name
            try:
                elements = extract_all_ifc_wall_dimensions(tmp_path, keyword)
            except Exception as exc:
                status.update(label="Audit stopped", state="error")
                st.error(f"Could not read the IFC file: {exc}")
                return None
            if not elements:
                status.update(label="Audit stopped", state="error")
                st.warning(f"No elements matching '{keyword}' were found in the IFC file.")
                return None

            cat_elements = defaultdict(list)
            for el in elements:
                cat_elements[category_key(el)].append(el)
            categories = []
            for cid, key in enumerate(sorted(cat_elements), start=1):
                items = cat_elements[key]
                categories.append({
                    "id": cid, "key": key, "ifc_class": items[0]["Type"],
                    "is_external": any(i["IsExternal"] for i in items),
                    "count": len(items),
                })
            audit.update(categories=categories, element_count=len(elements))

            # M3: LLM requirement mapping
            client = make_genai_client(gemini_key) if gemini_key else None
            llm_meta = {"model": None, "prompt_variant": "A",
                        "temperature": sm.TEMPERATURE, "seed": sm.SEED,
                         "thinking_level": sm.THINKING_LEVEL}
            if spec_context and client:
                st.write(f"🧠 Step 3/5: LLM maps {len(categories)} categories to specification requirements")
                try:
                    requirements, llm_meta = sm.extract_requirements(
                        client, spec_context, categories, variant="A")
                except Exception as exc:
                    audit["llm_error"] = str(exc)
                    requirements = sm.empty_requirements(categories, "LLM step failed")
            else:
                st.write("🧠 Step 3/5: skipped (no specification text or no API key)")
                requirements = sm.empty_requirements(categories, "No specification text")
            diagnostic = diagnose_mapping(
                spec_context, truncated, bool(client), audit.get("llm_error"), requirements
            )
            audit["mapping_diagnostic"] = diagnostic
            verified_checkable = diagnostic["stats"]["verified_checkable"]
            audit["verified_checkable_categories"] = verified_checkable
            audit["spec_context_truncated"] = truncated
            audit["compliance_ready"] = bool(
                spec_context and client and not audit.get("llm_error")
                and not truncated and verified_checkable > 0
            )
            if not audit["compliance_ready"]:
                audit["warnings"].append(
                    "INCOMPLETE AUDIT: severity results are only preliminary model-data "
                    "quality observations; specification compliance was NOT verified."
                )
            req_by_id = {r["category_id"]: r for r in requirements}
            # Report specification coverage explicitly: an audit can be usable
            # while only a subset of IFC categories has verified, checkable clauses.
            checkable_categories = [
                cat for cat in categories
                if (
                    req_by_id[cat["id"]].get("spec_found")
                    and req_by_id[cat["id"]].get("evidence_verified")
                    and (
                        req_by_id[cat["id"]].get("required_thickness_mm") is not None
                        or bool(req_by_id[cat["id"]].get("required_materials"))
                    )
                )
            ]
            covered_elements = sum(cat.get("count", 0) for cat in checkable_categories)
            total_categories = len(categories)
            audit["verified_checkable_categories"] = len(checkable_categories)
            audit["verified_checkable_elements"] = covered_elements
            audit["specification_coverage_pct"] = round(
                100.0 * len(checkable_categories) / total_categories, 1
            ) if total_categories else 0.0
            if audit["compliance_ready"] and len(checkable_categories) < total_categories:
                audit["warnings"].append(
                    f"PARTIAL SPECIFICATION COVERAGE: verified checkable requirements were found "
                    f"for {len(checkable_categories)} of {total_categories} categories "
                    f"({audit['specification_coverage_pct']}%), covering {covered_elements} "
                    f"of {len(elements)} elements. Other categories are not verified against "
                    f"a checkable specification requirement and must not be interpreted as compliant."
                )
            audit["llm_meta"] = llm_meta

            # M4: deterministic matching + severity
            st.write("⚖️ Step 4/5: deterministic matching and severity classification")
            df = build_results_dataframe(categories, cat_elements, req_by_id, tolerance_mm)
            req_df = build_requirements_dataframe(categories, req_by_id)
            assessment = (
                "Spec mapping complete" if audit["compliance_ready"]
                else "PRELIMINARY - specification not verified"
            )
            df.insert(0, "AssessmentStatus", assessment)
            req_df.insert(0, "AssessmentStatus", assessment)
            summary_df = (
                df.groupby(["SeverityRank", "Severity", "IssueType"]).size()
                .reset_index(name="Elements")
            )
            summary_df.insert(0, "AssessmentStatus", assessment)
            audit.update(
                df=df, req_df=req_df, summary_df=summary_df,
                excel_bytes=build_excel(df, summary_df, req_df),
            )

            # M5/M6: graph + narrative
            st.write("⛓️ Step 5/5: graph store and report")
            if do_graph and audit["compliance_ready"]:
                try:
                    from ifc_to_neo4j import IFCGraphMapper
                    model_id = hashlib.sha1(ifc_bytes).hexdigest()[:12]
                    graph_rows = [
                        {"gid": r.GlobalId, "severity": r.Severity, "issueType": r.IssueType,
                         "issue": r.Issue,
                         "requiredThicknessMm": None if pd.isna(r.RequiredThicknessMm) else float(r.RequiredThicknessMm),
                         "widthMm": None if pd.isna(r.ModelThicknessMm) else float(r.ModelThicknessMm)}
                        for r in df.itertuples()
                    ]
                    with IFCGraphMapper() as mapper:
                        if clear_graph:
                            mapper.clear_database()
                        stats = mapper.upload_ifc_to_graph(tmp_path, extracted=elements, model_id=model_id)
                        stats["audit_rows_written"] = mapper.write_audit_results(graph_rows)
                        if stats["audit_rows_written"] != len(graph_rows):
                            raise RuntimeError(
                                f"Neo4j matched {stats['audit_rows_written']} of {len(graph_rows)} "
                                "audit elements; graph write is incomplete."
                            )
                        audit["graph_readback"] = mapper.severity_summary(model_id=model_id)
                    audit["graph_stats"] = stats
                except Exception as exc:
                    audit["graph_error"] = str(exc)

            if do_graph and not audit["compliance_ready"]:
                audit["warnings"].append(
                    "Neo4j audit write skipped because specification mapping is incomplete."
                )

            if do_narrative and client and audit["compliance_ready"]:
                try:
                    audit["narrative"] = sm.generate_narrative(
                        client, build_narrative_input(df, tolerance_mm))
                except Exception as exc:
                    audit["warnings"].append(f"AI narrative failed: {exc}")

            status.update(
                label="Audit complete" if audit["compliance_ready"] else "Audit incomplete",
                state="complete" if audit["compliance_ready"] else "error",
                expanded=False,
            )
        return audit
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
DISPLAY_COLUMNS = [
    "AssessmentStatus", "Severity", "IssueType", "Issue", "Name", "Type", "GlobalId", "Storey",
    "ModelThicknessMm", "RequiredThicknessMm", "DeltaMm", "ThicknessSource",
    "ModelMaterials", "RequiredMaterials", "UnmatchedMaterials", "SpecClause",
    "EvidenceVerified",
]


def render_results(audit, gemini_key):
    df = audit["df"]

    if audit.get("llm_error"):
        st.error(
            "The LLM step failed, so no specification requirements were available. "
            "Only missing-data checks (geometry/material missing) are valid in this run. "
            f"Details: {audit['llm_error']}"
        )
    for warning in audit.get("warnings", []):
        st.warning(warning)

    message = (
        f"{audit['element_count']} '{audit['keyword']}' elements in "
        f"{len(audit['categories'])} groups processed "
        f"(tolerance ±{audit['tolerance_mm']:g} mm)."
    )
    if audit.get("compliance_ready"):
        st.success("✅ Specification-based audit complete. " + message)
        st.subheader("📐 Verified specification coverage")
        total_categories = len(audit.get("categories", []))
        verified_categories = int(audit.get("verified_checkable_categories", 0))
        covered_elements = int(audit.get("verified_checkable_elements", 0))
        coverage_pct = float(audit.get("specification_coverage_pct", 0.0))
        cov1, cov2, cov3 = st.columns(3)
        cov1.metric("Verified checkable categories", f"{verified_categories}/{total_categories}")
        cov2.metric("Category coverage", f"{coverage_pct:.1f}%")
        cov3.metric("Elements in covered categories", f"{covered_elements}/{audit.get('element_count', 0)}")
        st.caption(
            "Coverage measures categories with an evidence-verified thickness or material "
            "requirement. It is not a compliance/pass percentage; categories without a "
            "checkable clause remain unverified against the specification."
        )
    else:
        st.error("❌ SPECIFICATION COMPLIANCE NOT ASSESSED. " + message)
        diag = audit.get("mapping_diagnostic") or {}
        stats = diag.get("stats") or {}
        st.subheader("🔍 Why the specification audit is incomplete")
        for reason in diag.get("reasons", []):
            st.warning(reason)
        if not diag.get("reasons"):
            st.warning("Mapping completion criteria were not met. Review the Run metadata and extracted requirements below.")
        st.caption(
            f"Groups returned: {stats.get('mapped_categories', 0)} | "
            f"Applicable clauses: {stats.get('spec_found', 0)} | "
            f"Verified quotes: {stats.get('evidence_verified', 0)} | "
            f"Verified checkable groups: {stats.get('verified_checkable', 0)} | "
            f"Rejected unsupported: {stats.get('unsupported_rejected', 0)}"
        )
        for action in diag.get("actions", []):
            st.info(action)

    # --- severity overview ---------------------------------------------------
    counts = df["Severity"].value_counts()
    cols = st.columns(len(sm.SEVERITY_ORDER))
    for col, severity in zip(cols, sm.SEVERITY_ORDER):
        col.metric(severity, int(counts.get(severity, 0)))
    st.markdown(
        " ".join(
            f"<span style='background:{sm.SEVERITY_COLOURS[s]};color:#000;padding:2px 8px;"
            f"border-radius:4px;font-size:0.85em'>{s}</span>"
            for s in sm.SEVERITY_ORDER
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Critical = geometry & material missing · High = geometry or material missing · "
        "Medium = geometry mismatch (with or without material mismatch) · "
        "Low = material mismatch only · Info = specification "
        "silent / not assessable · Pass = compliant."
    )

    # --- table + export --------------------------------------------------------
    st.subheader("📋 Element results (colour-coded by severity)")
    selected = st.multiselect(
        "Filter by severity", sm.SEVERITY_ORDER, default=sm.SEVERITY_ORDER,
        key="severity_filter",
    )
    view = df[df["Severity"].isin(selected)][DISPLAY_COLUMNS + ["SeverityRank"]]
    if view.shape[0] * view.shape[1] <= 150000:
        st.dataframe(style_severity(view), hide_index=True)
    else:
        st.info("Table is large, so colouring is disabled on screen. The Excel export is still colour-coded.")
        st.dataframe(view, hide_index=True)

    export_prefix = (
        "compliance_results_by_severity" if audit.get("compliance_ready")
        else "PRELIMINARY_model_data_only"
    )
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Download results (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{export_prefix}.csv",
        mime="text/csv",
    )
    if audit.get("excel_bytes"):
        c2.download_button(
            "⬇️ Download colour-coded Excel",
            data=audit["excel_bytes"],
            file_name=f"{export_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        c2.caption("Excel export unavailable (openpyxl not installed). Use the CSV.")

    # --- transparency ----------------------------------------------------------
    with st.expander("🔎 Requirements the LLM extracted from the specification (per category)"):
        st.caption(
            "EvidenceVerifiedInPDF = the quoted evidence was found in the PDF text by a "
            "programmatic check. False means the LLM paraphrased or may have invented it."
        )
        st.dataframe(audit["req_df"], hide_index=True)
    with st.expander(f"📄 Specification text sent to the LLM ({audit['matched_chunks']} relevant pages)"):
        st.text(audit.get("spec_context") or "No specification text available.")
    with st.expander("🧾 Run metadata (for reproducibility)"):
        meta = audit.get("llm_meta", {})
        st.json({
            "build_version": BUILD_VERSION,
            "api_version": "v1beta",
            "started": audit["started"],
            "ifc_file": audit["ifc_name"],
            "pdf_file": audit["pdf_name"],
            "search_keyword": audit["keyword"],
            "tolerance_mm": audit["tolerance_mm"],
            "llm_model_used": meta.get("model"),
            "prompt_variant": meta.get("prompt_variant"),
            "prompt_revision": meta.get("prompt_revision"),
            "temperature": meta.get("temperature"),
            "seed": meta.get("seed"),
            "thinking_level": meta.get("thinking_level"),
            "llm_batches": meta.get("batches"),
            "prompt_strategy": meta.get("prompt_strategy"),
            "candidate_clause_counts": meta.get("candidate_clause_counts"),
            "candidate_clause_ids": meta.get("candidate_clause_ids"),
            "assessment_complete": bool(audit.get("compliance_ready")),
            "evidence_verified_checkable_categories": audit.get("verified_checkable_categories"),
            "evidence_verified_checkable_elements": audit.get("verified_checkable_elements"),
            "specification_category_coverage_pct": audit.get("specification_coverage_pct"),
            "spec_context_characters": len(audit.get("spec_context") or ""),
            "spec_context_truncated": audit.get("spec_context_truncated"),
            "mapping_diagnostic": audit.get("mapping_diagnostic"),
            "llm_error": audit.get("llm_error"),
        })
    if audit.get("llm_meta", {}).get("raw_responses"):
        with st.expander("🧪 Raw LLM JSON replies"):
            for i, raw in enumerate(audit["llm_meta"]["raw_responses"], 1):
                st.text(f"--- reply {i} ---\n{raw[:20000]}")

    # --- graph ------------------------------------------------------------------
    if audit.get("graph_stats"):
        st.success(f"⛓️ Neo4j synchronised: {audit['graph_stats']}")
        if audit.get("graph_readback"):
            st.caption("Severity counts read back from the Neo4j graph (round-trip check):")
            st.dataframe(pd.DataFrame(audit["graph_readback"]), hide_index=True)
    elif audit.get("graph_error"):
        st.warning(f"⚠️ Neo4j sync skipped: {audit['graph_error']}")

    # --- narrative --------------------------------------------------------------
    if audit.get("narrative"):
        st.subheader("🧠 AI summary (explains the computed results only)")
        st.markdown(audit["narrative"])

    render_reliability_tests(audit, gemini_key)


def render_reliability_tests(audit, gemini_key):
    st.divider()
    st.subheader("🧪 Reliability tests: repeatability and prompt sensitivity")
    st.caption(
        "Only the LLM mapping step (M3) can vary between runs; everything after it is "
        "deterministic. These tests measure that variation on the current inputs."
    )
    if not (gemini_key and audit.get("spec_context") and audit.get("compliance_ready")):
        st.info("First complete a successful specification mapping run; reliability tests are disabled for incomplete audits.")
        return

    n_runs = st.number_input("Repeat runs", min_value=2, max_value=10, value=5, step=1)
    b1, b2 = st.columns(2)
    run_repeat = b1.button("Run repeatability test")
    run_prompt = b2.button("Compare prompt wordings (A vs B)")

    if run_repeat or run_prompt:
        client = make_genai_client(gemini_key)
        bar = st.progress(0.0)

        def progress(done, total):
            bar.progress(done / total)

        try:
            if run_repeat:
                audit["repeatability"] = sm.run_repeatability(
                    client, audit["spec_context"], audit["categories"],
                    n_runs=int(n_runs), variant="A", progress_cb=progress)
            if run_prompt:
                audit["prompt_compare"] = sm.compare_prompt_variants(
                    client, audit["spec_context"], audit["categories"], progress_cb=progress)
        except Exception as exc:
            st.error(f"Reliability test failed: {exc}")
        bar.empty()

    if audit.get("repeatability"):
        res = audit["repeatability"]
        s = res["summary"]
        st.markdown(f"**Repeatability** ({s['runs']} identical runs, wording {s['prompt_variant']})")
        m1, m2 = st.columns(2)
        m1.metric("Categories with identical answer in every run", f"{s['fully_stable_pct']}%")
        m2.metric("Mean agreement with the most common answer", f"{s['mean_agreement_pct']}%")
        st.dataframe(pd.DataFrame(res["table"]), hide_index=True)
    if audit.get("prompt_compare"):
        res = audit["prompt_compare"]
        st.markdown("**Prompt sensitivity** (same input, two different wordings)")
        st.metric("Categories with identical extracted requirement", f"{res['summary']['identical_pct']}%")
        st.dataframe(pd.DataFrame(res["table"]), hide_index=True)


# ---------------------------------------------------------------------------
# Architecture tab
# ---------------------------------------------------------------------------
ARCHITECTURE_DOT = r"""
digraph G {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  edge [fontname="Helvetica", fontsize=9];
  ifc [label="IFC model\n(.ifc)", fillcolor="#E8F0FE"];
  pdf [label="Specification\n(.pdf)", fillcolor="#E8F0FE"];
  m1 [label="M1 IFC extraction\nIfcOpenShell, 4-tier width", fillcolor="#D9EAD3"];
  m2 [label="M2 Spec retrieval\npypdf + keyword clauses", fillcolor="#D9EAD3"];
  m3 [label="M3 Requirement mapping\nLLM to JSON, low thinking", fillcolor="#FCE5CD"];
  m4 [label="M4 Matching + severity\ndeterministic Python", fillcolor="#D9EAD3"];
  m5 [label="M5 Reporting\ncolour table, CSV, XLSX", fillcolor="#D9EAD3"];
  g [label="M6 Neo4j graph\nelements + audit results", fillcolor="#EAD1DC"];
  n [label="Optional AI summary\n(explains only)", fillcolor="#FCE5CD"];
  ifc -> m1;
  pdf -> m2;
  m1 -> m3 [label="category list\n(model values withheld)"];
  m2 -> m3 [label="clause text"];
  m1 -> m4 [label="element data"];
  m3 -> m4 [label="validated requirements"];
  m4 -> m5;
  m4 -> g [label="severity per element"];
  m1 -> g;
  m5 -> n [style=dashed];
}
"""


def render_architecture_tab():
    st.header("Module flow")
    st.graphviz_chart(ARCHITECTURE_DOT)
    st.caption("Orange = uses an LLM. Green = deterministic code. Pink = database.")
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        text = re.sub(r"```mermaid.*?```", "", text, flags=re.S)
        st.markdown(text)
    else:
        st.info("README.md not found next to app.py.")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
tab_audit, tab_arch = st.tabs(["🔎 Compliance Audit", "🧭 Architecture & Method"])

with tab_arch:
    render_architecture_tab()

with tab_audit:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("⚙️ Audit Parameters")
        configured_key = get_secret("GEMINI_API_KEY")
        if configured_key:
            st.success("🔒 API key loaded from server settings")
            gemini_key = str(configured_key).strip()
        else:
            raw_key = st.text_input("🔑 Paste Gemini API Key", type="password",
                                    help="Only needed if no key is configured on the server.")
            gemini_key = raw_key.strip() if raw_key else ""

        st.caption(
            f"Build: {BUILD_VERSION} | API: v1beta | "
            f"Primary model: {sm.LLM_MODELS[0]} | "
            f"Fallback(s): {', '.join(sm.LLM_MODELS[1:]) or 'none'} | "
            "Actual selected model is reported in Run metadata."
        )
        if st.button("🔌 Test Gemini API connection", help="One short API request, without reading your IFC/PDF"):
            if not gemini_key:
                st.error("Configure GEMINI_API_KEY in Streamlit secrets or enter it above.")
            else:
                try:
                    client = make_genai_client(gemini_key)
                    _text, model = sm.call_llm(
                        client, "Reply with the single word OK.",
                        system_instruction="You are testing API connectivity.",
                        json_mode=False, max_tokens=512,
                        models=sm.LLM_MODELS,
                    )
                    st.success(f"Gemini connection successful ({model}).")
                except Exception as exc:
                    st.error(f"Gemini connection test failed: {exc}")

        st.subheader("📁 1. Upload 3D BIM Model")
        ifc_file = st.file_uploader("Upload .ifc file", type=["ifc"])
        search_keyword = st.text_input("BIM Component Type to Audit", value="Wall")

        st.subheader("📄 2. Upload Technical Specification")
        pdf_file = st.file_uploader("Upload Contract/Standard PDF", type=["pdf"])

        st.subheader("🔧 3. Options")
        tolerance_mm = st.number_input(
            "Thickness tolerance (± mm)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        do_graph = st.checkbox("Synchronise to Neo4j graph", value=False)
        clear_graph = st.checkbox(
            "Clear the whole Neo4j database first", value=False,
            help="Deletes ALL nodes in the target database before loading this model.")
        do_narrative = st.checkbox("Add AI narrative summary", value=False)

        st.divider()
        run_audit = st.button("🚀 Run Batch Compliance Audit")

    with col2:
        st.header("📊 Live Compliance Audit Report")

        if run_audit:
            if not ifc_file or not pdf_file:
                st.error("⚠️ Please upload both an IFC file and a specification PDF.")
            elif not gemini_key:
                st.error("⚠️ A Gemini API key is required.")
            elif not search_keyword.strip():
                st.error("⚠️ Enter a component type to audit, for example Wall.")
            else:
                result = run_audit_pipeline(
                    ifc_file, pdf_file, search_keyword.strip(), float(tolerance_mm),
                    gemini_key, do_graph, clear_graph, do_narrative)
                if result is not None:
                    st.session_state["audit"] = result
                else:
                    st.session_state.pop("audit", None)

        if st.session_state.get("audit"):
            render_results(st.session_state["audit"], gemini_key)
        elif not run_audit:
            st.info("Upload an IFC model and a specification PDF, then run the audit.")
