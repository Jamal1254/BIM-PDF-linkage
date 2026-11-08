# BIM Auditor — incomplete-audit diagnosis patch

This patch does **not** mark unverified elements as compliant or change your IFC
extraction, severity calculation, Neo4j semantics, or Gemini configuration.

It adds an explicit `Why the specification audit is incomplete` section, with:
- Whether the LLM failed, the PDF supplied no relevant text, or text was truncated
- Number of applicable category requirements, verified quotes, and usable checks
- Number of requirements rejected due to evidence quotes missing from the PDF
- The next action specific to the actual reason

## Deploy

Replace **app.py** from this archive in your deployed Streamlit repository and
restart the app. Replacing the remaining files is optional if you already
applied the previous Gemini API hotfix.

The build label in the app must say:
`2026-11-08-compliance-diagnostics-v2`

1. Click **Test Gemini API connection**. Save the exact error if it fails.
2. Upload the IFC and specification PDF, with AI narrative disabled and Neo4j
   synchronization disabled.
3. Run audit. Scroll to **Why the specification audit is incomplete** and use
   the displayed diagnostics.
4. Expand **Requirements the LLM extracted** and **Run metadata**. These show
   whether your spec contained relevant requirements, whether evidence was
   confirmed, and whether clauses were truncated.

If `Verified checkable groups` remains 0, obtain a specification containing
explicit wall thickness or material criteria. Code cannot invent them safely.

Tests: `python -m unittest test_smoke -v`.

**Note:** A preliminary model-data report should never be represented as a
verified model-vs-specification compliance result.
