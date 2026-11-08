# Gemini API hotfix — deployment instructions

**Build identifier:** `2026-11-08-gemini-v1beta-hotfix`

This package fixes the observed v1 API incompatibility, unavailable 2.0 Flash fallback, unnecessary duplicate narrative calls, and excessive "wall" retrieval that had truncated the specification. It keeps the thesis system's original design, severity classifications, traceable evidence, and comparison tools. It does not contain your API key.

## Streamlit Community Cloud + GitHub (most likely deployment)

1. Open the GitHub repository and branch configured under the Streamlit Community Cloud app's settings. Find the existing `app.py` and `spec_matching.py` at the same path Streamlit uses as its entrypoint. **Replace them there**, not in a newly created folder.
2. Also replace `README.md`, `check_ifc_compliance.py`, `ifc_to_neo4j.py`, and `requirements.txt` from this ZIP. These files are at the ZIP root, not in a nested `bim_auditor_submission` directory.
3. Commit the changes to that *same* branch; Streamlit may auto-redeploy. Under Streamlit Cloud, select **Reboot app** (or redeploy) to clear the old running process and refresh dependencies.
4. Open/reload the app. Confirm the left panel shows: **`Build: 2026-11-08-gemini-v1beta-hotfix | API: v1beta | Model: gemini-3.8-flash`**. If you cannot see that text, the app is still running other files/branch. Check the Streamlit app entrypoint path and GitHub branch.
5. Check the configured key in Streamlit **App settings > Secrets**: `GEMINI_API_KEY = "YOUR_KEY"`. Never commit your key to GitHub or share it in screenshots.
6. Click **Test Gemini API connection** before uploading the IFC or PDF. A successful test confirms the configured model/key/endpoint can generate content. If it fails, capture the *new* exact error text (do not reveal the key).
7. Keep **Add AI narrative summary**, **Synchronise to Neo4j**, and **Clear the whole Neo4j database first** unchecked for your first audit. Upload the original `Project1.ifc` and specification PDF and run.
8. The report must show **Specification-based audit complete** before the exported workbook is evidence of specification compliance. If the text was truncated or no verified applicable requirements were extracted, the app correctly labels the run **incomplete**, even if IFC elements were extracted.
9. Only after a successful audit, optionally run **Repeatability (2 runs)** and **Compare prompt wordings**; capture the real results for the professor. These use additional API requests and free-tier quota.

## Local Streamlit

Replace the six project files (same directory as the `app.py` you launch), then run:

```bash
python -m pip install -r requirements.txt --upgrade
python -m streamlit run app.py
```

## Key changes in code

- `app.py`: `make_genai_client()` explicitly specifies `http_options={"api_version": "v1beta"}`; adds a connection-test button and build label; limits generic keyword matching; narrative defaults off.
- `spec_matching.py`: `LLM_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")`; removes temperature=0 for Gemini 3 and replaces legacy thinking budget with `thinking_level="low"`; retries transient errors only.
- `requirements.txt`: uses a newer Google Gen AI Python SDK compatible with Gemini 3.

## Verification status

The revised source code passes Python syntax compilation and the included offline smoke tests. **Live Gemini/IFC/PDF/Neo4j end-to-end testing has not been performed** in this workspace because the API key and original model/spec files were not provided. The free tier has model-specific and per-project request limits; a new 429 error means quota rather than the old 400/404 bug.
