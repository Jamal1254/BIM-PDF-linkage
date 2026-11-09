# BIM Hybrid Auditor — targeted mapping fix

**Build identifier:** `2026-11-08-compliance-diagnostics-v3`  
**Prompt revision:** `candidate-retrieval-v2`

## What changed

- M2 now ranks source-identified PDF clauses separately for each IFC category using German/English construction terminology.
- M3 receives a smaller category-specific set of candidate clauses instead of the full 89,000-character specification context in every request.
- The prompt distinguishes a shared material/number from a defensible construction-type match.
- The existing Python source-ID and evidence checks remain in place. Unsupported requirements are still rejected.
- LLM batch size is reduced to 8 categories per request.
- Run metadata now records the prompt revision, actual selected Gemini model, and candidate clause IDs/counts.
- The interface labels the configured primary model and fallback models separately; the actual selected model is reported in run metadata.
- The previous coverage reporting and Neo4j write/read-back checks are retained.

## Deploy

1. Open the same GitHub repository, branch and Streamlit entrypoint directory used by your live app.
2. Replace the files in that directory with the files from this ZIP. Keep the filenames exactly as shown.
3. Commit the changes to the same branch and wait for Streamlit to redeploy. Then reboot/reload the app.
4. Confirm the build identifier is `2026-11-08-compliance-diagnostics-v3`.
5. Confirm the API key is still configured in Streamlit Secrets as `GEMINI_API_KEY`. Do not put the key in GitHub or send it in screenshots.
6. For the first live test, upload the same IFC and specification PDF. Keep AI narrative OFF, Neo4j synchronization OFF, and database clearing OFF.
7. Run one audit. Expand **Run metadata** and confirm:
   - `prompt_revision` = `candidate-retrieval-v2`
   - `prompt_strategy` = `category-specific clause retrieval`
   - candidate clause counts are present for categories
   - the actual model is recorded in `llm_model_used`
8. Review the per-category requirement table. Check each returned source ID and quote against the displayed PDF clause. Do not treat the audit as complete unless the application says **Specification-based audit complete** and at least one verified checkable requirement exists.

## Reliability experiments and thesis reporting

The previously recorded repeatability and A/B prompt results (93.9% fully stable categories, 97.0% mean agreement, and 93.9% A/B agreement) were obtained using the earlier full-context prompt. They do **not** measure `candidate-retrieval-v2`.

If this revised prompt is used in the submitted application, rerun the repeatability and A/B tests on the final prompt revision and record the revision/model/input settings with the results. If time or API quota does not permit rerunning, report the earlier figures explicitly as results from the previous prompt version, not the final one.

## Offline verification

The package passed Python syntax compilation and all 15 offline smoke/diagnostic tests:

```bash
python -m py_compile app.py spec_matching.py test_smoke.py
python -m unittest test_smoke -v
```

These tests do not replace the required live Gemini + IFC + PDF test. No API key or secrets are included in this package.
