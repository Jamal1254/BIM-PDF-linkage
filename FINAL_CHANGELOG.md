# Final change log — build v3

## Root cause established from RESULT 5(2).pdf
- PDF extraction returned 34 relevant pages and 89,065 characters without truncation.
- IFC extraction returned 103 wall elements in 33 groups.
- Gemini returned two valid JSON batches, but every category had `spec_found=false`.
- Python reported zero unsupported-requirement rejections, so the current failure occurred before evidence rejection.
- The UI showed `gemini-3.8-flash`, while run metadata recorded `gemini-3.5-flash-lite` as the model actually used.

## v3 changes
- Added deterministic category-specific candidate-clause retrieval.
- Added German/English construction-term aliases for common IFC category labels.
- Replaced the previous full-context prompt with category-specific candidate clauses.
- Kept exact source-ID resolution and evidence checks; no bypass or automatic `spec_found=true`.
- Reduced category batch size to 8.
- Added prompt revision and candidate-source metadata.
- Corrected model-label wording so primary/fallback configuration is not confused with the actual selected model.
- Retained verified coverage reporting and Neo4j update/read-back verification from the previous package.

## Verification
- Python syntax compilation: passed.
- Offline smoke/diagnostic tests: 15 passed.
- Live Gemini end-to-end verification: not performed in this environment; must be checked after deployment.
