BIM HYBRID AUDITOR - GRAPH-BASED RETRIEVAL UPGRADE

BASE VERSION
This package is based on the advanced application previously shared as
app (8)(1). It preserves the M1-M5 extraction, requirement mapping, evidence
checks, deterministic matching, severity reporting, exports, repeatability test
and Prompt A/B comparison. The simpler app (8)(2) was treated as the earlier
prototype/reference, not as the replacement base.

FILES TO REPLACE
1. app.py
2. ifc_to_neo4j.py
3. requirements.txt (adds openpyxl for the existing Excel export)

KEEP UNCHANGED
- check_ifc_compliance.py
- spec_matching.py

WHAT THE GRAPH RETRIEVAL ADDITION DOES
1. Extracts page-addressable PDF clause chunks with stable source IDs such as
   P12-C3.
2. Stores the selected IFC elements, their names/types/material terms, and the
   specification clauses in Neo4j.
3. Links IFC elements and specification clauses through shared GraphTerm nodes.
4. Uses a Cypher graph traversal from selected IFC walls to connected clauses.
5. Reconstructs the retrieved clauses in the source-ID format already used by
   spec_matching.py, so its existing evidence-quote validation remains active.
6. Supplies only graph-retrieved clauses to Gemini for M3 requirement mapping.
7. Retains the existing deterministic M4 matching/severity logic, reports,
   repeatability test and Prompt A/B comparison.
8. Writes verified audit results to the same project graph and scopes severity
   readback to the current model/audit fingerprint.

DATABASE SAFETY
- The UI option that deleted the entire Neo4j database has been removed.
- The application now upserts project/audit graph records instead of clearing
  all nodes before each run.
- Neo4j credentials are read only from Streamlit secrets or environment variables.
- The previous uploaded older ifc_to_neo4j.py contained a hard-coded password.
  Rotate that password in Neo4j Aura before deployment. Do not put credentials
  in source code, GitHub, or chat.

STREAMLIT SECRETS
Configure:
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
NEO4J_URI = "neo4j+s://YOUR_INSTANCE.databases.neo4j.io"
NEO4J_USERNAME = "YOUR_USERNAME"
NEO4J_PASSWORD = "YOUR_NEW_PASSWORD"
NEO4J_DATABASE = "YOUR_DATABASE_NAME"  # optional if using server default

TECHNICAL SCOPE AND LIMITATIONS
- This is an initial graph-based retrieval implementation using shared terms
  and Cypher traversal. It does not yet use vector embeddings or semantic
  similarity search.
- It is intended to improve evidence grounding and traceability; it cannot
  guarantee zero hallucinations.
- Existing spec_matching.py checks that the source ID and quoted evidence can
  be resolved against the retrieved source text. This is stronger than trusting
  free-form model output, but it does not prove every interpretation is correct.
- If GraphRAG is enabled and Neo4j retrieval fails or returns no clauses, M3 is
  not allowed to proceed using ungrounded requirements; the audit is marked
  incomplete. Disable GraphRAG only when intentionally running the direct-text
  baseline for comparison.
- The existing IFC keyword selector and extraction logic are unchanged in this
  package. This upgrade is focused on Neo4j-to-Gemini graph retrieval.
- The existing BIMElement nodes use IFC GlobalId as their unique key. If the
  same GlobalIds are reused across different IFC model versions, the existing
  graph schema may update those shared element properties; a future multi-model
  history design would use project-scoped element nodes.

DEPLOYMENT
1. Back up the current advanced app.py and ifc_to_neo4j.py.
2. Rotate the previously exposed Neo4j password and update Streamlit secrets.
3. Replace app.py, ifc_to_neo4j.py and requirements.txt with the files here.
4. Keep check_ifc_compliance.py and spec_matching.py unchanged.
5. Commit/push and reboot Streamlit.
6. Run one audit with the same IFC and PDF.
7. Confirm the report shows a non-zero graph clause count and open the
   GraphRAG retrieval details expander to inspect page IDs and source text.
8. Only then assess Gemini's mapped requirements and the final compliance table.

TEST STATUS
- Python syntax compilation passed for app.py, ifc_to_neo4j.py,
  check_ifc_compliance.py and spec_matching.py.
- Offline helper checks passed for page/source-ID reconstruction, compatibility
  with spec_matching.py source-ID parsing, and IFC term normalization.
- Static checks confirmed the M3 graph-retrieval call is present and whole-
  database deletion is no longer called or exposed in the UI.
- A live Neo4j/Gemini audit has NOT been performed in this environment.
