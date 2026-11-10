"""ifc_to_neo4j.py

Module M6: Neo4j graph store.

Writes the IFC model (storeys, elements, materials, containment) to Neo4j and
stores the audit result on each element node, so results can be queried with
Cypher afterwards, e.g.
    MATCH (e:BIMElement {severity: 'Critical'}) RETURN e.name, e.issue

Credentials are read ONLY from Streamlit secrets or environment variables.
Never put them in the source code or commit them to GitHub.
"""

import os
import re
import unicodedata

import ifcopenshell
import ifcopenshell.util.element
import streamlit as st
from neo4j import GraphDatabase

from check_ifc_compliance import (
    WIDTH_KEYWORDS,
    extract_materials,
    get_unit_scale_to_mm,
    value_to_mm,
)


def _get_secret(name):
    """Read a setting from Streamlit secrets, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name) or None


class IFCGraphMapper:
    BATCH_SIZE = 500

    def __init__(self):
        uri = _get_secret("NEO4J_URI")
        username = _get_secret("NEO4J_USERNAME")
        password = _get_secret("NEO4J_PASSWORD")
        self.database = _get_secret("NEO4J_DATABASE")  # None = server default

        missing = [
            name for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USERNAME", username),
                ("NEO4J_PASSWORD", password),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing Neo4j settings: " + ", ".join(missing)
                + ". Add them to Streamlit secrets (never commit them to GitHub)."
            )

        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            self.driver.verify_connectivity()
        except Exception:
            self.driver.close()
            raise

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _session(self):
        return self.driver.session(database=self.database)

    def _run_batches(self, session, query, rows, **params):
        for start in range(0, len(rows), self.BATCH_SIZE):
            session.run(query, rows=rows[start:start + self.BATCH_SIZE], **params)

    # -- schema ---------------------------------------------------------------
    def setup_constraints(self):
        with self._session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:BIMElement) REQUIRE e.globalId IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:BuildingStorey) REQUIRE s.globalId IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Material) ON (m.name)")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:AuditProject) REQUIRE a.auditId IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:SpecificationClause) REQUIRE c.clauseKey IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:GraphTerm) REQUIRE t.term IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:WallCategory) REQUIRE k.catKey IS UNIQUE")

    # -- upload ---------------------------------------------------------------
    @staticmethod
    def _graph_terms(*values):
        """Normalize IFC names, classes and materials into shared graph terms."""
        text = unicodedata.normalize("NFKC", " ".join(str(value or "") for value in values))
        text = re.sub(r"([a-zäöüß])([A-ZÄÖÜ])", r"\1 \2", text)
        text = re.sub(r"\bIfc", " ", text, flags=re.IGNORECASE)
        # Unicode-aware: German/French/Spanish words (Dämmung, béton) stay whole.
        terms = {
            token.lower().strip(".+/-")
            for token in re.findall(r"[^\W\d_][\w.+/-]{2,}", text)
        }
        # Also index ASCII spellings (Dämmung -> daemmung) because IFC names and PDFs mix them.
        for term in list(terms):
            folded = (term.replace("ä", "ae").replace("ö", "oe")
                          .replace("ü", "ue").replace("ß", "ss"))
            if folded != term:
                terms.add(folded)
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was",
            "were", "shall", "should", "must", "will", "into", "over", "under",
            "all", "any", "not", "has", "have", "its", "their", "than", "then",
            "each", "name", "unnamed", "type", "element",
        }
        terms = {term for term in terms if len(term) >= 3 and term not in stop}
        low = text.lower()
        if "wall" in low:
            terms.add("wall")
        if "curtainwall" in low or "curtain wall" in low:
            terms.update({"curtain", "wall"})
        if "plasterboard" in low or "drywall" in low:
            terms.update({"plasterboard", "drywall"})
        return sorted(terms)

    @staticmethod
    def _quick_props(product, unit_scale_to_mm):
        """Light-weight width / external flag for elements the audit did not extract."""
        width_mm, is_external = None, False
        try:
            flat = {}
            for props in ifcopenshell.util.element.get_psets(product).values():
                for k, v in props.items():
                    flat[str(k).lower()] = v
                    if "isexternal" in str(k).lower() and (
                        v is True or (isinstance(v, str) and v.strip().lower() == "true")
                    ):
                        is_external = True
            for kw in WIDTH_KEYWORDS:
                if flat.get(kw) not in (None, "", "Unknown"):
                    try:
                        width_mm = round(value_to_mm(flat[kw], unit_scale_to_mm), 2)
                        break
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
        return width_mm, is_external

    def upload_ifc_to_graph(self, ifc_filepath, extracted=None, model_id=None,
                            audit_id=None, selected_global_ids=None):
        """Write storeys, elements, containment and material links.

        extracted: optional output of extract_all_ifc_wall_dimensions, so
        audited elements carry exactly the values used in the audit.
        Returns a dict of counts.
        """
        self.setup_constraints()
        ifc_model = ifcopenshell.open(ifc_filepath)
        unit_scale_to_mm = get_unit_scale_to_mm(ifc_model)
        extracted_by_gid = {e["GlobalId"]: e for e in (extracted or [])}
        selected_global_ids = set(selected_global_ids or extracted_by_gid.keys())

        storey_rows = [
            {"gid": s.GlobalId, "name": s.Name or "Unnamed Level"}
            for s in ifc_model.by_type("IfcBuildingStorey")
        ]
        element_rows, contained_rows, material_rows, element_term_rows = [], [], [], []

        for product in ifc_model.by_type("IfcProduct"):
            if product.is_a("IfcSpatialStructureElement"):
                continue
            gid = product.GlobalId

            known = extracted_by_gid.get(gid)
            if known:
                width_mm, is_external = known.get("WidthMm"), bool(known.get("IsExternal"))
                material_names = known.get("MaterialNames") or []
            else:
                width_mm, is_external = self._quick_props(product, unit_scale_to_mm)
                material_names = extract_materials(product, unit_scale_to_mm)["names"]

            element_rows.append({
                "gid": gid,
                "name": product.Name or "Unnamed Element",
                "type": product.is_a(),
                "width": width_mm,
                "isExternal": is_external,
            })

            for relation in getattr(product, "ContainedInStructure", None) or []:
                structure = getattr(relation, "RelatingStructure", None)
                if structure is not None:
                    contained_rows.append({"element_id": gid, "storey_id": structure.GlobalId})

            for mat_name in material_names:
                material_rows.append({"element_id": gid, "mat_name": mat_name})

            # Only the selected audit elements need term links for retrieval.
            if gid in selected_global_ids:
                terms = set(self._graph_terms(product.Name or "", product.is_a()))
                for mat_name in material_names:
                    terms.update(self._graph_terms(mat_name))
                for term in terms:
                    element_term_rows.append({"element_id": gid, "term": term})

        with self._session() as session:
            self._run_batches(
                session,
                "UNWIND $rows AS r MERGE (s:BuildingStorey {globalId: r.gid}) SET s.name = r.name",
                storey_rows,
            )
            for start in range(0, len(element_rows), self.BATCH_SIZE):
                session.run(
                    "UNWIND $rows AS r MERGE (e:BIMElement {globalId: r.gid}) "
                    "SET e.name = r.name, e.type = r.type, e.widthMm = r.width, "
                    "e.isExternal = r.isExternal, e.modelId = $model_id",
                    rows=element_rows[start:start + self.BATCH_SIZE],
                    model_id=model_id,
                )
            self._run_batches(
                session,
                "UNWIND $rows AS r "
                "MATCH (e:BIMElement {globalId: r.element_id}), (s:BuildingStorey {globalId: r.storey_id}) "
                "MERGE (e)-[:CONTAINED_IN]->(s)",
                contained_rows,
            )
            self._run_batches(
                session,
                "UNWIND $rows AS r "
                "MATCH (e:BIMElement {globalId: r.element_id}) "
                "MERGE (m:Material {name: r.mat_name}) "
                "MERGE (e)-[:HAS_MATERIAL]->(m)",
                material_rows,
            )
            if selected_global_ids:
                session.run(
                    "UNWIND $ids AS gid MATCH (e:BIMElement {globalId: gid}) "
                    "OPTIONAL MATCH (e)-[old:HAS_TERM]->(:GraphTerm) DELETE old",
                    ids=sorted(selected_global_ids),
                )
            self._run_batches(
                session,
                "UNWIND $rows AS r "
                "MATCH (e:BIMElement {globalId: r.element_id}) "
                "MERGE (t:GraphTerm {term: r.term}) "
                "MERGE (e)-[:HAS_TERM]->(t)",
                element_term_rows,
            )
            if audit_id:
                session.run(
                    "MERGE (a:AuditProject {auditId: $audit_id}) "
                    "SET a.modelId = $model_id, a.updatedAt = datetime()",
                    audit_id=audit_id,
                    model_id=model_id,
                )
                selected_rows = [{"gid": gid} for gid in selected_global_ids]
                self._run_batches(
                    session,
                    "UNWIND $rows AS r "
                    "MATCH (a:AuditProject {auditId: $audit_id}), "
                    "(e:BIMElement {globalId: r.gid}) "
                    "MERGE (a)-[:USES_ELEMENT]->(e)",
                    selected_rows,
                    audit_id=audit_id,
                )

        return {
            "storeys": len(storey_rows),
            "elements": len(element_rows),
            "containment_links": len(contained_rows),
            "material_links": len(material_rows),
            "element_term_links": len(element_term_rows),
            "selected_audit_elements": len(selected_global_ids),
        }

    # -- specification graph and retrieval -----------------------------------
    def upload_specification_clauses(self, audit_id, clauses, source_name="specification.pdf"):
        """Store page-addressable PDF clauses and link their terms into the graph."""
        with self._session() as session:
            session.run(
                "MERGE (a:AuditProject {auditId: $audit_id}) "
                "SET a.specificationSource = $source_name, a.updatedAt = datetime()",
                audit_id=audit_id,
                source_name=source_name,
            )
            clause_rows = []
            term_rows = []
            for clause in clauses:
                source_id = str(clause["source_id"])
                clause_key = f"{audit_id}|{source_id}"
                clause_rows.append({
                    "clause_key": clause_key,
                    "source_id": source_id,
                    "page": int(clause["page"]),
                    "text": clause["text"],
                    "source": source_name,
                })
                for term in self._graph_terms(clause["text"]):
                    term_rows.append({"clause_key": clause_key, "term": term})
            self._run_batches(
                session,
                "UNWIND $rows AS r "
                "MERGE (c:SpecificationClause {clauseKey: r.clause_key}) "
                "SET c.sourceId = r.source_id, c.page = r.page, c.text = r.text, c.source = r.source "
                "WITH c MATCH (a:AuditProject {auditId: $audit_id}) "
                "MERGE (a)-[:HAS_SPEC_CLAUSE]->(c)",
                clause_rows,
                audit_id=audit_id,
            )
            self._run_batches(
                session,
                "UNWIND $rows AS r "
                "MATCH (c:SpecificationClause {clauseKey: r.clause_key}) "
                "MERGE (t:GraphTerm {term: r.term}) "
                "MERGE (c)-[:MENTIONS_TERM]->(t)",
                term_rows,
            )
        return {"specification_clauses": len(clause_rows), "clause_term_links": len(term_rows)}

    def retrieve_specification_clauses(self, audit_id, limit=100):
        """Graph traversal: selected IFC walls -> shared terms <- spec clauses."""
        query = (
            "MATCH (a:AuditProject {auditId: $audit_id})-[:USES_ELEMENT]->(e:BIMElement) "
            "MATCH (e)-[:HAS_TERM]->(t:GraphTerm)<-[:MENTIONS_TERM]-(c:SpecificationClause) "
            "MATCH (a)-[:HAS_SPEC_CLAUSE]->(c) "
            "WITH c, count(DISTINCT t.term) AS matched_terms, "
            "collect(DISTINCT e.globalId) AS related_global_ids "
            "ORDER BY matched_terms DESC, c.page ASC "
            "RETURN c.sourceId AS source_id, c.page AS page, c.text AS text, "
            "c.source AS source, matched_terms, related_global_ids "
            "ORDER BY matched_terms DESC, page ASC LIMIT $limit"
        )
        with self._session() as session:
            result = session.run(query, audit_id=audit_id, limit=int(limit))
            return [record.data() for record in result]


    # -- category layer: the BIM <-> PDF link --------------------------------
    def upload_categories(self, audit_id, categories, cat_terms, cat_element_ids):
        """Store each IFC category as a node, linked to its elements and terms.

        categories:      [{'id','key','ifc_class','is_external'}, ...]
        cat_terms:       {category_id: [term, ...]}
        cat_element_ids: {category_id: [globalId, ...]}
        """
        cat_rows, term_rows, member_rows = [], [], []
        for c in categories:
            cat_key = f"{audit_id}|{c['id']}"
            cat_rows.append({
                "cat_key": cat_key, "cat_id": int(c["id"]), "name": c["key"],
                "ifc_class": c["ifc_class"], "is_external": bool(c["is_external"]),
            })
            term_rows += [{"cat_key": cat_key, "term": t} for t in cat_terms.get(c["id"], [])]
            member_rows += [{"cat_key": cat_key, "gid": g} for g in cat_element_ids.get(c["id"], [])]
        with self._session() as session:
            # Re-running the same audit rebuilds its categories (and their links).
            session.run(
                "MATCH (a:AuditProject {auditId: $audit_id})-[:HAS_CATEGORY]->(k:WallCategory) "
                "DETACH DELETE k",
                audit_id=audit_id,
            )
            self._run_batches(
                session,
                "UNWIND $rows AS r MERGE (k:WallCategory {catKey: r.cat_key}) "
                "SET k.catId = r.cat_id, k.name = r.name, k.ifcClass = r.ifc_class, "
                "k.isExternal = r.is_external "
                "WITH k MATCH (a:AuditProject {auditId: $audit_id}) MERGE (a)-[:HAS_CATEGORY]->(k)",
                cat_rows,
                audit_id=audit_id,
            )
            self._run_batches(
                session,
                "UNWIND $rows AS r MATCH (k:WallCategory {catKey: r.cat_key}) "
                "MERGE (t:GraphTerm {term: r.term}) MERGE (k)-[:HAS_TERM]->(t)",
                term_rows,
            )
            self._run_batches(
                session,
                "UNWIND $rows AS r MATCH (k:WallCategory {catKey: r.cat_key}), "
                "(e:BIMElement {globalId: r.gid}) MERGE (e)-[:IN_CATEGORY]->(k)",
                member_rows,
            )
        return {"categories": len(cat_rows), "category_term_links": len(term_rows),
                "category_element_links": len(member_rows)}

    def retrieve_candidates_by_category(self, audit_id, per_category=12, neighbours=4):
        """Graph traversal per category: Category -> shared terms <- Clause.

        Returns {category_id: [clause dicts]}. The LLM may only cite clauses
        returned here for that category.
        """
        query = (
            "MATCH (a:AuditProject {auditId: $audit_id})-[:HAS_CATEGORY]->(k:WallCategory)"
            "-[:HAS_TERM]->(t:GraphTerm)<-[:MENTIONS_TERM]-(c:SpecificationClause)"
            "<-[:HAS_SPEC_CLAUSE]-(a) "
            "WITH k, c, count(DISTINCT t) AS score, collect(DISTINCT t.term) AS shared "
            "ORDER BY k.catId ASC, score DESC, c.page ASC, c.sourceId ASC "
            "WITH k, collect({source_id: c.sourceId, page: c.page, text: c.text, "
            "score: score, shared_terms: shared}) AS hits "
            "RETURN k.catId AS category_id, hits[0..$limit] AS clauses ORDER BY category_id"
        )
        with self._session() as session:
            result = {r["category_id"]: r["clauses"]
                      for r in session.run(query, audit_id=audit_id, limit=int(per_category))}

            # Adjacent clauses on the same page give section context (as before).
            wanted = set()
            for hits in result.values():
                have = {h["source_id"] for h in hits}
                added = 0
                for h in hits:
                    m = re.fullmatch(r"P(\d+)-C(\d+)", h["source_id"])
                    if not m:
                        continue
                    page, num = int(m.group(1)), int(m.group(2))
                    for n in (num - 1, num + 1):
                        nid = f"P{page}-C{n}"
                        if n > 0 and nid not in have and added < neighbours:
                            wanted.add(nid)
                            have.add(nid)
                            added += 1
            neighbour_rows = {}
            if wanted:
                rows = session.run(
                    "MATCH (a:AuditProject {auditId: $audit_id})-[:HAS_SPEC_CLAUSE]->(c) "
                    "WHERE c.sourceId IN $ids "
                    "RETURN c.sourceId AS source_id, c.page AS page, c.text AS text",
                    audit_id=audit_id, ids=sorted(wanted),
                )
                neighbour_rows = {r["source_id"]: r.data() for r in rows}

        for cid, hits in result.items():
            have = {h["source_id"] for h in hits}
            for h in list(hits):
                m = re.fullmatch(r"P(\d+)-C(\d+)", h["source_id"])
                if not m:
                    continue
                page, num = int(m.group(1)), int(m.group(2))
                for n in (num - 1, num + 1):
                    nid = f"P{page}-C{n}"
                    if nid in neighbour_rows and nid not in have:
                        row = neighbour_rows[nid]
                        hits.append({**row, "score": 0, "shared_terms": []})
                        have.add(nid)
        return result

    def write_verified_links(self, audit_id, requirements, prompt_revision=""):
        """Persist evidence-grounded (Category)-[:GOVERNED_BY]->(Clause) links.

        Grounded = the cited clause exists and contains the reported values.
        It does not prove the clause applies; review high-severity findings.
        """
        rows = []
        for r in requirements:
            if not (r.get("spec_found") and r.get("evidence_verified") and r.get("source_id")):
                continue
            rows.append({
                "cat_key": f"{audit_id}|{r['category_id']}",
                "clause_key": f"{audit_id}|{r['source_id']}",
                "thickness": r.get("required_thickness_mm"),
                "materials": list(r.get("required_materials") or []),
                "quote": r.get("evidence_quote") or "",
                "ref": r.get("clause_reference") or "",
            })
        with self._session() as session:
            self._run_batches(
                session,
                "UNWIND $rows AS r "
                "MATCH (k:WallCategory {catKey: r.cat_key}), "
                "(c:SpecificationClause {clauseKey: r.clause_key}) "
                "MERGE (k)-[l:GOVERNED_BY]->(c) "
                "SET l.requiredThicknessMm = r.thickness, l.requiredMaterials = r.materials, "
                "l.evidenceQuote = r.quote, l.clauseReference = r.ref, l.evidenceGrounded = true, "
                "l.promptRevision = $rev, l.linkedAt = datetime()",
                rows,
                rev=prompt_revision,
            )
        return len(rows)

    # -- audit results --------------------------------------------------------
    def write_audit_results(self, rows, model_id=None):
        """Store severity / issue / required thickness on each audited element.

        rows: dicts with gid, severity, issueType, issue,
        requiredThicknessMm, widthMm.
        """
        with self._session() as session:
            self._run_batches(
                session,
                "UNWIND $rows AS r MATCH (e:BIMElement {globalId: r.gid}) "
                "WHERE $model_id IS NULL OR e.modelId = $model_id "
                "SET e.severity = r.severity, e.issueType = r.issueType, "
                "e.issue = r.issue, e.requiredThicknessMm = r.requiredThicknessMm, "
                "e.auditedWidthMm = r.widthMm",
                rows,
                model_id=model_id,
            )
        return len(rows)

    def severity_summary(self, model_id=None):
        """Read the severity counts back FROM the graph (round-trip check)."""
        with self._session() as session:
            result = session.run(
                "MATCH (e:BIMElement) WHERE e.severity IS NOT NULL "
                "AND ($model_id IS NULL OR e.modelId = $model_id) "
                "RETURN e.severity AS severity, count(*) AS elements "
                "ORDER BY elements DESC",
                model_id=model_id,
            )
            return [record.data() for record in result]
