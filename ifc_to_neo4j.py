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

    # -- upload ---------------------------------------------------------------
    @staticmethod
    def _graph_terms(*values):
        """Normalize IFC names, classes and materials into shared graph terms."""
        text = " ".join(str(value or "") for value in values)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"\bIfc", " ", text, flags=re.IGNORECASE)
        terms = {
            token.lower().strip(".+/-")
            for token in re.findall(r"[A-Za-z][A-Za-z0-9.+/-]{2,}", text)
        }
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
