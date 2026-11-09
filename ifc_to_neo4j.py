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

    def _run_batches(self, session, query, rows):
        for start in range(0, len(rows), self.BATCH_SIZE):
            session.run(query, rows=rows[start:start + self.BATCH_SIZE])

    # -- schema ---------------------------------------------------------------
    def setup_constraints(self):
        with self._session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:BIMElement) REQUIRE e.globalId IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:BuildingStorey) REQUIRE s.globalId IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Material) ON (m.name)")

    def clear_database(self):
        """Delete EVERYTHING in the target database (fresh audit graph)."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # -- upload ---------------------------------------------------------------
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

    def upload_ifc_to_graph(self, ifc_filepath, extracted=None, model_id=None):
        """Write storeys, elements, containment and material links.

        extracted: optional output of extract_all_ifc_wall_dimensions, so
        audited elements carry exactly the values used in the audit.
        Returns a dict of counts.
        """
        self.setup_constraints()
        ifc_model = ifcopenshell.open(ifc_filepath)
        unit_scale_to_mm = get_unit_scale_to_mm(ifc_model)
        extracted_by_gid = {e["GlobalId"]: e for e in (extracted or [])}

        storey_rows = [
            {"gid": s.GlobalId, "name": s.Name or "Unnamed Level"}
            for s in ifc_model.by_type("IfcBuildingStorey")
        ]
        element_rows, contained_rows, material_rows = [], [], []

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

        return {
            "storeys": len(storey_rows),
            "elements": len(element_rows),
            "containment_links": len(contained_rows),
            "material_links": len(material_rows),
        }

    # -- audit results --------------------------------------------------------
    def write_audit_results(self, rows):
        """Store audit results and return the number of graph nodes actually updated.

        rows: dicts with gid, severity, issueType, issue,
        requiredThicknessMm, widthMm.
        """
        updated_total = 0
        query = (
            "UNWIND $rows AS r MATCH (e:BIMElement {globalId: r.gid}) "
            "SET e.severity = r.severity, e.issueType = r.issueType, "
            "e.issue = r.issue, e.requiredThicknessMm = r.requiredThicknessMm, "
            "e.auditedWidthMm = r.widthMm "
            "RETURN count(e) AS updated"
        )
        with self._session() as session:
            for start in range(0, len(rows), self.BATCH_SIZE):
                result = session.run(query, rows=rows[start:start + self.BATCH_SIZE])
                record = result.single()
                updated_total += int(record["updated"]) if record else 0
        return updated_total

    def severity_summary(self, model_id=None):
        """Read severity counts back FROM the graph, optionally for one IFC model."""
        with self._session() as session:
            if model_id is None:
                result = session.run(
                    "MATCH (e:BIMElement) WHERE e.severity IS NOT NULL "
                    "RETURN e.severity AS severity, count(*) AS elements "
                    "ORDER BY elements DESC"
                )
            else:
                result = session.run(
                    "MATCH (e:BIMElement) "
                    "WHERE e.severity IS NOT NULL AND e.modelId = $model_id "
                    "RETURN e.severity AS severity, count(*) AS elements "
                    "ORDER BY elements DESC",
                    model_id=model_id,
                )
            return [record.data() for record in result]
