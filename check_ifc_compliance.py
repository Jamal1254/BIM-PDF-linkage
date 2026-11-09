"""check_ifc_compliance.py

Module M1 of the BIM compliance auditor: IFC extraction.

Scans an IFC model for every element matching a keyword and extracts, for
each one: thickness (with the source it came from), materials, cavity width,
external flag and storey. Thickness uses a tiered fallback; every result
records WHICH tier produced it so the audit is traceable.

Width tiers (first plausible value wins):
    0. Sum of material-layer thicknesses (the actual build-up)
    1. Property-set sweep (WallThickness, OverallThickness, ..., Width)
    2. A "123mm" pattern in the element name / type
    3. 3D bounding box (shortest horizontal side)
"""

import re

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element

try:
    import ifcopenshell.util.unit as _ifc_unit
except Exception:  # pragma: no cover - very old IfcOpenShell
    _ifc_unit = None

# Generous upper limit; even thick masonry cavity walls are well below this.
MAX_PLAUSIBLE_THICKNESS_MM = 600.0

WIDTH_KEYWORDS = [
    "wallthickness",
    "overallthickness",
    "nominalthickness",
    "thickness/width",
    "thickness",
    "width",
]
CAVITY_KEYWORDS = ["cavity", "cavitywidth", "airgap", "voidwidth"]

# IFC classes routinely NAMED after what they sit in or next to (e.g. "Wall
# Lamp", "Door-Curtain-Wall-Double"). A name-only match for these is a false
# positive. A match on the IFC class itself (IfcCurtainWall) is still trusted.
NON_STRUCTURAL_EXCLUDE_TYPES = {
    "IfcDoor", "IfcWindow",
    "IfcFurnishingElement", "IfcFurniture",
    "IfcFlowTerminal", "IfcSanitaryTerminal",
    "IfcRailing", "IfcLightFixture",
    "IfcProxy", "IfcBuildingElementProxy",
}


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def _legacy_unit_scale_to_mm(ifc_file):
    """Name-based unit detection, used only if IfcOpenShell can't compute it."""
    scale = 1000.0
    try:
        for unit in ifc_file.by_type("IfcUnitAssignment")[0].Units:
            if getattr(unit, "UnitType", "") == "LENGTHUNIT":
                prefix = str(getattr(unit, "Prefix", "")).upper()
                name = str(getattr(unit, "Name", "")).upper()
                if "FOOT" in name or "FEET" in name:
                    scale = 304.8
                elif "INCH" in name:
                    scale = 25.4
                elif "MILLI" in prefix or "MILLIMETRE" in name:
                    scale = 1.0
                elif "CENTI" in prefix or "CENTIMETRE" in name:
                    scale = 10.0
                elif "METRE" in name:
                    scale = 1000.0
    except Exception:
        scale = 1000.0
    return scale


def get_unit_scale_to_mm(ifc_file):
    """Millimetres per project length unit (1.0 for mm models, 1000.0 for m)."""
    if _ifc_unit is not None:
        try:
            metres_per_unit = float(_ifc_unit.calculate_unit_scale(ifc_file))
            if metres_per_unit > 0:
                return metres_per_unit * 1000.0
        except Exception:
            pass
    return _legacy_unit_scale_to_mm(ifc_file)


def value_to_mm(value, unit_scale_to_mm):
    """Convert a property value from project units to mm.

    Property values come back in project units. If the converted number is
    implausibly large but the raw number is plausible, the exporter probably
    wrote the value in mm already, so the raw number is used instead.
    """
    raw = float(value)
    converted = raw * unit_scale_to_mm
    if (
        unit_scale_to_mm != 1.0
        and converted > MAX_PLAUSIBLE_THICKNESS_MM
        and raw <= MAX_PLAUSIBLE_THICKNESS_MM
    ):
        return raw
    return converted


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def _clean_name(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_materials(element, unit_scale_to_mm):
    """Return the element's materials in a uniform structure.

    Handles IfcMaterial, IfcMaterialList, layer sets (and usages),
    constituent sets and profile sets.

    Keys: display (strings for reports), names (real material names only),
    layer_sum (mm), has_layer_set, cavity_mm (first cavity/air layer).
    """
    out = {"display": [], "names": [], "layer_sum": 0.0,
           "has_layer_set": False, "cavity_mm": None}

    def add_name(name):
        if name and name not in out["names"]:
            out["names"].append(name)

    try:
        mats = ifcopenshell.util.element.get_material(element)
        if not mats:
            return out

        layer_set = None
        if mats.is_a("IfcMaterialLayerSet"):
            layer_set = mats
        elif mats.is_a("IfcMaterialLayerSetUsage"):
            layer_set = mats.ForLayerSet

        if layer_set is not None:
            out["has_layer_set"] = True
            for layer in layer_set.MaterialLayers or []:
                mat_name = _clean_name(layer.Material.Name) if layer.Material else None
                thick = (
                    round(float(layer.LayerThickness) * unit_scale_to_mm, 2)
                    if layer.LayerThickness else 0.0
                )
                out["display"].append(f"{mat_name or 'Unnamed Layer'} ({thick}mm)")
                out["layer_sum"] += thick
                add_name(mat_name)
                if (
                    out["cavity_mm"] is None
                    and mat_name
                    and ("cavity" in mat_name.lower() or mat_name.lower() == "air")
                ):
                    out["cavity_mm"] = thick
        elif mats.is_a("IfcMaterialConstituentSet"):
            for constituent in getattr(mats, "MaterialConstituents", None) or []:
                name = _clean_name(constituent.Material.Name) if constituent.Material else None
                out["display"].append(name or "Unnamed Constituent")
                add_name(name)
        elif mats.is_a("IfcMaterialProfileSet") or mats.is_a("IfcMaterialProfileSetUsage"):
            profile_set = mats if mats.is_a("IfcMaterialProfileSet") else mats.ForProfileSet
            for profile in getattr(profile_set, "MaterialProfiles", None) or []:
                name = _clean_name(profile.Material.Name) if profile.Material else None
                out["display"].append(name or "Unnamed Profile")
                add_name(name)
        elif mats.is_a("IfcMaterialList"):
            for material in mats.Materials or []:
                name = _clean_name(getattr(material, "Name", None))
                out["display"].append(name or "Unnamed Material")
                add_name(name)
        else:
            name = _clean_name(getattr(mats, "Name", None))
            out["display"].append(name or "Unnamed Material")
            add_name(name)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Width helpers
# ---------------------------------------------------------------------------
def _consider(state, source, value_mm):
    """Record a width candidate: first plausible wins; remember first implausible."""
    if state["chosen"] is not None or value_mm is None or value_mm <= 0:
        return
    if value_mm <= MAX_PLAUSIBLE_THICKNESS_MM:
        state["chosen"] = (source, round(value_mm, 2))
    elif state["implausible"] is None:
        state["implausible"] = (source, round(value_mm, 2))


def _is_true(value):
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() == "true"


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_all_ifc_wall_dimensions(file_path, search_keyword="Wall"):
    """Extract IFC wall components for the canonical ALL_WALLS selection.

    The app normalizes natural-language wall descriptions to ALL_WALLS before
    calling this function. The same IFC class/name rules are therefore used
    regardless of whether the user typed "wall", "wall systems" or "drywall".
    Non-canonical keywords retain the original substring-search behaviour.

    Raises an exception if the IFC file cannot be opened. Returns [] if no
    matching elements are found.
    """
    ifc_file = ifcopenshell.open(file_path)
    unit_scale_to_mm = get_unit_scale_to_mm(ifc_file)
    raw_selection = re.sub(r"\s+", " ", str(search_keyword or "")).strip().casefold()
    all_walls = raw_selection in {
        "all_walls", "all walls", "wall", "walls", "walling",
        "wall component", "wall components", "all wall components",
        "wall structure", "wall structures", "wall system", "wall systems",
        "wall construction", "wall constructions", "partition", "partitions",
        "partitioning", "drywall", "dry wall", "dry lining", "plasterboard",
        "masonry", "brickwork", "blockwork", "trockenbau", "mauerwerk",
        "mauer", "ziegel", "trennwand", "innenwand", "aussenwand",
        "außenwand", "fassade", "fassaden", "facade", "façade",
        "liftwand", "schachtwand", "ifcwall", "ifcwallstandardcase",
        "ifccurtainwall",
    }
    keyword = "wall" if all_walls else raw_selection
    if not keyword:
        return []

    wall_name_pattern = re.compile(
        r"\b(?:wall|walls|walling|partition|partitions|partitioning|drywall|"
        r"dry\s+wall|dry\s+lining|plasterboard|masonry|brickwork|blockwork|"
        r"trockenbau|mauerwerk|mauer|ziegel|trennwand|innenwand|aussenwand|"
        r"außenwand|fassade|fassaden|facade|façade|liftwand|schachtwand)\b",
        re.IGNORECASE,
    )

    # --- gather candidates -------------------------------------------------
    candidates = []
    for element in ifc_file.by_type("IfcProduct"):
        if element.is_a("IfcOpeningElement") or element.is_a("IfcOpening"):
            continue
        el_type = element.is_a()
        el_name = str(getattr(element, "Name", "") or "")

        if all_walls:
            # Include IFC wall entities and curtain walls by class, plus other
            # non-excluded IFC products explicitly named as wall construction.
            try:
                is_wall_class = element.is_a("IfcWall") or element.is_a("IfcCurtainWall")
            except Exception:
                is_wall_class = el_type in {"IfcWall", "IfcWallStandardCase", "IfcCurtainWall"}
            if is_wall_class:
                candidates.append(element)
            elif el_type not in NON_STRUCTURAL_EXCLUDE_TYPES and wall_name_pattern.search(el_name):
                candidates.append(element)
        else:
            # Preserve the original keyword-search behaviour for existing callers.
            if keyword in el_type.casefold():
                candidates.append(element)
            elif keyword in el_name.casefold() and el_type not in NON_STRUCTURAL_EXCLUDE_TYPES:
                candidates.append(element)

    if not candidates:
        return []

    geom_settings = None  # created lazily, only if a geometry fallback is needed
    results = []

    for target in candidates:
        data = {
            "Type": target.is_a(),
            "Name": getattr(target, "Name", None) or "Unnamed",
            "GlobalId": target.GlobalId,
            "IsExternal": False,
            "Width": "Unknown",           # display value (number or text)
            "WidthMm": None,              # numeric value used for matching
            "WidthSource": "None",
            "WidthStatus": "unknown",     # ok | unknown | unreliable
            "WidthRaw": None,
            "CavityWidth": "None/Undefined",
            "Materials": [],              # display strings
            "MaterialNames": [],          # real material names only
            "Storey": "Unknown",
        }

        # --- storey (IfcRelContainedInSpatialStructure) ---------------------
        if hasattr(target, "ContainedInStructure"):
            for rel in target.ContainedInStructure:
                if rel.is_a("IfcRelContainedInSpatialStructure"):
                    data["Storey"] = getattr(rel.RelatingStructure, "Name", None) or "Unknown"
                    break

        # Fallback: aggregate parts inherit the storey of their parent.
        if data["Storey"] == "Unknown" and hasattr(target, "Decomposes"):
            for rel in target.Decomposes:
                if rel.is_a("IfcRelAggregates"):
                    parent = getattr(rel, "RelatingObject", None)
                    if parent is not None and hasattr(parent, "ContainedInStructure"):
                        for prel in parent.ContainedInStructure:
                            if prel.is_a("IfcRelContainedInSpatialStructure"):
                                data["Storey"] = getattr(prel.RelatingStructure, "Name", None) or "Unknown"
                                break
                if data["Storey"] != "Unknown":
                    break

        # --- materials -----------------------------------------------------
        mat_info = extract_materials(target, unit_scale_to_mm)
        data["Materials"] = mat_info["display"]
        data["MaterialNames"] = mat_info["names"]

        # --- property sets (instance + type) ---------------------------------
        try:
            psets = ifcopenshell.util.element.get_psets(target)
        except Exception:
            psets = {}
        try:
            type_element = ifcopenshell.util.element.get_type(target)
            if type_element is not None:
                for t_name, t_props in ifcopenshell.util.element.get_psets(type_element).items():
                    if t_name not in psets:
                        psets[t_name] = t_props
                    else:
                        for k, v in t_props.items():
                            if k not in psets[t_name]:
                                psets[t_name][k] = v
        except Exception:
            pass

        flat_props = {}
        for _pset_name, props in psets.items():
            for k, v in props.items():
                flat_props[str(k).lower()] = v
                if "isexternal" in str(k).lower() and _is_true(v):
                    data["IsExternal"] = True

        # --- width -----------------------------------------------------------
        state = {"chosen": None, "implausible": None}

        # Tier 0: sum of material layers (the real assembly build-up)
        if mat_info["has_layer_set"] and mat_info["layer_sum"] > 0:
            _consider(state, "Material layer sum", mat_info["layer_sum"])

        # Tier 1: property sweep, specific names before the bare "width"
        if state["chosen"] is None:
            for kw in WIDTH_KEYWORDS:
                raw = flat_props.get(kw)
                if raw in (None, "", "Unknown"):
                    continue
                try:
                    _consider(state, f"Property '{kw}'", value_to_mm(raw, unit_scale_to_mm))
                except (ValueError, TypeError):
                    continue
                if state["chosen"] is not None:
                    break

        # Tier 2: "123mm" in the element name or type
        if state["chosen"] is None:
            match = re.search(r"(\d+(?:\.\d+)?)\s*mm", f"{data['Name']} {data['Type']}", re.IGNORECASE)
            if match:
                _consider(state, "Name/type text", float(match.group(1)))

        # Tier 3: bounding box. Assumes a roughly rectangular, axis-aligned
        # element; unreliable for curved or oddly rotated walls.
        if state["chosen"] is None:
            try:
                if geom_settings is None:
                    geom_settings = ifcopenshell.geom.settings()
                    geom_settings.set(geom_settings.USE_WORLD_COORDINATES, True)
                shape = ifcopenshell.geom.create_shape(geom_settings, target)
                verts = shape.geometry.verts
                xs = verts[0::3]
                ys = verts[1::3]
                if xs and ys:
                    # IfcOpenShell returns geometry in METRES whatever the
                    # file's own unit is, so convert with 1000 (not the
                    # project unit scale).
                    thickness_mm = min(max(xs) - min(xs), max(ys) - min(ys)) * 1000.0
                    _consider(state, "3D bounding box", thickness_mm)
            except Exception:
                pass

        if state["chosen"] is not None:
            source, value = state["chosen"]
            data["Width"] = value
            data["WidthMm"] = value
            data["WidthSource"] = source
            data["WidthStatus"] = "ok"
        elif state["implausible"] is not None:
            source, value = state["implausible"]
            data["Width"] = (
                f"Unreliable ({value}mm - exceeds plausible wall thickness; "
                "verify source property/geometry)"
            )
            data["WidthSource"] = source
            data["WidthStatus"] = "unreliable"
            data["WidthRaw"] = value

        # --- cavity ------------------------------------------------------------
        for ck in CAVITY_KEYWORDS:
            if ck in flat_props and flat_props[ck] not in (None, ""):
                data["CavityWidth"] = flat_props[ck]
                break
        if data["CavityWidth"] == "None/Undefined" and mat_info["cavity_mm"] is not None:
            data["CavityWidth"] = f"{mat_info['cavity_mm']} (from material layer)"

        results.append(data)

    return results
