"""Quick offline checks; no API key, IFC model or Neo4j connection required.

Run: python test_smoke.py
"""
import ast
import re
import unittest
from pathlib import Path

import spec_matching as sm

# The Streamlit module is not imported because it starts the UI on import.
app_ast = ast.parse((Path(__file__).parent / 'app.py').read_text(encoding='utf-8'))
func = next(n for n in app_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'category_key')
namespace = {'re': re}
exec(compile(ast.Module(body=[func], type_ignores=[]), '<category_key>', 'exec'), namespace)
category_key = namespace['category_key']


class AuditorSmokeTests(unittest.TestCase):
    def test_revit_instance_ids_are_grouped(self):
        a = {'Name': 'Basic Wall:Beschichtung:1196604', 'Type': 'IfcWall'}
        b = {'Name': 'Basic Wall:Beschichtung:1196605', 'Type': 'IfcWall'}
        c = {'Name': 'Basic Wall:Beschichtung:1196606', 'Type': 'IfcWallStandardCase'}
        self.assertEqual(category_key(a), category_key(b))
        self.assertNotEqual(category_key(a), category_key(c))

    def test_unverified_requirement_is_rejected(self):
        data = [{'category_id': 1, 'spec_found': True, 'required_thickness_mm': 200,
                 'required_materials': ['brick'], 'evidence_quote': 'not in specification'}]
        req = sm.normalise_requirements(data, {1}, 'The wall shall have 250 mm concrete block.')
        self.assertFalse(req[1]['spec_found'])
        self.assertIsNone(req[1]['required_thickness_mm'])
        self.assertIn('rejected', req[1]['note'])

    def test_verified_requirement_is_retained(self):
        quote = 'The wall shall have 250 mm concrete block.'
        data = [{'category_id': 1, 'spec_found': True, 'required_thickness_mm': 250,
                 'required_materials': ['concrete block'], 'evidence_quote': quote}]
        req = sm.normalise_requirements(data, {1}, quote)
        self.assertTrue(req[1]['spec_found'])
        self.assertTrue(req[1]['evidence_verified'])

    def test_matching_is_deterministic(self):
        el = {'WidthMm': 260, 'MaterialNames': ['concrete block'], 'WidthStatus': 'ok'}
        req = {'spec_found': True, 'required_thickness_mm': 250,
               'required_materials': ['concrete block']}
        first = sm.evaluate_element(el, req, 5)
        self.assertEqual(first, sm.evaluate_element(el, req, 5))
        self.assertEqual((first['Severity'], first['IssueType']), ('Medium', 'Geometry mismatch'))

    def test_missing_both(self):
        el = {'WidthMm': None, 'MaterialNames': [], 'WidthStatus': 'unknown'}
        req = {'spec_found': False}
        out = sm.evaluate_element(el, req, 5)
        self.assertEqual(out['Severity'], 'Critical')
        self.assertEqual(out['IssueType'], 'Geometry & material missing')



# Additional checks can also be run via: python -m unittest test_smoke -v
class ApiHotfixTests(unittest.TestCase):
    def test_supported_model_ids(self):
        self.assertEqual(sm.LLM_MODELS[0], 'gemini-3.8-flash')
        self.assertNotIn('gemini-2.0-flash', sm.LLM_MODELS)

    def test_client_uses_beta_api(self):
        from types import SimpleNamespace
        source = (Path(__file__).parent / 'app.py').read_text(encoding='utf-8')
        module = ast.parse(source)
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'make_genai_client')
        args_seen = []
        fake_genai = SimpleNamespace(Client=lambda **kwargs: args_seen.append(kwargs) or kwargs)
        scope = {'genai': fake_genai}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<make_genai_client>', 'exec'), scope)
        scope['make_genai_client']('fake-key')
        self.assertEqual(args_seen[0]['http_options']['api_version'], 'v1beta')

    def test_fallback_when_model_returns_404(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        class FakeModels:
            def generate_content(self, *, model, contents, config):
                if model == 'gemini-3.8-flash':
                    raise RuntimeError('404 NOT_FOUND unavailable in test')
                return SimpleNamespace(text='{"requirements": []}')
        fake_client = SimpleNamespace(models=FakeModels())
        with patch.object(sm, '_build_config', return_value={}):
            answer, selected_model = sm.call_llm(fake_client, 'test')
        self.assertEqual(selected_model, 'gemini-3.5-flash-lite')
        self.assertIn('requirements', answer)



class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from types import SimpleNamespace
        module = ast.parse((Path(__file__).parent / 'app.py').read_text(encoding='utf-8'))
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'diagnose_mapping')
        cls.scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<diagnose_mapping>', 'exec'), cls.scope)

    def test_api_failure_is_reported(self):
        d = self.scope['diagnose_mapping']('text', False, True, '429 quota exceeded', [])
        self.assertTrue(any('quota' in r for r in d['reasons']))

    def test_truncation_is_reported(self):
        d = self.scope['diagnose_mapping']('text', True, True, None, [])
        self.assertTrue(any('omitted' in r for r in d['reasons']))

    def test_no_applicable_clauses_is_reported(self):
        d = self.scope['diagnose_mapping']('text', False, True, None, [sm.empty_requirement(1)])
        self.assertTrue(any('no applicable' in r.lower() for r in d['reasons']))

    def test_rejected_quotes_are_reported(self):
        r = sm.empty_requirement(1, 'Unsupported requirement rejected: evidence quote not found')
        d = self.scope['diagnose_mapping']('text', False, True, None, [r])
        self.assertEqual(d['stats']['unsupported_rejected'], 1)

    def test_verified_requirement_is_checkable(self):
        r = {'spec_found': True, 'evidence_verified': True, 'required_thickness_mm': 200, 'required_materials': []}
        d = self.scope['diagnose_mapping']('text', False, True, None, [r])
        self.assertEqual(d['stats']['verified_checkable'], 1)
        self.assertFalse(d['reasons'])

if __name__ == '__main__':
    unittest.main(verbosity=2)


class CandidateRetrievalTests(unittest.TestCase):
    def test_drywall_category_retrieves_plasterboard_and_stud_wall_clauses(self):
        context = """[Page 14]
[P14-C2] 352mm cavity wall construction comprising 102mm brickwork and 100mm blockwork.
[Page 19]
[P19-C3] Integral garage walls use 100mm aircrete blockwork, 60mm insulation between timber studwork, faced with 12.5mm gypsum based plasterboard.
[Page 26]
[P26-C12] To vertical stud walls separating a roof void from habitable areas fit 100mm glass mineral wool between studs and face with 12.5mm vapour check fire resisting plasterboard.
"""
        category = {
            "id": 1,
            "key": "Basic Wall:Trockenbau - GK 100 (IfcWall; internal/unspecified)",
            "ifc_class": "IfcWall",
            "is_external": False,
        }
        candidates = sm._candidate_clauses_for_category(category, context)
        ids = {source_id for source_id, _ in candidates}
        self.assertIn("P19-C3", ids)
        self.assertIn("P26-C12", ids)
        self.assertNotIn("P14-C2", ids)

    def test_prompt_contains_source_ids_and_category_specific_candidates(self):
        context = "[Page 19]\\n[P19-C3] 12.5mm gypsum based plasterboard fixed to timber studwork."
        category = {
            "id": 1,
            "key": "Basic Wall:Trockenbau - GK 100",
            "ifc_class": "IfcWall",
            "is_external": False,
        }
        prompt = sm.build_prompt(context, [category])
        self.assertIn("CATEGORY ID 1", prompt)
        self.assertIn("[P19-C3]", prompt)
        self.assertIn("CATEGORY-SPECIFIC CANDIDATE CLAUSES", prompt)
