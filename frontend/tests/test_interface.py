from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from agent.config import Settings


class InterfaceTests(unittest.TestCase):
    def test_live_mode_shows_credit_failure_and_unavailable_score(self):
        class PaymentRequired(Exception):
            status_code = 402

        app = Path(__file__).resolve().parents[1]/"app.py"
        with patch("agent.config.load_settings", return_value=Settings(backend="ollama", api_key="TEST_KEY")), \
             patch("agent.runner.create_review_model", side_effect=PaymentRequired("PRIVATE_PROVIDER_TEXT")):
            at = AppTest.from_file(str(app), default_timeout=30).run()
            self.assertTrue(any("Live Ollama model" in info.value for info in at.info))
            next(b for b in at.button if b.label == "Review posting").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("credits or a subscription" in error.value for error in at.error))
            self.assertFalse(any("PRIVATE_PROVIDER_TEXT" in error.value for error in at.error))
            self.assertEqual(at.metric[0].value, "Unavailable")
            self.assertEqual(at.session_state["case"].error_code, "credits_required")

    def test_user_can_analyse_change_scenario_and_confirm_a_review(self):
        app = Path(__file__).resolve().parents[1]/"app.py"
        with tempfile.TemporaryDirectory() as folder, patch("agent.config.load_settings") as settings:
            audit_path = Path(folder)/"reviews.jsonl"
            settings.return_value = Settings(backend="dummy", audit_path=audit_path)
            at = AppTest.from_file(str(app), default_timeout=30).run()
            self.assertFalse(at.exception)
            self.assertTrue(any("OFFLINE DEMO" in warning.value for warning in at.warning))
            next(b for b in at.button if b.label == "Review posting").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[0].value, "0.92")
            self.assertEqual(at.session_state["case"].assessment.recommendation, "escalate_to_review")
            # The UI holds a case snapshot; another scenario clears it.
            at.selectbox[0].set_value("ordinary").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.metric), 0)
            next(b for b in at.button if b.label == "Review posting").click().run()
            self.assertEqual(at.metric[0].value, "0.25")
            self.assertEqual(at.session_state["case"].assessment.recommendation, "escalate_to_review")
            self.assertFalse(audit_path.exists())
            at.text_input(key="reviewer_name").set_value("Demo reviewer").run()
            next(b for b in at.button if b.label == "Confirm and save local decision").click().run()
            self.assertFalse(at.exception)
            record = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(record["reviewer_decision"]["reviewer"], "Demo reviewer")
            self.assertEqual(record["reviewer_decision"]["decision"], "escalated_to_review")


if __name__ == "__main__":
    unittest.main()
