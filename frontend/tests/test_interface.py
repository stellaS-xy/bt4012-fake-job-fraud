from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from agent.tests.runtime_fixtures import offline_runtime
from agent.core.config import Settings


class InterfaceTests(unittest.TestCase):
    def test_live_mode_shows_credit_failure_and_unavailable_score(self):
        class PaymentRequired(Exception):
            status_code = 402

        app = Path(__file__).resolve().parents[1]/"app.py"
        with patch("agent.core.config.load_settings", return_value=Settings(backend="ollama", api_key="TEST_KEY")), \
             patch("agent.core.runner.create_review_model", side_effect=PaymentRequired("PRIVATE_PROVIDER_TEXT")):
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
        with offline_runtime(), tempfile.TemporaryDirectory() as folder, patch("agent.core.config.load_settings") as settings:
            audit_path = Path(folder)/"reviews.jsonl"
            settings.return_value = Settings(backend="ollama_local")
            at = AppTest.from_file(str(app), default_timeout=30).run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Local Ollama model" in info.value for info in at.info))
            next(b for b in at.button if b.label == "Review posting").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[0].value, "0.920")
            self.assertEqual(at.session_state["case"].assessment.recommendation, "review_needed")
            # The UI holds a case snapshot; another scenario clears it.
            at.selectbox[0].set_value("ordinary").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.metric), 0)
            next(b for b in at.button if b.label == "Review posting").click().run()
            self.assertEqual(at.metric[0].value, "0.250")
            self.assertEqual(at.session_state["case"].assessment.recommendation, "review_needed")
            self.assertFalse(audit_path.exists())
            self.assertFalse(any(b.label == "Confirm and save local decision" for b in at.button))
            self.assertFalse(any(i.label == "Reviewer name" for i in at.text_input))
            self.assertTrue(any("Recommendation: review needed" in i.value for i in at.info))



if __name__ == "__main__":
    unittest.main()
