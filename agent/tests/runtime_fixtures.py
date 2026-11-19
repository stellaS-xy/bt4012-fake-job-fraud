"""Explicit test-only dependency patches. Application code never imports this."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

from agent.core.config import Settings
from agent.tests.chat_model_fixture import ScriptedChatModel
from agent.tools import build_tools as real_build_tools
from agent.tools.lexicon import match_lexicon

SCORES = {'suspicious': .92, 'ordinary': .25, 'missing_info': .65}


def classifier_fixture(posting, scenario='ordinary'):
    score = SCORES[scenario]
    return dict(posting_id=posting.posting_id, model_score=score, cutoff=.7,
                flagged=score >= .7, model_version='classifier-test-fixture-v1',
                is_test_fixture=True, passed=False,
                review_reasons=['A test fixture cannot pass the real classifier check.'])


def lexicon_fixture(posting, settings):
    output = match_lexicon(posting)
    output.update(is_test_fixture=True, context_status='completed',
                  context_model='scripted-context-fixture', passed=False,
                  review_reasons=['Context observations are a test fixture.'])
    return output


def tools_fixture(posting, scenario, trace, settings):
    # Capture patched dependencies when constructing the real LangChain tools.
    with patch('agent.tools.score_real', side_effect=lambda p, s: classifier_fixture(p, scenario)), \
         patch('agent.tools.check_lexicon', side_effect=lexicon_fixture):
        return real_build_tools(posting, scenario, trace, settings)


@contextmanager
def offline_runtime():
    with ExitStack() as stack:
        stack.enter_context(patch('agent.core.runner.create_review_model', return_value=ScriptedChatModel()))
        stack.enter_context(patch('agent.core.runner.build_tools', side_effect=tools_fixture))
        stack.enter_context(patch('agent.core.runner.load_settings', return_value=Settings(model='scripted-chat-test-fixture')))
        yield
