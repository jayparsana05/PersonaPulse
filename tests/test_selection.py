"""
Unit tests for the Select Topic stage (src/selection.py).

Covers the Prompt 3 review hardening:
- criteria are injected from config (single source of truth), system prompt
  describes role/contract only
- strict selected_index validation (int>=0 or null; no coercion)
- strict evaluation validation (index, fit_score, reason, exactly-once coverage)
- deterministic heuristic fallback on any invalid LLM response
- question/aspect validation and normalization (trim, dedupe, cap at 6)

The LLM is fully mocked (src.selection.complete_text); no network or API
keys are needed. Dummy env vars are installed before importing src.selection
so config validation passes without a populated .env file.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest.mock import patch

# Dummy env vars (see tests/test_ingestion_discovery.py) so src.config
# imports cleanly in CI / without a populated .env.
_REQUIRED_ENV = {
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-sb-key",
    "TELEGRAM_BOT_TOKEN": "test-bot",
    "TELEGRAM_CHAT_ID": "12345",
    "LINKEDIN_ACCESS_TOKEN": "test-li-token",
    "LINKEDIN_AUTHOR_URN": "urn:li:person:TEST",
    "LINKEDIN_TOKEN_EXPIRY_DATE": "2099-12-31",
}
for _key, _value in _REQUIRED_ENV.items():
    os.environ.setdefault(_key, _value)

from src.config import TOPIC_SELECTION_CRITERIA  # noqa: E402
from src.models import ResearchQuestion, TopicCandidate, TopicSelection  # noqa: E402
from src.selection import frame_question, select_topic  # noqa: E402
from src.selection import (  # noqa: E402
    _SELECTION_SYSTEM_PROMPT,
    _build_selection_prompt,
    _normalize_aspects,
)


def candidate(title="Agentic orchestration", score=0.7, url=None) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        url=url or f"https://example.com/{title.lower().replace(' ', '-')}",
        description=f"A description about {title}.",
        source="example.com",
        published="2026-09-19",
        search_score=score,
    )


def _dump(payload: dict) -> str:
    import json
    return json.dumps(payload)


def _topics():
    """Three candidates with distinct search scores (highest = Y @ 0.9)."""
    return [
        candidate(title="X", score=0.4),
        candidate(title="Y", score=0.9),
        candidate(title="Z", score=0.6),
    ]


def _full_evals(n=3, overrides=None):
    """A valid, complete evaluations list (one entry per candidate index)."""
    base = [
        {"index": 0, "fit_score": 40, "reason": "Thin engineering angle.", "strengths": "ok", "concerns": "thin"},
        {"index": 1, "fit_score": 55, "reason": "Decent but crowded.", "strengths": "ok", "concerns": "thin"},
        {"index": 2, "fit_score": 90, "reason": "Deep, actionable, novel.", "strengths": "deep", "concerns": "none"},
    ]
    evals = [dict(e) for e in base[:n]]
    for idx, changes in (overrides or {}).items():
        evals[idx].update(changes)
    return evals


def _assert_heuristic_fallback(testcase, result, expected_title="Y"):
    """Assert the result is a deterministic search_score fallback (not LLM)."""
    testcase.assertEqual(result.mode, TopicSelection.MODE_HEURISTIC)
    testcase.assertEqual(result.selected.title, expected_title)
    testcase.assertIn("search_score", result.reasoning)
    testcase.assertIn("no LLM evaluation used", result.reasoning)
    testcase.assertEqual(result.evaluations, [])


class SelectTopicEdgeCasesTest(unittest.TestCase):
    def test_no_candidates(self):
        result = select_topic([], query="agentic AI")
        self.assertIsNone(result.selected)
        self.assertEqual(result.mode, TopicSelection.MODE_EMPTY)
        self.assertIn("no topic candidates", result.reasoning.lower())
        self.assertEqual(result.criteria, list(TOPIC_SELECTION_CRITERIA))

    def test_no_candidates_skips_llm(self):
        with patch("src.selection.complete_text") as m:
            select_topic([])
        m.assert_not_called()

    def test_single_candidate_heuristic(self):
        topic = candidate(title="Only topic", score=0.5)
        result = select_topic([topic], use_llm=False)
        self.assertEqual(result.selected, topic)
        self.assertEqual(result.mode, TopicSelection.MODE_HEURISTIC)
        self.assertIn("0.50", result.reasoning)

    def test_multiple_candidates_heuristic_picks_highest_score(self):
        topics = [
            candidate(title="Low", score=0.2),
            candidate(title="High", score=0.95),
            candidate(title="Mid", score=0.5),
        ]
        result = select_topic(topics, use_llm=False)
        self.assertEqual(result.selected.title, "High")
        self.assertEqual(result.mode, TopicSelection.MODE_HEURISTIC)

    def test_duplicate_candidates_yield_one_selection(self):
        dup_a = candidate(title="Duplicate", score=0.8, url="https://example.com/dup")
        dup_b = candidate(title="Duplicate", score=0.8, url="https://example.com/dup")
        topics = [dup_a, dup_b, candidate(title="Unique", score=0.6)]
        result = select_topic(topics, use_llm=False)
        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.title, "Duplicate")

    def test_llm_markdown_fenced_json_is_parsed(self):
        payload = {"selected_index": 2, "reasoning": "Best.", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value="```json\n" + _dump(payload) + "\n```"):
            result = select_topic(_topics())
        self.assertEqual(result.selected.title, "Z")
        self.assertEqual(result.mode, TopicSelection.MODE_LLM)

    def test_empty_llm_text_falls_back_to_heuristic(self):
        with patch("src.selection.complete_text", return_value=""):
            result = select_topic(_topics())
        _assert_heuristic_fallback(self, result)


class SelectedIndexValidationTest(unittest.TestCase):
    """selected_index must be a plain int >= 0 (or null). No coercion."""

    def test_valid_integer_is_accepted(self):
        payload = {"selected_index": 2, "reasoning": "Best.", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            result = select_topic(_topics())
        self.assertEqual(result.mode, TopicSelection.MODE_LLM)
        self.assertEqual(result.selected.title, "Z")

    def test_null_means_none_fit(self):
        payload = {"selected_index": None, "reasoning": "Nothing meets the bar.", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            result = select_topic(_topics())
        self.assertIsNone(result.selected)
        self.assertEqual(result.mode, TopicSelection.MODE_NONE_FIT)
        self.assertIn("meets the bar", result.reasoning)
        self.assertEqual(len(result.evaluations), 3)

    def test_string_integer_is_rejected(self):
        payload = {"selected_index": "2", "reasoning": "x", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_float_is_rejected(self):
        payload = {"selected_index": 2.8, "reasoning": "x", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_boolean_true_is_rejected(self):
        payload = {"selected_index": True, "reasoning": "x", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_boolean_false_is_rejected(self):
        payload = {"selected_index": False, "reasoning": "x", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_negative_integer_is_rejected(self):
        payload = {"selected_index": -1, "reasoning": "x", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_out_of_range_integer_is_rejected(self):
        payload = {"selected_index": 999, "reasoning": "x", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))


class EvaluationValidationTest(unittest.TestCase):
    """evaluations must cover every candidate exactly once, strictly typed."""

    def test_complete_valid_evaluations(self):
        payload = {"selected_index": 1, "reasoning": "Y wins.", "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            result = select_topic(_topics())
        self.assertEqual(result.mode, TopicSelection.MODE_LLM)
        self.assertEqual(result.selected.title, "Y")
        self.assertEqual([e["index"] for e in result.evaluations], [0, 1, 2])
        self.assertEqual(result.evaluations[0]["reason"], "Thin engineering angle.")

    def test_evaluation_order_does_not_matter(self):
        evals = [_full_evals()[1], _full_evals()[2], _full_evals()[0]]  # scrambled
        payload = {"selected_index": 1, "reasoning": "Y.", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            result = select_topic(_topics())
        self.assertEqual(result.mode, TopicSelection.MODE_LLM)
        self.assertEqual([e["index"] for e in result.evaluations], [0, 1, 2])

    def test_missing_candidate_evaluation_is_rejected(self):
        evals = [_full_evals()[0], _full_evals()[2]]  # index 1 missing
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_duplicate_evaluation_index_is_rejected(self):
        evals = [_full_evals()[0], _full_evals(overrides={0: {"fit_score": 99}})[0], _full_evals()[1]]
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_out_of_range_evaluation_index_is_rejected(self):
        evals = _full_evals(overrides={1: {"index": 5}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_negative_evaluation_index_is_rejected(self):
        evals = _full_evals(overrides={1: {"index": -1}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_float_evaluation_index_is_rejected(self):
        evals = _full_evals(overrides={1: {"index": 1.5}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_boolean_evaluation_index_is_rejected(self):
        evals = _full_evals(overrides={1: {"index": True}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_string_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": "85"}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_negative_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": -1}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_over_100_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": 101}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_boolean_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": True}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_boolean_false_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": False}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_float_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": 85.5}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_null_is_rejected(self):
        evals = _full_evals(overrides={1: {"fit_score": None}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_fit_score_boundaries_are_accepted(self):
        for score in (0, 50, 100):
            evals = _full_evals(overrides={1: {"fit_score": score}})
            payload = {"selected_index": 1, "reasoning": "Y wins.", "evaluations": evals}
            with patch("src.selection.complete_text", return_value=_dump(payload)):
                result = select_topic(_topics())
            self.assertEqual(result.mode, TopicSelection.MODE_LLM)
            self.assertEqual(result.selected.title, "Y")
            self.assertEqual(result.evaluations[1]["fit_score"], score)

    def test_missing_reason_is_rejected(self):
        evals = _full_evals()
        evals[1] = {"index": 1, "fit_score": 55}  # no "reason" key at all
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_empty_reason_is_rejected(self):
        evals = _full_evals(overrides={1: {"reason": "   "}})
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))

    def test_non_dict_evaluation_is_rejected(self):
        evals = [_full_evals()[0], 42, _full_evals()[2]]
        payload = {"selected_index": 1, "reasoning": "x", "evaluations": evals}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            _assert_heuristic_fallback(self, select_topic(_topics()))


class ReasoningValidationTest(unittest.TestCase):
    """reasoning must be a non-empty string of at most 200 words."""

    def _with_reasoning(self, reasoning):
        payload = {"selected_index": 1, "reasoning": reasoning, "evaluations": _full_evals()}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            return select_topic(_topics())

    def _assert_valid(self, reasoning, expected="Y"):
        result = self._with_reasoning(reasoning)
        self.assertEqual(result.mode, TopicSelection.MODE_LLM)
        self.assertEqual(result.selected.title, expected)
        self.assertEqual(result.reasoning, reasoning.strip())

    def _assert_fallback(self, reasoning):
        result = self._with_reasoning(reasoning)
        self.assertEqual(result.mode, TopicSelection.MODE_HEURISTIC)
        self.assertEqual(result.selected.title, "Y")
        self.assertEqual(result.evaluations, [])

    def test_valid_short_reasoning_is_accepted(self):
        self._assert_valid("A concise explanation.")

    def test_valid_reasoning_is_stripped(self):
        result = self._with_reasoning("  Padded reasoning.  ")
        self.assertEqual(result.reasoning, "Padded reasoning.")

    def test_empty_string_is_rejected(self):
        self._assert_fallback("")

    def test_whitespace_only_is_rejected(self):
        self._assert_fallback("   ")

    def test_integer_is_rejected(self):
        self._assert_fallback(123)

    def test_boolean_is_rejected(self):
        self._assert_fallback(True)

    def test_list_is_rejected(self):
        self._assert_fallback([])

    def test_dict_is_rejected(self):
        self._assert_fallback({})

    def test_null_is_rejected(self):
        self._assert_fallback(None)

    def test_199_words_are_accepted(self):
        self._assert_valid(" ".join(["word"] * 199))

    def test_200_words_are_accepted(self):
        self._assert_valid(" ".join(["word"] * 200))

    def test_201_words_are_rejected(self):
        self._assert_fallback(" ".join(["word"] * 201))

    def test_over_200_words_falls_back_not_truncated(self):
        result = self._with_reasoning(" ".join(["word"] * 1000))
        self.assertEqual(result.mode, TopicSelection.MODE_HEURISTIC)


class CriteriaSourceOfTruthTest(unittest.TestCase):
    """TOPIC_SELECTION_CRITERIA is the single source; no hardcoding in prompts."""

    def test_custom_criteria_are_injected_into_prompt(self):
        custom = ["Cost efficiency", "Industry adoption"]
        with patch("src.selection.complete_text", return_value="{}") as m:
            select_topic([candidate()], criteria=custom)
        prompt = m.call_args.args[1]
        self.assertIn("Cost efficiency", prompt)
        self.assertIn("Industry adoption", prompt)
        self.assertNotIn(TOPIC_SELECTION_CRITERIA[0], prompt)
        self.assertIn("Evaluation criteria:", prompt)

    def test_default_criteria_are_injected_when_not_overridden(self):
        with patch("src.selection.complete_text", return_value="{}") as m:
            select_topic([candidate()])
        prompt = m.call_args.args[1]
        for criterion in TOPIC_SELECTION_CRITERIA:
            self.assertIn(criterion.strip(), prompt)

    def test_system_prompt_does_not_contain_criteria(self):
        for criterion in TOPIC_SELECTION_CRITERIA:
            label = criterion.split(":")[0].strip()  # e.g. "Engineering relevance"
            self.assertNotIn(label, _SELECTION_SYSTEM_PROMPT)

    def test_system_prompt_still_describes_role_and_contract(self):
        self.assertIn("topic-selection agent", _SELECTION_SYSTEM_PROMPT)
        self.assertIn("selected_index", _SELECTION_SYSTEM_PROMPT)
        self.assertIn("evaluations", _SELECTION_SYSTEM_PROMPT)

    def test_prompt_contains_all_candidate_metadata(self):
        topics = [candidate(title="A", score=0.9, url="https://ex.com/a")]
        topics[0].keywords = ["agentic", "orchestration"]
        prompt = _build_selection_prompt(topics, TOPIC_SELECTION_CRITERIA, "agentic AI")
        self.assertIn('"title": "A"', prompt)
        self.assertIn('"search_score": 0.9', prompt)
        self.assertIn('"keywords":', prompt)
        self.assertIn('"agentic"', prompt)


class FrameQuestionTest(unittest.TestCase):
    def test_none_topic_returns_none(self):
        self.assertIsNone(frame_question(None))

    def test_valid_question_and_aspects(self):
        payload = {"question": "Which orchestration framework scales best?", "aspects": ["cost", "reliability", "evaluation"]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Orchestration"))
        self.assertIsInstance(question, ResearchQuestion)
        self.assertEqual(question.question, payload["question"])
        self.assertEqual(question.aspects, ["cost", "reliability", "evaluation"])
        self.assertEqual(question.status, ResearchQuestion.STATUS_PROPOSED)
        self.assertEqual(question.priority, ResearchQuestion.PRIORITY_NORMAL)
        self.assertIsNotNone(question.created_at)

    def test_empty_question_uses_default(self):
        with patch("src.selection.complete_text", return_value=_dump({"question": "", "aspects": ["a"]})):
            question = frame_question(candidate(title="Groovy thing"))
        self.assertIn("Groovy thing", question.question)
        self.assertEqual(question.aspects, [])

    def test_non_string_question_uses_default(self):
        with patch("src.selection.complete_text", return_value=_dump({"question": 123, "aspects": ["a"]})):
            question = frame_question(candidate(title="Numeric thing"))
        self.assertIn("Numeric thing", question.question)
        self.assertEqual(question.aspects, [])

    def test_non_list_aspects_uses_default(self):
        with patch("src.selection.complete_text", return_value=_dump({"question": "Q?", "aspects": "cost"})):
            question = frame_question(candidate(title="Solo"))
        self.assertNotEqual(question.question, "Q?")
        self.assertEqual(question.aspects, [])

    def test_numeric_aspect_items_are_filtered(self):
        payload = {"question": "Q?", "aspects": ["a", 5, "b"]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Filtered"))
        self.assertEqual(question.question, "Q?")
        self.assertEqual(question.aspects, ["a", "b"])

    def test_empty_aspect_items_are_filtered(self):
        payload = {"question": "Q?", "aspects": ["a", "", "   ", "b"]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Spaced"))
        self.assertEqual(question.aspects, ["a", "b"])

    def test_duplicate_aspects_are_deduped_in_order(self):
        payload = {"question": "Q?", "aspects": ["a", "a", "b", "a", "b"]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Dup"))
        self.assertEqual(question.aspects, ["a", "b"])

    def test_more_than_six_aspects_are_capped(self):
        aspects = [f"aspect {i}" for i in range(9)]
        payload = {"question": "Q?", "aspects": aspects}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Many"))
        self.assertEqual(question.aspects, aspects[:6])

    def test_zero_valid_aspects_uses_default(self):
        payload = {"question": "Q?", "aspects": ["", "   ", 5, {"nested": 1}]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Zero"))
        self.assertNotEqual(question.question, "Q?")
        self.assertEqual(question.aspects, [])

    def test_one_to_two_valid_aspects_are_preserved(self):
        payload = {"question": "Q?", "aspects": ["cost"]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Few"))
        self.assertEqual(question.question, "Q?")
        self.assertEqual(question.aspects, ["cost"])

    def test_aspect_normalization_example(self):
        payload = {"question": "Q?", "aspects": ["  backend scaling  ", "", "backend scaling", "database design"]}
        with patch("src.selection.complete_text", return_value=_dump(payload)):
            question = frame_question(candidate(title="Norm"))
        self.assertEqual(question.aspects, ["backend scaling", "database design"])

    def test_normalize_aspects_helper_directly(self):
        self.assertEqual(_normalize_aspects(None), None)
        self.assertEqual(_normalize_aspects("cost"), None)
        self.assertEqual(_normalize_aspects(["a", 5, "b"]), ["a", "b"])
        self.assertEqual(_normalize_aspects(["  x  ", "x"]), ["x"])

    def test_llm_failure_uses_default_question(self):
        with patch("src.selection.complete_text", side_effect=RuntimeError("LLM down")):
            question = frame_question(candidate(title="Memory for agents"))
        self.assertEqual(question.status, ResearchQuestion.STATUS_PROPOSED)
        self.assertIn("Memory for agents", question.question)
        self.assertEqual(question.aspects, [])

    def test_use_llm_false_uses_default(self):
        topic = candidate(title="Tool use")
        question = frame_question(topic, use_llm=False)
        self.assertIn("Tool use", question.question)
        self.assertEqual(question.aspects, [])


class TopicSelectionSerializationTest(unittest.TestCase):
    def test_round_trip_with_selected(self):
        selection = TopicSelection(
            selected=candidate(title="Best", score=0.9),
            reasoning="Strongest fit.",
            criteria=list(TOPIC_SELECTION_CRITERIA),
            evaluations=[{"index": 0, "fit_score": 80, "reason": "good"}],
            mode=TopicSelection.MODE_LLM,
            created_at=datetime(2026, 9, 20, 15, 0, 0),
        )
        restored = TopicSelection.from_dict(selection.to_dict())
        self.assertEqual(restored, selection)
        self.assertEqual(restored.selected, candidate(title="Best", score=0.9))

    def test_round_trip_without_selected(self):
        selection = TopicSelection(reasoning="No candidates.", mode=TopicSelection.MODE_EMPTY)
        restored = TopicSelection.from_dict(selection.to_dict())
        self.assertEqual(restored, selection)
        self.assertIsNone(restored.selected)


if __name__ == "__main__":
    unittest.main()