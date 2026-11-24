"""Task 1.12: grade_prediction (SPEC 13.3)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from knotwork.knotcore.contracts import Answer, AnswerType, PremiseValue
from knotwork.knotcore.grading import grade_prediction

LABEL = Answer(type=AnswerType.LABEL, value="7", canonical="7")
LETTER = Answer(type=AnswerType.LABEL, value="C", canonical="C")
INT = Answer(type=AnswerType.INT, value=3, canonical="3")
FLOAT = Answer(type=AnswerType.FLOAT, value=47.0, tolerance=1.6, canonical="47")
ZONE = Answer(type=AnswerType.ZONE, value="red", canonical="red")
PREM = Answer(
    type=AnswerType.PREMISE,
    value=PremiseValue(premise_holds=False, value=47.0),
    tolerance=1.6,
    canonical="no; 47",
)

CASES: list[tuple[Any, Answer, float]] = [
    ("7", LABEL, 1.0), (" 7 ", LABEL, 1.0), ("I think it ends at 7.\nANSWER: 7", LABEL, 1.0),
    ("ANSWER: 6\nANSWER: 7", LABEL, 1.0), ("ANSWER: 7\nANSWER: 6", LABEL, 0.0), ("it is 7", LABEL, 0.0),
    ("answer: c", LETTER, 1.0), ("ANSWER: D", LETTER, 0.0), ("Ｃ", LETTER, 1.0),
    ("ANSWER: 3", INT, 1.0), ("three crossings, ANSWER: 3 times", INT, 1.0), ("ANSWER: 4", INT, 0.0),
    (3, INT, 1.0), (True, INT, 0.0), ("2 or 3\nANSWER: 2", INT, 0.0), ("none", INT, 0.0),
    ("ANSWER: 47", FLOAT, 1.0), ("ANSWER: 48.5", FLOAT, 1.0), ("ANSWER: 49", FLOAT, 0.0),
    ("about 46", FLOAT, 1.0), (47, FLOAT, 1.0), (45.0, FLOAT, 0.0), ("ANSWER: nan", FLOAT, 0.0),
    ("ANSWER: red", ZONE, 1.0), ("ANSWER: Red zone", ZONE, 1.0), ("ANSWER: red or amber", ZONE, 0.0),
    ("ANSWER: green", ZONE, 0.0), ("reddish", ZONE, 0.0),
    ("ANSWER: no; 47", PREM, 1.0), ("ANSWER: No, it reads 46.2", PREM, 1.0), ("ANSWER: yes; 47", PREM, 0.0),
    ("ANSWER: no; 52", PREM, 0.0), ({"premise_holds": False, "value": 47}, PREM, 1.0),
    ({"premise_holds": True, "value": 47}, PREM, 0.0), ("ANSWER: 47", PREM, 0.0),
]  # fmt: skip


@pytest.mark.parametrize(("pred", "answer", "want"), CASES)
def test_table(pred: Any, answer: Answer, want: float) -> None:
    assert grade_prediction(pred, answer) == want
    assert grade_prediction(pred, answer.model_dump(mode="json")) == want


@settings(max_examples=400, deadline=None)
@given(
    pred=st.one_of(
        st.text(),
        st.integers(),
        st.floats(),
        st.booleans(),
        st.none(),
        st.dictionaries(st.text(), st.text()),
    ),
    answer=st.sampled_from(
        [LABEL, LETTER, INT, FLOAT, ZONE, PREM, {"type": "float"}, {}, None, "junk"]
    ),
)
def test_fuzz_never_raises(pred: Any, answer: Any) -> None:
    assert grade_prediction(pred, answer) in (0.0, 1.0)


def test_stdlib_only_imports() -> None:
    src = (Path(__file__).resolve().parents[2] / "src/knotwork/knotcore/grading.py").read_text(
        encoding="utf-8"
    )
    names: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= {"json", "math", "re", "unicodedata", "typing", "__future__"}, names
