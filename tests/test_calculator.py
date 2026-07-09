import pytest

from src.calculator import (
    expression_has_unsupported_calls,
    normalize_expression,
    safe_eval,
)
from src.postprocess import (
    finalize_answer,
    is_percentage_question,
    maybe_fix_percentage_scale,
)
from src.router import is_reask_question, route_turn


class TestCalculator:
    def test_safe_eval_basic(self) -> None:
        assert safe_eval("206588 - 181001") == 25587.0
        assert safe_eval("(15-19)/19") == pytest.approx(-0.210526, rel=1e-4)

    def test_normalize_expression_strips_symbols(self) -> None:
        assert normalize_expression("= $1,234 + 5") == "1234 + 5"
        assert normalize_expression("(15-19)/19") == "(15-19)/19"

    def test_unsupported_calls_rejected(self) -> None:
        assert expression_has_unsupported_calls("min(1, 2)")
        with pytest.raises(ValueError):
            normalize_expression("min(1, 2)")


class TestPostprocess:
    def test_percentage_question_detection(self) -> None:
        assert is_percentage_question("what was the percentage change?")
        assert not is_percentage_question("what was revenue in 2007?")

    def test_percentage_scale_fix(self) -> None:
        assert maybe_fix_percentage_scale(-21.05, "percentage change", True) == pytest.approx(
            -0.2105
        )
        assert maybe_fix_percentage_scale(60.94, "exercise price in 2007", False) == 60.94

    def test_finalize_answer(self) -> None:
        assert finalize_answer(-21.05, "so what was the percentage change?", True) == pytest.approx(
            -0.2105
        )


class TestRouter:
    def test_turn_zero_is_two_step(self) -> None:
        assert route_turn(0, "what was revenue?", has_type2=False) == "two_step"

    def test_math_follow_up_is_single(self) -> None:
        assert (
            route_turn(2, "what was the difference between the two years?", has_type2=False)
            == "single"
        )

    def test_reask_is_two_step(self) -> None:
        assert is_reask_question("and what was it in 2005?")
        assert route_turn(1, "and what was it in 2005?", has_type2=False) == "two_step"
