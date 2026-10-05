import pytest

from codeguardian.ai.edit import AIEdit, apply_ai_edit


def test_delete_edit_changes_exactly_one_line():
    source = "one\ntwo\nthree\n"
    candidate = apply_ai_edit(source, AIEdit("delete", 2))
    assert candidate == "one\nthree\n"


def test_replace_edit_changes_exactly_one_line():
    source = "one\ntwo\nthree\n"
    candidate = apply_ai_edit(
        source,
        AIEdit("replace", 2, "TWO"),
    )
    assert candidate == "one\nTWO\nthree\n"


def test_insert_before_preserves_existing_lines():
    source = "one\nthree\n"
    candidate = apply_ai_edit(
        source,
        AIEdit("insert_before", 2, "two"),
    )
    assert candidate == "one\ntwo\nthree\n"


def test_insert_after_preserves_existing_lines():
    source = "one\ntwo\n"
    candidate = apply_ai_edit(
        source,
        AIEdit("insert_after", 2, "three"),
    )
    assert candidate == "one\ntwo\nthree\n"


def test_delete_rejects_content():
    with pytest.raises(ValueError, match="must not contain"):
        apply_ai_edit("one\ntwo\n", AIEdit("delete", 1, "bad"))


def test_edit_rejects_out_of_range_line():
    with pytest.raises(ValueError, match="outside"):
        apply_ai_edit("one\n", AIEdit("delete", 2))


def test_edit_rejects_unknown_operation():
    with pytest.raises(ValueError, match="unsupported"):
        apply_ai_edit("one\n", AIEdit("rewrite_file", 1))
