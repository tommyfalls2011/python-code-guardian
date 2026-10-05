import pytest

from codeguardian.ai.edit import (
    AIEdit,
    apply_ai_edit,
    apply_ai_edits,
    normalize_ai_edits,
)


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

def test_apply_ai_edits_uses_original_line_numbers():
    source = (
        "first = 1\n"
        "second = 2\n"
        "third = 3\n"
        "fourth = 4\n"
    )

    result = apply_ai_edits(
        source,
        [
            AIEdit("replace", 2, "second = 20"),
            AIEdit(
                "insert_after",
                3,
                "inserted = 99",
            ),
        ],
    )

    assert result == (
        "first = 1\n"
        "second = 20\n"
        "third = 3\n"
        "inserted = 99\n"
        "fourth = 4\n"
    )


def test_apply_ai_edits_supports_mutable_default_pattern():
    source = (
        "def example(items=[]):\n"
        "    return len(items)\n"
    )

    result = apply_ai_edits(
        source,
        [
            AIEdit(
                "replace",
                1,
                "def example(items=None):",
            ),
            AIEdit(
                "insert_after",
                1,
                "    if items is None:\n"
                "        items = []",
            ),
        ],
    )

    assert result == (
        "def example(items=None):\n"
        "    if items is None:\n"
        "        items = []\n"
        "    return len(items)\n"
    )


def test_apply_ai_edits_rejects_empty_transaction():
    import pytest

    with pytest.raises(
        ValueError,
        match="transaction is empty",
    ):
        apply_ai_edits("value = 1\n", [])


def test_apply_ai_edits_rejects_too_many_edits():
    import pytest

    edits = [
        AIEdit("replace", 1, "a = 10"),
        AIEdit("replace", 2, "b = 20"),
        AIEdit("replace", 3, "c = 30"),
        AIEdit("replace", 4, "d = 40"),
    ]

    with pytest.raises(
        ValueError,
        match="exceeds maximum",
    ):
        apply_ai_edits(
            "a = 1\nb = 2\nc = 3\nd = 4\n",
            edits,
        )

def test_apply_ai_edits_rejects_conflicting_lines():
    import pytest

    with pytest.raises(
        ValueError,
        match="conflicting lines",
    ):
        apply_ai_edits(
            "value = 1\n",
            [
                AIEdit("replace", 1, "value = 2"),
                AIEdit("replace", 1, "value = 3"),
            ],
        )


def test_apply_ai_edits_rejects_delete_with_insertion():
    import pytest

    with pytest.raises(
        ValueError,
        match="conflicting lines",
    ):
        apply_ai_edits(
            "value = 1\n",
            [
                AIEdit("delete", 1),
                AIEdit(
                    "insert_after",
                    1,
                    "other = 2",
                ),
            ],
        )

def test_normalize_adjacent_inserted_block():
    edits = [
        AIEdit(
            "replace",
            6,
            "def example(items=None):",
        ),
        AIEdit(
            "insert_after",
            6,
            "    if items is None:",
        ),
        AIEdit(
            "insert_after",
            7,
            "        items = []",
        ),
    ]

    result = normalize_ai_edits(edits)

    assert result == [
        AIEdit(
            "replace",
            6,
            "def example(items=None):",
        ),
        AIEdit(
            "insert_after",
            6,
            (
                "    if items is None:\n"
                "        items = []"
            ),
        ),
    ]


def test_normalize_does_not_merge_unrelated_insertions():
    edits = [
        AIEdit(
            "insert_after",
            6,
            "    value = 1",
        ),
        AIEdit(
            "insert_after",
            7,
            "    other = 2",
        ),
    ]

    assert normalize_ai_edits(edits) == edits


def test_normalize_does_not_merge_without_deeper_indent():
    edits = [
        AIEdit(
            "insert_after",
            6,
            "    if ready:",
        ),
        AIEdit(
            "insert_after",
            7,
            "    other = 2",
        ),
    ]

    assert normalize_ai_edits(edits) == edits


def test_apply_ai_edit_delete_range():
    source = (
        "before = 1\n"
        "def duplicate():\n"
        "    value = 2\n"
        "    return value\n"
        "after = 3\n"
    )

    edit = AIEdit(
        operation="delete_range",
        line=2,
        end_line=4,
    )

    assert apply_ai_edit(source, edit) == (
        "before = 1\n"
        "after = 3\n"
    )


def test_apply_ai_edit_delete_range_requires_end_line():
    source = "one = 1\ntwo = 2\n"

    edit = AIEdit(
        operation="delete_range",
        line=1,
    )

    with pytest.raises(
        ValueError,
        match="requires end_line",
    ):
        apply_ai_edit(source, edit)


def test_apply_ai_edit_delete_range_rejects_reverse_range():
    source = "one = 1\ntwo = 2\n"

    edit = AIEdit(
        operation="delete_range",
        line=2,
        end_line=1,
    )

    with pytest.raises(
        ValueError,
        match="precedes start line",
    ):
        apply_ai_edit(source, edit)


def test_apply_ai_edit_delete_range_rejects_past_eof():
    source = "one = 1\ntwo = 2\n"

    edit = AIEdit(
        operation="delete_range",
        line=1,
        end_line=3,
    )

    with pytest.raises(
        ValueError,
        match="outside the source",
    ):
        apply_ai_edit(source, edit)


def test_apply_ai_edit_delete_range_rejects_content():
    source = "one = 1\ntwo = 2\n"

    edit = AIEdit(
        operation="delete_range",
        line=1,
        end_line=2,
        content="replacement",
    )

    with pytest.raises(
        ValueError,
        match="must not contain content",
    ):
        apply_ai_edit(source, edit)


def test_apply_ai_edit_non_range_rejects_end_line():
    source = "one = 1\ntwo = 2\n"

    edit = AIEdit(
        operation="replace",
        line=1,
        end_line=2,
        content="one = 3",
    )

    with pytest.raises(
        ValueError,
        match="must not specify end_line",
    ):
        apply_ai_edit(source, edit)


def test_apply_ai_edits_rejects_overlapping_delete_ranges():
    source = (
        "one = 1\n"
        "two = 2\n"
        "three = 3\n"
        "four = 4\n"
    )

    edits = [
        AIEdit(
            operation="delete_range",
            line=1,
            end_line=2,
        ),
        AIEdit(
            operation="delete_range",
            line=2,
            end_line=3,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="overlapping ranges",
    ):
        apply_ai_edits(source, edits)


def test_apply_ai_edits_rejects_edit_inside_delete_range():
    source = (
        "one = 1\n"
        "two = 2\n"
        "three = 3\n"
        "four = 4\n"
    )

    edits = [
        AIEdit(
            operation="delete_range",
            line=1,
            end_line=3,
        ),
        AIEdit(
            operation="replace",
            line=2,
            content="two = 20",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="overlapping ranges",
    ):
        apply_ai_edits(source, edits)


def test_apply_ai_edits_allows_nonoverlapping_range_and_edit():
    source = (
        "one = 1\n"
        "two = 2\n"
        "three = 3\n"
        "four = 4\n"
    )

    edits = [
        AIEdit(
            operation="delete_range",
            line=1,
            end_line=2,
        ),
        AIEdit(
            operation="replace",
            line=4,
            content="four = 40",
        ),
    ]

    assert apply_ai_edits(source, edits) == (
        "three = 3\n"
        "four = 40\n"
    )


def test_apply_ai_edits_rejects_insert_inside_delete_range():
    source = (
        "one = 1\n"
        "two = 2\n"
        "three = 3\n"
    )

    edits = [
        AIEdit(
            operation="delete_range",
            line=1,
            end_line=2,
        ),
        AIEdit(
            operation="insert_after",
            line=2,
            content="added = True",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="overlapping ranges",
    ):
        apply_ai_edits(source, edits)
