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
