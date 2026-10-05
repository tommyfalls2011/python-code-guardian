from codeguardian.ai.localized import (
    SourceWindow,
    apply_source_window,
    extract_source_window,
)


def test_extract_source_window_is_bounded():
    source = "".join(f"line_{number}\n" for number in range(1, 11))

    window = extract_source_window(source, 5, context=2)

    assert window.start_line == 3
    assert window.end_line == 7
    assert window.source == (
        "line_3\nline_4\nline_5\nline_6\nline_7\n"
    )


def test_extract_source_window_clamps_at_start():
    source = "one\ntwo\nthree\n"

    window = extract_source_window(source, 1, context=4)

    assert window.start_line == 1
    assert window.end_line == 3
    assert window.source == source


def test_apply_source_window_changes_only_window():
    source = "one\ntwo\nthree\nfour\nfive\n"
    window = SourceWindow(2, 4, "two\nthree\nfour\n")

    candidate = apply_source_window(
        source,
        window,
        "TWO\nTHREE\nFOUR\n",
    )

    assert candidate == "one\nTWO\nTHREE\nFOUR\nfive\n"


def test_apply_source_window_rejects_stale_source():
    source = "one\ntwo\nthree\n"
    window = SourceWindow(2, 2, "different\n")

    try:
        apply_source_window(source, window, "TWO\n")
    except ValueError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("stale source window was accepted")


def test_extract_source_window_rejects_bad_line():
    try:
        extract_source_window("one\n", 2)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("out-of-range line was accepted")
