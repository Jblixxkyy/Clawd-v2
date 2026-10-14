"""Tests for the full-screen command center (src/repl/command_center.py)."""

from src.repl.command_center import CenterItem, CommandCenter, filter_items


def _items():
    return [
        CenterItem("/compact", "Compact conversation", "command"),
        CenterItem("/model", "Pick a model", "command"),
        CenterItem("/explain-code", "Explains code", "skill"),
    ]


def test_filter_groups_commands_then_skills():
    groups = filter_items(_items(), "")
    assert [s for s, _ in groups] == ["Commands", "Skills"]
    assert [i.name for _, g in groups for i in g] == ["/compact", "/model", "/explain-code"]


def test_filter_query_matches_name_and_desc():
    groups = filter_items(_items(), "mod")
    assert [(s, [i.name for i in g]) for s, g in groups] == [("Commands", ["/model"])]
    groups = filter_items(_items(), "explains")
    assert [(s, [i.name for i in g]) for s, g in groups] == [("Skills", ["/explain-code"])]


def test_filter_no_matches():
    assert filter_items(_items(), "zzz-nope") == []


def test_filter_case_insensitive():
    groups = filter_items(_items(), "COMPACT")
    assert [i.name for _, g in groups for i in g] == ["/compact"]


def test_navigation_wraps():
    cc = CommandCenter(_items())
    assert cc.flat_items()[cc.selected].name == "/compact"
    cc.move(1)
    assert cc.flat_items()[cc.selected].name == "/model"
    cc.move(1)
    cc.move(1)  # wrap past the end
    assert cc.flat_items()[cc.selected].name == "/compact"
    cc.move(-1)  # wrap backwards
    assert cc.flat_items()[cc.selected].name == "/explain-code"


def test_typing_resets_selection():
    cc = CommandCenter(_items())
    cc.move(2)
    cc.type_char("m")
    cc.type_char("o")
    cc.type_char("d")
    assert cc.selected == 0
    assert cc.query == "mod"
    assert [i.name for i in cc.flat_items()] == ["/model"]


def test_backspace():
    cc = CommandCenter(_items())
    cc.type_char("m")
    cc.type_char("o")
    cc.backspace()
    assert cc.query == "m"


def test_initial_query_prefills():
    cc = CommandCenter(_items(), initial_query="mod")
    assert cc.query == "mod"
    assert [i.name for i in cc.flat_items()] == ["/model"]


def test_fragments_render_without_error():
    cc = CommandCenter(_items())
    frags = cc._fragments()
    text = "".join(t for _, t in frags)
    assert "Commands" in text
    assert "/compact" in text
    assert "Skills" in text
    assert "enter run" in text


def test_fragments_empty_state():
    cc = CommandCenter(_items(), initial_query="zzz-nope")
    text = "".join(t for _, t in cc._fragments())
    assert "No matches" in text
