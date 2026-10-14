"""Full-screen command center: opencode-style centered picker for slash commands.

Triggered by a bare "/" (or "/query") at the REPL prompt. Shows a centered
panel with a search field, commands grouped into sections with descriptions,
and keyboard navigation. Falls back to the printed palette when stdin is not
a TTY (scripted/piped use).
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame


@dataclass
class CenterItem:
    name: str  # "/compact"
    desc: str = ""
    kind: str = "command"  # "command" | "skill"


def collect_palette_items(repl) -> list[CenterItem]:
    """Gather every invokable command/skill. Same sources as the old palette."""
    items: list[CenterItem] = []
    seen: set[str] = set()

    def add(name: str, desc: str, kind: str = "command") -> None:
        if name in seen:
            return
        seen.add(name)
        items.append(CenterItem(name=name, desc=desc or "", kind=kind))

    for cmd in getattr(repl, "_original_built_ins", []):
        if cmd == "/":
            continue
        add(cmd, "")

    try:
        for cmd in repl.command_registry.list_commands():
            cmd_name = f"/{cmd.name}"
            if cmd_name in seen:
                continue
            alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            add(f"{cmd_name}{alias_str}", cmd.description or "")
    except Exception:
        pass

    try:
        from src.skills.loader import get_all_skills

        cwd = repl.tool_context.cwd or repl.tool_context.workspace_root
        skills = sorted(get_all_skills(project_root=cwd), key=lambda s: s.name.lower())
        for s in skills:
            add(f"/{s.name}", (s.description or "").strip(), "skill")
    except Exception:
        pass

    return items


def filter_items(items: list[CenterItem], query: str) -> list[tuple[str, list[CenterItem]]]:
    """Filter into (section, items) groups, Commands first, then Skills."""
    q = query.strip().lower()

    def match(it: CenterItem) -> bool:
        return not q or q in it.name.lower() or q in it.desc.lower()

    cmds = sorted([it for it in items if it.kind != "skill" and match(it)], key=lambda it: it.name.lower())
    skills = sorted([it for it in items if it.kind == "skill" and match(it)], key=lambda it: it.name.lower())
    groups: list[tuple[str, list[CenterItem]]] = []
    if cmds:
        groups.append(("Commands", cmds))
    if skills:
        groups.append(("Skills", skills))
    return groups


CENTER_STYLE = Style.from_dict(
    {
        "frame.border": "#4a4a4a",
        "cc.title": "bold #ffffff",
        "cc.esc": "#8a8a8a",
        "cc.search.label": "#8a8a8a",
        "cc.search.text": "#ffffff",
        "cc.search.cursor": "bg:#ffffff #000000",
        "cc.section": "bold #d4a017",
        "cc.name": "#e8e8e8",
        "cc.name.skill": "#d48ad4",
        "cc.desc": "#7a7a7a",
        "cc.selected": "bg:#7a5c14 bold #ffffff",
        "cc.hint": "#666666",
        "cc.empty": "#666666 italic",
    }
)

LIST_HEIGHT = 12


class CommandCenter:
    def __init__(self, items: list[CenterItem], initial_query: str = "") -> None:
        self.items = items
        self.query = initial_query or ""
        self.selected = 0  # index into flat selectable list
        self.scroll = 0
        self.app: Application | None = None

    # -- model ---------------------------------------------------------
    def groups(self) -> list[tuple[str, list[CenterItem]]]:
        return filter_items(self.items, self.query)

    def flat_items(self) -> list[CenterItem]:
        out: list[CenterItem] = []
        for _, group in self.groups():
            out.extend(group)
        return out

    def _clamp(self) -> None:
        n = len(self.flat_items())
        if n == 0:
            self.selected = 0
            self.scroll = 0
            return
        self.selected = max(0, min(self.selected, n - 1))
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + LIST_HEIGHT:
            self.scroll = self.selected - LIST_HEIGHT + 1

    def move(self, delta: int) -> None:
        if self.flat_items():
            self.selected = (self.selected + delta) % len(self.flat_items())
            self._clamp()

    def type_char(self, ch: str) -> None:
        self.query += ch
        self.selected = 0
        self.scroll = 0

    def backspace(self) -> None:
        self.query = self.query[:-1]
        self.selected = 0
        self.scroll = 0

    # -- rendering ------------------------------------------------------
    def _fragments(self):
        frags: list[tuple[str, str]] = []
        # title row
        title, esc = "Commands", "esc"
        pad = max(2, 66 - len(title) - len(esc))
        frags += [("class:cc.title", title), ("", " " * pad), ("class:cc.esc", esc), ("", "\n\n")]
        # search row
        frags += [
            ("class:cc.search.label", "Search "),
            ("class:cc.search.text", self.query),
            ("class:cc.search.cursor", " "),
            ("", "\n\n"),
        ]
        flat = self.flat_items()
        if not flat:
            frags.append(("class:cc.empty", "  No matches\n"))
        else:
            self._clamp()
            # rows carry their section so headers render above the first
            # visible item of each group
            rows: list[tuple[str, CenterItem]] = []
            for section, group in self.groups():
                for it in group:
                    rows.append((section, it))
            start, end = self.scroll, self.scroll + LIST_HEIGHT
            last_section = ""
            for item_idx, (section, it) in enumerate(rows):
                if not (start <= item_idx < end):
                    continue
                if section != last_section:
                    frags.append(("class:cc.section", f"  {section}\n"))
                    last_section = section
                is_sel = item_idx == self.selected
                name_style = "class:cc.selected" if is_sel else ("class:cc.name.skill" if it.kind == "skill" else "class:cc.name")
                name = f"  {it.name}"
                desc = it.desc.strip().replace("\n", " ")
                max_desc = 64 - len(name) - 2
                if desc and max_desc > 8:
                    if len(desc) > max_desc:
                        desc = desc[: max_desc - 1] + "…"
                    line = name + " " * (64 - len(name) - len(desc)) + desc
                else:
                    line = name
                frags.append((name_style, line + "\n"))
        frags += [("", "\n"), ("class:cc.hint", "  enter run · esc close · ↑↓ navigate · type to filter")]
        return frags

    # -- app ------------------------------------------------------------
    def run(self) -> str | None:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            self.move(-1)

        @kb.add("down")
        def _down(event):
            self.move(1)

        @kb.add("enter")
        def _enter(event):
            flat = self.flat_items()
            if flat and 0 <= self.selected < len(flat):
                event.app.exit(result=flat[self.selected].name)
            else:
                event.app.exit(result=None)

        @kb.add("escape")
        def _esc(event):
            event.app.exit(result=None)

        @kb.add("c-c")
        def _cc(event):
            event.app.exit(result=None)

        @kb.add("backspace")
        def _bs(event):
            self.backspace()

        @kb.add(Keys.Any)
        def _type(event):
            data = event.data
            if data and len(data) == 1 and data.isprintable():
                self.type_char(data)

        panel = Frame(
            body=HSplit([Window(content=FormattedTextControl(text=self._fragments), height=LIST_HEIGHT + 8)]),
            style="class:frame",
        )
        layout = Layout(
            HSplit(
                [
                    Window(),
                    VSplit([Window(width=14), panel, Window(width=14)]),
                    Window(),
                ]
            )
        )
        self.app = Application(
            layout=layout, key_bindings=kb, style=CENTER_STYLE, full_screen=True, mouse_support=False
        )
        return self.app.run()


def run_command_center(items: list[CenterItem], initial_query: str = "") -> str | None:
    """Open the full-screen picker. Returns the chosen "/command" or None."""
    return CommandCenter(items, initial_query).run()
