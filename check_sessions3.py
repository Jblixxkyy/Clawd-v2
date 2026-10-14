import json
from pathlib import Path

session_dir = Path.home() / ".clawd" / "sessions"
if session_dir.exists():
    for f in sorted(session_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        print(f"=== {f.name} ===")
        with open(f) as fp:
            data = json.load(fp)
            for i, msg in enumerate(data.get("conversation", {}).get("messages", [])):
                content = msg.get("content")
                ctype = "str" if isinstance(content, str) else "list"
                print(f"  Msg {i}: role={msg.get('role')}, content_type={ctype}, _is_internal={msg.get('_is_internal')}")
                if isinstance(content, list):
                    for block in content:
                        print(f"    Block: {block.get('type')}")
                elif isinstance(content, str) and len(content) > 100:
                    # Skip unicode issues
                    print(f"    Text: (length {len(content)})")
                elif isinstance(content, str):
                    print(f"    Text: {content}")
else:
    print("No sessions dir")