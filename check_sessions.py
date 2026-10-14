import json
from pathlib import Path

session_dir = Path.home() / ".clawd" / "sessions"
if session_dir.exists():
    for f in session_dir.glob("*.json"):
        print(f.name)
        with open(f) as fp:
            data = json.load(fp)
            for i, msg in enumerate(data.get("conversation", {}).get("messages", [])[:5]):
                content = msg.get("content")
                ctype = "str" if isinstance(content, str) else "list"
                print(f"  Msg {i}: role={msg.get('role')}, content_type={ctype}")
                if isinstance(content, list):
                    for block in content:
                        print(f"    Block: {block.get('type')}")
else:
    print("No sessions dir")