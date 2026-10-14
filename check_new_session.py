import json
from pathlib import Path

session_file = Path.home() / '.clawd' / 'sessions' / '20261014_004612_fb33c155.json'
with open(session_file) as f:
    data = json.load(f)
    for i, msg in enumerate(data.get('conversation', {}).get('messages', [])):
        content = msg.get('content')
        ctype = 'str' if isinstance(content, str) else 'list'
        print(f'  Msg {i}: role={msg.get("role")}, content_type={ctype}, _is_internal={msg.get("_is_internal")}')
        if isinstance(content, list):
            for block in content:
                print(f'    Block: {block.get("type")}')
        elif isinstance(content, str) and len(content) > 100:
            print(f'    Text: (length {len(content)})')
        elif isinstance(content, str):
            print(f'    Text: {content}')