#!/usr/bin/env python3
import ast, os

tools_dir = '/home/hatch/workspace/Clawd-Code/src/tool_system/tools'
specified_files = [
    'agent.py', 'ask_user_question.py', 'bash.py', 'brief.py', 'config.py',
    'cron.py', 'edit.py', 'glob.py', 'grep.py', 'lsp.py',
    'mcp.py', 'mcp_resources.py', 'misc.py', 'plan_mode.py', 'read.py',
    'send_user_message.py', 'skill.py', 'sleep.py', 'structured_output.py',
    'subagent.py', 'task_stop.py', 'tasks_v2.py', 'todo_write.py',
    'tool_search.py', 'team.py', 'web_fetch.py', 'web_search.py', 'write.py'
]

results = []

for fname in specified_files:
    fpath = os.path.join(tools_dir, fname)
    if not os.path.exists(fpath):
        results.append((fname, "FILE NOT FOUND"))
        continue
    with open(fpath) as f:
        content = f.read()
    try:
        tree = ast.parse(content)
        name = None
        desc = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'spec':
                        for call in ast.walk(item):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'ToolSpec':
                                for kw in call.keywords:
                                    if isinstance(kw, ast.keyword) and kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                        name = kw.value.value
                                    if isinstance(kw, ast.keyword) and kw.arg == 'description' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                        desc = kw.value.value
        if name and desc:
            # Clean up description to one line
            desc = desc.replace('\n', ' ').replace('\r', ' ')
            # Split by period and take first meaningful sentence
            sentences = [s.strip() for s in desc.split('.') if s.strip()]
            first_sentence = sentences[0] if sentences else desc.strip()
            # Remove leading dash/bullet
            while first_sentence and first_sentence[0] in '-•*':
                first_sentence = first_sentence[1:].strip()
            first_sentence = first_sentence.strip()
            # Ensure it's one line - cap words
            first_sentence = ' '.join(first_sentence.split()[:30])
            if not first_sentence.endswith(('.', '!', '?')):
                first_sentence = first_sentence.rstrip()
            results.append((name, first_sentence))
        else:
            results.append((fname, "COULD_NOT_EXTRACT"))
    except Exception as e:
        results.append((fname, f"ERROR: {e}"))

for name, desc in results:
    print(f"{name}: {desc}")