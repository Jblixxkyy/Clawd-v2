import sys
sys.path.insert(0, r'D:\clawd\Clawd-v2')
from src.command_surface import run_skill_tool, parse_slash
from src.tool_system.defaults import build_default_registry
from src.tool_system.context import ToolContext
from pathlib import Path

registry = build_default_registry()
tool_context = ToolContext(workspace_root=Path(r'D:\clawd\Clawd-v2'))

result = run_skill_tool('customize-clawd', 'list', registry, tool_context)
print(f'Result type: {type(result)}')
if hasattr(result, 'output'):
    output = result.output
    print(f'Output type: {type(output)}')
    if isinstance(output, dict):
        print(f'Keys: {output.keys()}')
        print(f'Success: {output.get("success")}')
        print(f'Prompt: {output.get("prompt", "")[:200] if output.get("prompt") else "None"}')
    else:
        print(f'Output: {str(output)[:200]}')
if hasattr(result, 'is_error'):
    print(f'Error: {result.is_error}')
if hasattr(result, 'error'):
    print(f'Error msg: {result.error}')