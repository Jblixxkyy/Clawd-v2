import sys
sys.path.insert(0, r'D:\clawd\Clawd-v2')
from src.command_system.skills_integration import load_and_register_skills
from src.command_system.engine import create_command_context
from src.tool_system.context import ToolContext
from pathlib import Path

load_and_register_skills(project_root=r'D:\clawd\Clawd-v2')

class MockCostTracker:
    pass

class MockHistoryLog:
    pass

ctx = create_command_context(
    workspace_root=Path(r'D:\clawd\Clawd-v2'),
    conversation=None,
    cost_tracker=MockCostTracker(),
    history=MockHistoryLog(),
)

from src.command_system.registry import get_command_registry
reg = get_command_registry()
cmd = reg.get('customize-clawd')
print('Markdown content length:', len(cmd.markdown_content) if cmd.markdown_content else 0)
if cmd.markdown_content:
    print('First 200 chars:', cmd.markdown_content[:200])
else:
    print('Markdown content is None or empty')