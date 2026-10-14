from __future__ import annotations

from pathlib import Path

from .loader import load_tools_from_dir
from .registry import ToolRegistry
from .tools import (
    AskUserQuestionTool,
    BashTool,
    BriefTool,
    ConfigTool,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
    DelegateAgentTool,
    EnterPlanModeTool,
    EnterWorktreeTool,
    ExitPlanModeTool,
    ExitWorktreeTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
    LSPTool,
    ListMcpResourcesTool,
    MCPTool,
    NotebookEditTool,
    PowerShellTool,
    REPLTool,
    ReadMcpResourceTool,
    RemoteTriggerTool,
    SendMessageTool,
    SendUserMessageTool,
    SkillTool,
    SleepTool,
    StructuredOutputTool,
    TeamCreateTool,
    TeamDeleteTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
    TestingPermissionTool,
    TodoWriteTool,
    CompleteTaskTool,
    ReportFailureTool,
    ExitTool,
    WebFetchTool,
    WebSearchTool,
    SubagentOutputTool,
    SubagentTool,
)
from .tools.agent import AgentTool
from .tools.browser import BROWSER_TOOLS
from .tools.tool_search import ToolSearchTool


def build_default_registry(*, include_user_tools: bool = True) -> ToolRegistry:
    registry = ToolRegistry(
        tools=[
            SendUserMessageTool(),
            BashTool(),
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            GlobTool(),
            GrepTool(),
            WebFetchTool(),
            WebSearchTool(),
            SleepTool(),
            TaskStopTool(),
            ConfigTool(),
            MCPTool(),
            ListMcpResourcesTool(),
            ReadMcpResourceTool(),
            LSPTool(),
            SkillTool(),
            BriefTool(),
            AskUserQuestionTool(),
            TodoWriteTool(),
            # Brief 020: hard termination protocol — complete_task is the
            # only terminal success path; ReportFailure the typed failure
            # exit; Exit the low-level isComplete signal.
            CompleteTaskTool(),
            ReportFailureTool(),
            ExitTool(),
            TaskCreateTool(),
            TaskGetTool(),
            TaskListTool(),
            TaskUpdateTool(),
            TaskOutputTool(),
            TeamCreateTool(),
            TeamDeleteTool(),
            EnterPlanModeTool(),
            ExitPlanModeTool(),
            EnterWorktreeTool(),
            ExitWorktreeTool(),
            CronCreateTool(),
            CronListTool(),
            CronDeleteTool(),
            SendMessageTool(),
            StructuredOutputTool(),
            RemoteTriggerTool(),
            PowerShellTool(),
            NotebookEditTool(),
            REPLTool(),
            TestingPermissionTool(),
            DelegateAgentTool(),
        ]
        + list(BROWSER_TOOLS)
    )
    registry.register(AgentTool(registry))
    registry.register(SubagentTool(registry))
    registry.register(SubagentOutputTool())
    registry.register(ToolSearchTool(registry))

    if include_user_tools:
        user_dir = Path.home() / ".clawd" / "tools"
        for tool in load_tools_from_dir(user_dir):
            registry.register(tool)

    return registry
