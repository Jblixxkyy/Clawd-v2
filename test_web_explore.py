"""Simple exploration script for the Clawd web app."""
import sys
import os
sys.path.insert(0, r'D:\clawd\Clawd-v2')

def test_imports():
    """Test that all web app modules can be imported."""
    print("Testing imports...")
    
    # Test basic imports
    from src.web_app import _HTML, _AppState, _Job, _make_handler
    print("  ✓ _HTML, _AppState, _Job, _make_handler imported")
    
    from src.agent import Session
    print("  ✓ Session imported")
    
    from src.config import get_default_provider, get_provider_config, load_config, save_config, set_api_key
    print("  ✓ Config functions imported")
    
    from src.providers import PROVIDER_INFO, get_provider_class
    print("  ✓ Provider functions imported")
    
    from src.tool_system.agent_loop import (
        AgentLoopCancelled, ToolEvent, run_agent_loop,
        summarize_tool_result, summarize_tool_use
    )
    print("  ✓ Agent loop functions imported")
    
    from src.tool_system.context import ToolContext
    print("  ✓ ToolContext imported")
    
    from src.tool_system.defaults import build_default_registry
    print("  ✓ build_default_registry imported")
    
    print("\nAll imports successful!\n")

def test_html_content():
    """Test that the HTML has expected content."""
    print("Testing HTML content...")
    from src.web_app import _HTML
    
    expected_markers = [
        'id="sidebar"',
        'id="welcome"',
        'id="newSession"',
        'id="sessionsList"',
        'className="markdown-body"',
        'function renderMarkdown',
        'function renderTimelineItem',
        'Used',
        'function isTableSeparator',
        'activity-pulse',
        'activity==="typing"?"Typing":"Thinking"',
    ]
    
    missing = []
    for marker in expected_markers:
        if marker not in _HTML:
            missing.append(marker)
    
    if missing:
        print(f"  ✗ Missing markers: {missing}")
    else:
        print("  ✓ All HTML markers present")
    
    # Check for streaming and session features
    streaming_features = ['EventSource', '/api/jobs/', 'stream?last_seq=', '/api/session/rename', 'diff-block', 'copy-button']
    for feature in streaming_features:
        if feature not in _HTML:
            print(f"  ! Missing streaming feature: {feature}")
    
    print(f"\nHTML size: {len(_HTML)} characters\n")

def test_core_functionality():
    """Test core web app functionality."""
    print("Testing core functionality...")
    
    # Test _source_signature
    from src.web_app import _source_signature
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        source_root = os.path.join(tmp, 'src')
        os.makedirs(source_root)
        with open(os.path.join(source_root, 'test.py'), 'w') as f:
            f.write('VERSION = 1\n')
        sig1 = _source_signature(source_root)
        
        with open(os.path.join(source_root, 'test.py'), 'w') as f:
            f.write('VERSION = 2\n')
        sig2 = _source_signature(source_root)
        
        if sig1 != sig2:
            print("  ✓ _source_signature detects changes")
        else:
            print("  ✗ _source_signature does not detect changes")
    
    # Test _diff_stat
    from src.web_app import _diff_stat
    diff = "@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 3\n"
    result = _diff_stat(diff)
    if result == {'added': 1, 'removed': 1}:
        print("  ✓ _diff_stat works correctly")
    else:
        print(f"  ✗ _diff_stat returned: {result}")
    
    # Test _drive_label
    from src.web_app import _drive_label
    from pathlib import PurePosixPath, PureWindowsPath
    
    posix_result = _drive_label(PurePosixPath("/tmp/project"))
    windows_result = _drive_label(PureWindowsPath("C:\\project"))
    
    if posix_result == "/" and windows_result == "C:":
        print("  ✓ _drive_label handles POSIX and Windows anchors")
    else:
        print(f"  ✗ _drive_label: posix={posix_result}, windows={windows_result}")
    
    print()

def test_job_lifecycle():
    """Test job creation and state management."""
    print("Testing job lifecycle...")
    
    from src.web_app import _AppState, _Job, _make_handler, _job_emit
    from src.tool_system.context import ToolContext
    from src.agent import Session
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp).resolve()
        state = _AppState(
            workspace=workspace,
            provider_name='nvidia',
            provider=type('P', (), {'model': 'test-model'})(),
            session=Session.create('nvidia', 'test-model'),
            registry=object(),
            tool_context=ToolContext(workspace_root=workspace),
        )
        
        # Create a job
        job = _Job(
            job_id="test-job",
            session_id=state.session.session_id,
            session=state.session,
            tool_context=state.tool_context,
        )
        
        # Test _job_emit
        _job_emit(job, "text", {"text": "hello"})
        if job.events and job.events[-1]["kind"] == "text":
            print("  ✓ _job_emit works")
        else:
            print("  ✗ _job_emit failed")
        
        # Test job status
        job.status = "completed"
        job.elapsed_ms = 1500
        job.usage = {"input_tokens": 100, "output_tokens": 200}
        _job_emit(job, "status", {"status": "completed", "error": None, "usage": job.usage, "elapsed_ms": job.elapsed_ms})
        
        # Check payload includes seq and usage
        payload = job.events[-1] if job.events else {}
        if payload.get("seq") == 1 and payload.get("usage") == {"input_tokens": 100, "output_tokens": 200}:
            print("  ✓ Job payload includes seq and usage")
        else:
            print(f"  ✗ Job payload issue: {payload}")
    
    print()

if __name__ == "__main__":
    test_imports()
    test_html_content()
    test_core_functionality()
    test_job_lifecycle()
    print("=" * 50)
    print("Exploration complete!")