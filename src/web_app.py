"""LAN-only browser UI backed by the existing Clawd agent loop."""

from __future__ import annotations

import base64
import difflib
import hashlib
import hmac
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from src.agent import Session
from src.config import (
    get_default_provider,
    get_provider_config,
    load_config,
    save_config,
    set_api_key,
)
from src.providers import PROVIDER_INFO, get_provider_class
from src.tool_system.agent_loop import (
    AgentLoopCancelled,
    ToolEvent,
    run_agent_loop,
    summarize_tool_result,
    summarize_tool_use,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry

_MAX_REQUEST_BYTES = 1_000_000
_INTERACTION_TIMEOUT_SECONDS = 300
_MAX_TOOL_DETAIL_CHARS = 20_000
_MAX_DIRECTORY_SEARCH_RESULTS = 100
_MAX_DIRECTORY_SEARCH_VISITED = 10_000
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SOURCE_ROOT = Path(__file__).resolve().parent
_RELOAD_POLL_SECONDS = 0.5
_RELOAD_STABLE_POLLS = 3

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>Clawd</title>
<script>try{var t=localStorage.getItem("clawd-theme")||"dark";if(t==="system")t=matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";document.documentElement.dataset.theme=t}catch(e){}</script>
<style>
:root{color-scheme:dark;--bg:#080808;--sidebar:#101010;--line:#252525;--text:#ececec;--muted:#858585;--subtle:#191919;--hover:#1b1b1b;--accent:#d6d6d6;--warning:#f0c66b;--danger:#ff7777;--panel:#151515;--border:#2b2b2b;--code-bg:#0d0d0d;--code-fg:#dddddd;--link:#9bbcff;--green:#78d99a;--blue:#81baff;--purple:#d6b5ff;--amber:#e6c477;--danger-soft:#3a1b1b;--warn-bg:#17140e;--warn-border:#4a4028;--hover-strong:#383838;--btn-primary-bg:#2f2f2f;--btn-primary-fg:#ffffff;--diff-add-bg:#0e2a1a;--diff-add-fg:#8fe3a8;--diff-del-bg:#3a1b1b;--diff-del-fg:#ff9d9d}
html[data-theme="light"]{color-scheme:light;--bg:#f7f7f8;--sidebar:#efeff1;--line:#e2e2e5;--text:#1f2023;--muted:#6b6d72;--subtle:#eaeaee;--hover:#e4e4e9;--accent:#3b3d42;--warning:#8a6d1a;--danger:#c62828;--panel:#ffffff;--border:#dcdce1;--code-bg:#f0f0f3;--code-fg:#3a3a40;--link:#2f5fd0;--green:#1a7f4b;--blue:#2456c9;--purple:#7a3fc5;--amber:#8a6b12;--danger-soft:#fdecec;--warn-bg:#fdf6e3;--warn-border:#e6d5a8;--hover-strong:#d7d7de;--btn-primary-bg:#1f2023;--btn-primary-fg:#ffffff;--diff-add-bg:#e3f5e9;--diff-add-fg:#1a7f4b;--diff-del-bg:#fdecec;--diff-del-fg:#c62828}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0}body{background:var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,textarea{font:inherit}button{color:inherit;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
.app-shell{display:grid;grid-template-columns:244px minmax(0,1fr);width:100%;height:100dvh;overflow:hidden}
.sidebar{display:flex;flex-direction:column;min-width:0;padding:12px 10px;background:var(--sidebar);border-right:1px solid var(--line)}
.brand-row{height:34px;display:flex;align-items:center;gap:9px;padding:0 8px 10px;font-size:14px;font-weight:600;letter-spacing:-.2px}
.brand-mark{display:grid;place-items:center;width:21px;height:21px;border-radius:6px;background:var(--btn-primary-bg);color:var(--btn-primary-fg);font-size:13px;font-weight:700}
.sidebar-close,.mobile-nav{display:none}
.new-session{display:flex;align-items:center;gap:9px;height:36px;margin:8px 2px 22px;padding:0 10px;border:1px solid var(--border);border-radius:7px;background:var(--panel);text-align:left;font-size:13px;box-shadow:0 1px 2px #00000008}
.new-session:hover,.workspace-card:hover,.session-item:hover{background:var(--hover)}
.plus{font-size:18px;font-weight:300;line-height:1}
.section-label{padding:0 9px 7px;color:var(--muted);font-size:11px;font-weight:500}
.workspace-card{display:flex;align-items:center;gap:9px;width:100%;padding:8px;border:0;border-radius:7px;background:transparent;text-align:left}
.project-mark{display:grid;place-items:center;flex:none;width:27px;height:27px;border:1px solid var(--border);border-radius:7px;background:var(--subtle);font-size:12px}
.project-copy{display:flex;flex:1;flex-direction:column;min-width:0}.project-name{overflow:hidden;font-size:12px;font-weight:500;text-overflow:ellipsis;white-space:nowrap}.project-path{overflow:hidden;color:var(--muted);font-size:10px;text-overflow:ellipsis;white-space:nowrap}
.chevron{color:var(--muted);font-size:12px}.sessions-label{margin-top:24px}
.session-list{display:flex;flex-direction:column;gap:2px;max-height:40vh;overflow:auto}
.session-item{display:flex;align-items:center;gap:5px;min-height:32px;padding:3px 5px 3px 9px;border:0;border-radius:6px;background:transparent;color:var(--muted);font-size:12px}
.session-item:hover{color:var(--text)}.session-item.active{background:var(--hover-strong);color:var(--text)}.session-open{flex:1;min-width:0;overflow:hidden;padding:3px 0;border:0;background:transparent;text-align:left;color:inherit}.session-title{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.session-state{flex:none;color:var(--green);font-size:10px}.session-cost{flex:none;color:var(--muted);font-size:10px}.session-delete{flex:none;padding:2px 5px;border:0;border-radius:4px;background:transparent;color:var(--muted);font-size:14px;line-height:1}.session-delete:hover{background:var(--danger-soft);color:var(--danger)}.sessions-empty{padding:3px 9px;color:var(--muted);font-size:11px}
.sidebar-spacer{flex:1}.sidebar-footer{display:flex;flex-direction:column;gap:7px;padding:10px 6px 2px;border-top:1px solid var(--line)}
.change-folder{padding:5px 0;border:0;background:transparent;text-align:left;color:var(--muted);font-size:11px}.change-folder:hover{color:var(--text)}
.footer-row{display:flex;align-items:center;justify-content:space-between;gap:8px}.footer-row .change-folder{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gear-button{display:grid;flex:none;place-items:center;width:26px;height:26px;padding:0;border:0;border-radius:6px;background:transparent;color:var(--muted);font-size:14px;line-height:1}.gear-button:hover{background:var(--hover);color:var(--text)}
.main-column{display:flex;flex-direction:column;min-width:0;min-height:0;background:var(--bg)}
.topbar{display:flex;flex:none;align-items:center;justify-content:space-between;gap:12px;height:48px;padding:0 20px;border-bottom:1px solid var(--line)}
.breadcrumb{display:flex;align-items:center;gap:8px;min-width:0;color:var(--muted);font-size:12px}.breadcrumb strong{overflow:hidden;color:var(--text);font-weight:500;text-overflow:ellipsis;white-space:nowrap}.crumb-separator{color:var(--muted)}
.conversation{display:flex;flex:1;flex-direction:column;min-height:0;position:relative}
.conversation-scroll{flex:1;min-height:0;overflow:auto;overscroll-behavior:contain}
.welcome{display:flex;flex-direction:column;align-items:center;width:min(100% - 40px,720px);margin:0 auto;padding:clamp(74px,17vh,150px) 0 32px;text-align:center}#welcome[hidden]{display:none}
.welcome-brand{font-size:32px;font-weight:600;letter-spacing:-1.5px}.welcome-subtitle{margin:7px 0 26px;color:var(--muted);font-size:14px}
.starter-prompts{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}.starter{padding:7px 11px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--muted);font-size:11px}.starter:hover{background:var(--hover-strong);color:var(--text)}
#messages{display:flex;flex-direction:column;gap:24px;width:min(100%,900px);margin:0 auto;padding:28px 28px 24px}
.message-row{display:flex;gap:10px;min-width:0;position:relative}.message-row.user{justify-content:flex-end}
.message-row{flex-wrap:wrap}.message-actions{flex-basis:100%;display:flex;gap:2px;justify-content:flex-start;padding:4px 2px 0;opacity:0;transition:opacity .15s}.message-row:hover .message-actions,.message-actions:focus-within{opacity:1}.message-row.user .message-actions{justify-content:flex-end;padding-right:4px}.msg-action{display:grid;place-items:center;width:26px;height:26px;padding:0;border:0;border-radius:6px;background:transparent;color:var(--muted);font-size:14px;line-height:1;cursor:pointer}.msg-action:hover{background:var(--hover);color:var(--text)}.msg-action:active{transform:scale(.92)}
.message-content{min-width:0;max-width:100%;overflow-wrap:anywhere;white-space:pre-wrap}.message-row.assistant .message-content{flex:1;padding:0 0 2px;line-height:1.6}
.markdown-body{min-width:0;max-width:100%;overflow-wrap:break-word;color:var(--text);line-height:1.6}
.markdown-body>*:first-child{margin-top:0}.markdown-body>*:last-child{margin-bottom:0}
.markdown-body p{margin:0 0 12px}.markdown-body h1,.markdown-body h2,.markdown-body h3,.markdown-body h4,.markdown-body h5,.markdown-body h6{color:var(--text);font-weight:600;line-height:1.35}
.markdown-body h1{margin:26px 0 12px;font-size:21px}.markdown-body h2{margin:23px 0 10px;font-size:18px}.markdown-body h3{margin:19px 0 8px;font-size:15px}.markdown-body h4,.markdown-body h5,.markdown-body h6{margin:17px 0 7px;font-size:14px;color:var(--muted)}
.markdown-body strong,.markdown-body b{font-weight:600;color:var(--text)}.markdown-body em{color:var(--text)}
.markdown-body ul,.markdown-body ol{margin:8px 0 14px;padding-left:26px}.markdown-body li{padding-left:3px;margin:5px 0}.markdown-body li::marker{color:var(--muted)}
.markdown-body a{color:var(--link);text-decoration:none}.markdown-body a:hover{text-decoration:underline;text-underline-offset:2px}
.markdown-body :not(pre)>code{padding:2px 5px;border:1px solid var(--border);border-radius:4px;background:var(--code-bg);color:var(--purple);font:0.9em "JetBrains Mono",Consolas,monospace}
.markdown-body pre{position:relative;max-width:100%;overflow:auto;margin:12px 0;padding:13px 15px;border:1px solid var(--border);border-radius:7px;background:var(--code-bg);color:var(--code-fg);font:12px/1.6 "JetBrains Mono",Consolas,monospace;tab-size:2}
.markdown-body pre code{font:inherit;white-space:pre}
.markdown-body blockquote{margin:12px 0;padding:2px 0 2px 14px;border-left:2px solid var(--border);color:var(--muted)}
.markdown-body hr{height:1px;margin:22px 0;border:0;background:var(--border)}
.markdown-table-wrap{max-width:100%;overflow-x:auto;margin:12px 0 16px;border:1px solid var(--border);border-radius:7px}
.markdown-body table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}
.markdown-body th,.markdown-body td{min-width:105px;padding:9px 12px;border-right:1px solid var(--border);border-bottom:1px solid var(--border);vertical-align:top}
.markdown-body th{background:var(--subtle);color:var(--text);font-weight:600}.markdown-body tr:last-child td{border-bottom:0}.markdown-body th:last-child,.markdown-body td:last-child{border-right:0}.markdown-body td p:last-child{margin-bottom:0}
.assistant-activity{display:flex;align-items:center;gap:7px;margin-top:7px;color:var(--muted);font-size:12px;white-space:nowrap}.assistant-activity[hidden]{display:none}.activity-dots{display:inline-flex;gap:3px}.activity-dots span{width:4px;height:4px;border-radius:50%;background:var(--muted);animation:activity-pulse 1.2s ease-in-out infinite}.activity-dots span:nth-child(2){animation-delay:.16s}.activity-dots span:nth-child(3){animation-delay:.32s}@keyframes activity-pulse{0%,60%,100%{opacity:.3;transform:translateY(0)}30%{opacity:1;transform:translateY(-3px)}}@media(prefers-reduced-motion:reduce){.activity-dots span{animation:none;opacity:.7}}
.message-row.user .message-content{max-width:min(82%,620px);padding:9px 12px;border-radius:9px;background:var(--subtle);line-height:1.55}
.tool-group{margin:0 0 0 32px;color:var(--muted);font-size:12px}.tool-group>summary{display:flex;align-items:center;gap:8px;min-height:25px;padding:2px 0;cursor:pointer;list-style:none}.tool-group>summary::-webkit-details-marker,.tool-card>summary::-webkit-details-marker{display:none}.tool-group>summary::before,.tool-card>summary::before{content:"›";flex:none;color:var(--muted);font-size:15px;line-height:12px;transition:transform .12s}.tool-group[open]>summary::before,.tool-card[open]>summary::before{transform:rotate(90deg)}.tool-group-label{color:var(--muted)}.tool-group-count{color:var(--text);font-weight:600}.tool-group-actions{color:var(--text)}.tool-group-status{margin-left:auto;color:var(--muted);font-size:10px}.tool-group[data-status="running"] .tool-group-status{color:var(--text)}
.tool-group-content{margin-left:8px;padding:2px 0 5px 12px;border-left:1px solid var(--border)}.tool-card{margin:0;color:var(--muted);font-size:11px}.tool-card>summary{display:flex;align-items:baseline;gap:8px;min-height:25px;padding:4px 2px;cursor:pointer;list-style:none}.tool-name{flex:none;color:var(--green);font-weight:600}.tool-card.tool-web .tool-name{color:var(--blue)}.tool-card.tool-file .tool-name{color:var(--purple)}.tool-card.tool-default .tool-name{color:var(--amber)}.tool-group-action{color:var(--green);font-weight:600}.tool-group-actions .tool-web{color:var(--blue)}.tool-group-actions .tool-file{color:var(--purple)}.tool-group-actions .tool-default{color:var(--amber)}.tool-group-actions .tool-error{color:var(--danger)}
.tool-summary{min-width:0;overflow:hidden;color:var(--muted);text-overflow:ellipsis;white-space:nowrap}.tool-status{flex:none;margin-left:auto;color:var(--muted);font-size:10px}.tool-card.tool-error .tool-name,.tool-card.tool-error .tool-status{color:var(--danger)}
.tool-note{display:block;margin:0 0 7px 27px;color:var(--muted);font-size:11px;overflow-wrap:anywhere}
.tool-detail{display:grid;grid-template-columns:minmax(0,1fr);gap:8px;padding:4px 0 10px 27px}
.tool-detail[hidden]{display:none}.tool-detail-label{margin-bottom:3px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.07em}
.tool-detail pre{max-height:280px;overflow:auto;margin:0;padding:9px 10px;border:1px solid var(--border);border-radius:5px;background:var(--code-bg);color:var(--code-fg);font:11px/1.55 "JetBrains Mono",Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.interaction{margin:10px 0 0 32px;padding:14px;border:1px solid var(--warn-border);border-radius:8px;background:var(--warn-bg);font-size:13px}
.interaction button{margin:7px 7px 0 0;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:var(--panel);font-size:12px}
.composer-zone{flex:none;width:min(100%,840px);margin:0 auto;padding:0 26px max(14px,env(safe-area-inset-bottom))}
.composer-card{padding:11px 13px 10px;border:1px solid var(--border);border-radius:12px;background:var(--panel);box-shadow:0 2px 6px #00000010;transition:border-color .12s}.composer-card:focus-within{border-color:var(--hover-strong)}
#prompt{display:block;width:100%;min-height:44px;max-height:180px;resize:none;overflow-y:auto;padding:4px 5px 9px;border:0;outline:0;background:transparent;color:var(--text);font:inherit;font-size:14px;line-height:1.5}
#prompt::placeholder{color:var(--muted)}#prompt:focus-visible{outline:0}
.composer-toolbar{display:flex;align-items:center;justify-content:space-between;gap:8px;padding-top:6px}.composer-actions{display:flex;align-items:center;gap:8px;margin-left:auto}
.composer-model{display:flex;align-items:center;min-width:0;max-width:72%;overflow:hidden;padding:4px 8px;border:0;border-radius:6px;background:transparent;color:var(--muted);font-size:10px;line-height:1;text-align:left;text-overflow:ellipsis;white-space:nowrap}.composer-model:hover{background:var(--hover);color:var(--text)}
.composer-model::after{content:"▾";flex:none;margin-left:5px;font-size:9px}
.send-button{display:grid;flex:none;place-items:center;width:29px;height:29px;padding:0;border:0;border-radius:8px;background:var(--btn-primary-bg);color:var(--btn-primary-fg);font-size:17px;line-height:1}
.send-button:hover{background:var(--hover-strong)}.send-button:disabled{background:var(--border)}.send-button[hidden]{display:none}
.stop-button{display:grid;flex:none;place-items:center;width:32px;height:32px;padding:0;border:0;border-radius:50%;background:#fff;color:#000;cursor:pointer;transition:transform .1s,opacity .15s}html[data-theme="light"] .stop-button{background:#000;color:#fff}.stop-button:hover{transform:scale(1.06)}.stop-button:active{transform:scale(.94)}.stop-button:disabled{cursor:wait}.stop-button .stop-icon{display:block;width:11px;height:11px;border-radius:2.5px;background:currentColor}.stop-button.stopping{animation:stop-pulse 1s ease-in-out infinite}@keyframes stop-pulse{0%,100%{opacity:1}50%{opacity:.4}}.stop-button[hidden]{display:none}
.composer-hint{padding-top:6px;color:var(--muted);text-align:center;font-size:10px}
.notice{width:min(100%,790px);margin:0 auto;padding:0 26px 8px;color:var(--warning);font-size:12px}
.folder-path{width:100%;padding:10px 11px;border:1px solid var(--border);border-radius:6px;background:var(--code-bg);color:var(--text);font:inherit;font-size:12px}
.primary-button{padding:9px 13px;border:0;border-radius:6px;background:var(--btn-primary-bg);color:var(--btn-primary-fg);font-size:12px}
.hint{color:var(--muted);font-size:10px}
.folder-picker{position:fixed;inset:0;z-index:8;display:grid;place-items:center;padding:20px;background:#0009}
.model-picker{position:fixed;inset:0;z-index:9;display:grid;place-items:end center;padding:0 20px max(92px,11vh);background:#0007}
.model-picker[hidden]{display:none}
.model-picker-dialog{width:min(100%,380px);max-height:min(58vh,440px);display:flex;flex-direction:column;padding:14px;border:1px solid var(--border);border-radius:12px;background:var(--panel);box-shadow:0 16px 60px #0009}
.model-picker-list{min-height:0;overflow:auto;margin-top:10px;padding:2px}
.model-picker-group{padding:8px 9px 4px;color:var(--muted);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.model-picker-item{display:flex;align-items:center;gap:8px;width:100%;min-height:34px;padding:6px 10px;border:0;border-radius:7px;background:transparent;text-align:left;color:var(--text);font-size:12px}
.model-picker-item:hover{background:var(--hover-strong)}
.model-picker-item.active{background:var(--subtle)}
.model-picker-item .tick{flex:none;width:14px;color:var(--green);font-size:11px}
.model-picker-empty{padding:12px 9px;color:var(--muted);font-size:11px}
.folder-picker[hidden]{display:none}.folder-dialog{width:min(100%,520px);max-height:min(80vh,650px);display:flex;flex-direction:column;padding:18px;border:1px solid var(--border);border-radius:10px;background:var(--panel);box-shadow:0 16px 60px #0008}
.folder-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:14px;font-weight:600}.folder-close{width:28px;height:28px;border:0;border-radius:5px;background:transparent;color:var(--muted);font-size:18px}.folder-close:hover{background:var(--subtle)}
.folder-search{display:flex;gap:8px;margin-top:12px}.folder-search-input{flex:1;min-width:0;padding:11px 12px;border:1px solid var(--border);border-radius:7px;background:var(--code-bg);color:var(--text);font:inherit;font-size:13px}.folder-location-label{margin-top:14px;color:var(--muted);font-size:11px}
.folder-list{min-height:100px;max-height:40vh;overflow:auto;margin-top:10px;padding:3px}
.folder-entry{display:flex;align-items:center;gap:10px;width:100%;min-height:36px;padding:7px 9px;border:0;border-radius:6px;background:transparent;text-align:left;color:var(--muted);font-size:12px}
.folder-entry:hover,.folder-entry.selected{background:var(--hover-strong);color:var(--text)}.folder-entry-icon{flex:none;width:18px;text-align:center}.folder-entry-copy{display:flex;min-width:0;flex:1;flex-direction:column;gap:2px}.folder-entry-name{overflow:hidden;color:var(--text);font-weight:500;text-overflow:ellipsis;white-space:nowrap}.folder-entry-path{overflow:hidden;color:var(--muted);font-size:10px;text-overflow:ellipsis;white-space:nowrap}
.folder-actions{display:flex;justify-content:flex-end;gap:8px;padding-top:12px;border-top:1px solid var(--border)}
.secondary-button{padding:8px 11px;border:1px solid var(--border);border-radius:6px;background:var(--subtle);color:var(--text);font-size:11px}
.settings-group{margin-top:16px}.settings-label{display:block;margin-bottom:6px;color:var(--muted);font-size:11px;font-weight:500}
.settings-select,.settings-input{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:var(--code-bg);color:var(--text);font:inherit;font-size:12px}
.settings-note{margin-top:6px;color:var(--muted);font-size:10px}
.theme-picker{display:flex;gap:6px}.theme-picker button{flex:1;padding:7px 0;border:1px solid var(--border);border-radius:6px;background:var(--subtle);color:var(--text);font-size:11px}.theme-picker button.active{border-color:var(--accent);background:var(--hover-strong);color:var(--text)}
.session-rename{flex:none;display:none;padding:2px 5px;border:0;border-radius:4px;background:transparent;color:var(--muted);font-size:11px;line-height:1}
.session-item:hover .session-rename{display:block}
.session-rename:hover{background:var(--hover-strong);color:var(--text)}
.session-rename-input{width:100%;padding:2px 4px;border:1px solid var(--border);border-radius:4px;background:var(--code-bg);color:var(--text);font-size:12px}
.diff-block{margin:0;padding:6px 0}
.diff-block div{padding:1px 8px;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.55 "JetBrains Mono",Consolas,monospace}
.diff-add{background:var(--diff-add-bg);color:var(--diff-add-fg)}
.diff-del{background:var(--diff-del-bg);color:var(--diff-del-fg)}
.diff-hunk{color:var(--muted)}
.copy-button{position:absolute;top:8px;right:8px;padding:3px 8px;border:1px solid var(--border);border-radius:5px;background:var(--panel);color:var(--muted);font-size:10px;cursor:pointer;opacity:.35;transition:opacity .15s}
.markdown-body pre:hover .copy-button,.copy-button:focus-visible{opacity:1}
.copy-button:hover{color:var(--text)}
/* Slim ChatGPT-style tool rows */
.tool-card>summary{min-height:22px;padding:3px 2px}
.tool-slim{flex:1;min-width:0;overflow:hidden;color:var(--text);font-weight:500;text-overflow:ellipsis;white-space:nowrap}
.tool-slim .tool-verb{font-weight:600;margin-right:6px}
.tool-slim .tool-verb{color:var(--green)}
.tool-card.tool-web .tool-slim .tool-verb{color:var(--blue)}
.tool-card.tool-file .tool-slim .tool-verb{color:var(--purple)}
.tool-card.tool-default .tool-slim .tool-verb{color:var(--amber)}
.tool-card.tool-error .tool-slim .tool-verb{color:var(--danger)}
.tool-meta{flex:none;display:flex;align-items:center;gap:8px}
.tool-diffstat:empty{display:none}
.tool-diffstat .add{color:var(--green);font-weight:600}
.tool-diffstat .del{color:var(--danger);font-weight:600}
.tool-elapsed:empty{display:none}
/* Persistent thought indicator */
.assistant-thought{margin-top:8px;color:var(--muted);font-size:12px}
/* Code block header + highlighting */
.code-header{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:-2px -2px 8px}
.code-lang{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}
.code-header .copy-button{position:static;opacity:.55}
.tok-kw{color:var(--purple);font-weight:600}
.tok-str{color:var(--green)}
.tok-com{color:var(--muted);font-style:italic}
.tok-num{color:var(--amber)}
.tok-fn{color:var(--blue)}
/* Sidebar chat search */
.session-search{width:calc(100% - 4px);margin:0 2px 8px;padding:7px 10px;border:1px solid var(--border);border-radius:6px;background:var(--code-bg);color:var(--text);font-size:12px}
.session-search::placeholder{color:var(--muted)}
.session-search:focus{outline:none;border-color:var(--hover-strong)}
/* Scroll-to-bottom */
.scroll-bottom{position:absolute;left:50%;bottom:88px;transform:translateX(-50%);z-index:4;display:grid;place-items:center;width:32px;height:32px;border:1px solid var(--border);border-radius:50%;background:var(--panel);color:var(--muted);font-size:15px;box-shadow:0 4px 14px #0006;cursor:pointer}
.scroll-bottom:hover{color:var(--text);background:var(--hover-strong)}
.scroll-bottom[hidden]{display:none}
/* In-place message editing */
.message-edit{width:100%;min-height:70px;padding:9px 12px;border:1px solid var(--border);border-radius:9px;background:var(--code-bg);color:var(--text);font:inherit;line-height:1.55;resize:vertical}
.message-edit-actions{display:flex;gap:8px;justify-content:flex-end;padding-top:6px}
/* Feedback toggle */
.msg-action.active{color:var(--green);opacity:1}
/* Topbar actions */
.topbar-right{display:flex;align-items:center;gap:2px}
.topbar-action{display:grid;place-items:center;width:30px;height:30px;padding:0;border:0;border-radius:6px;background:transparent;color:var(--muted);font-size:15px;cursor:pointer}
.topbar-action:hover{background:var(--hover);color:var(--text)}
.interaction strong{display:block;margin-bottom:6px}
.interaction p{margin:6px 0}
.mobile-backdrop{display:none}
@media(max-width:760px){.app-shell{grid-template-columns:minmax(0,1fr)}.sidebar{position:fixed;inset:0 auto 0 0;z-index:6;width:min(82vw,280px);transform:translateX(-102%);transition:transform .18s ease;box-shadow:8px 0 30px #00000012}.sidebar.open{transform:translateX(0)}.sidebar-close{display:block;margin-left:auto;width:26px;height:26px;border:0;border-radius:5px;background:transparent;color:var(--muted);font-size:17px}.mobile-nav{display:grid;place-items:center;width:28px;height:28px;padding:0;border:0;border-radius:5px;background:transparent;color:var(--muted);font-size:16px}.mobile-nav:hover{background:var(--hover)}.topbar{justify-content:flex-start;padding:0 12px}.mobile-backdrop.open{display:block;position:fixed;inset:0;z-index:5;background:#0005}.welcome{padding-top:clamp(64px,14vh,120px)}#messages{padding:24px 15px}.composer-zone{padding-right:14px;padding-left:14px}.notice{padding-right:15px;padding-left:15px}}
@media(max-width:420px){.welcome-brand{font-size:28px}.starter-prompts{gap:6px}.starter{font-size:10px}.breadcrumb{gap:5px}}
</style>
</head>
<body>
<div class="mobile-backdrop" id="mobileBackdrop"></div>
<div class="app-shell">
<aside class="sidebar" id="sidebar">
<div class="brand-row"><span class="brand-mark">C</span><span>Clawd</span><button class="sidebar-close" id="sidebarClose" type="button" aria-label="Close navigation">×</button></div>
<button class="new-session" id="newSession" type="button"><span class="plus">+</span><span>New session</span></button>
<div class="section-label">Workspace</div>
<button class="workspace-card" id="chooseWorkspace" type="button"><span class="project-mark">◆</span><span class="project-copy"><span class="project-name" id="projectName">Connecting…</span><span class="project-path" id="sidebarPath"></span></span><span class="chevron">⌄</span></button>
<div class="section-label sessions-label">Sessions</div>
<input class="session-search" id="sessionSearch" type="search" placeholder="Search chats" aria-label="Search chats" autocomplete="off">
<div class="session-list" id="sessionsList"><div class="sessions-empty">No saved chats</div></div>
<div class="sidebar-spacer"></div>
<div class="sidebar-footer"><div class="footer-row"><button class="change-folder" id="changeFolder" type="button">New chat in another folder</button><button class="gear-button" id="openSettings" type="button" aria-label="Settings" title="Settings">⚙</button></div></div>
</aside>
<div class="main-column">
<header class="topbar"><button class="mobile-nav" id="mobileNav" type="button" aria-label="Open navigation">☰</button><div class="breadcrumb"><span id="topProject">Workspace</span><span class="crumb-separator">/</span><strong id="sessionTitle">New session</strong></div><div class="topbar-right"><button class="topbar-action" id="exportChat" type="button" title="Export chat as markdown" aria-label="Export chat as markdown">⤓</button></div></header>
<section class="conversation">
<div class="conversation-scroll" id="conversationScroll">
<div class="welcome" id="welcome"><div class="welcome-brand">Clawd</div><p class="welcome-subtitle">What are we building today?</p><div class="starter-prompts"><button class="starter" type="button" data-prompt="Explain this codebase">Explain this codebase</button><button class="starter" type="button" data-prompt="Help me find a bug">Help me find a bug</button><button class="starter" type="button" data-prompt="Build a new feature">Build a new feature</button></div></div>
<div id="messages"></div>
</div>
<button class="scroll-bottom" id="scrollBottom" type="button" hidden aria-label="Scroll to bottom">↓</button>
<div class="notice" id="notice" aria-live="polite"></div>
<div class="composer-zone"><form id="composer"><div class="composer-card"><textarea id="prompt" placeholder="Ask anything about this project…" rows="2"></textarea><div class="composer-toolbar"><button class="composer-model" id="composerModel" type="button" title="Change provider and model"></button><span class="composer-actions"><button class="stop-button" id="stop" type="button" hidden aria-label="Stop response"><span class="stop-icon"></span></button><button class="send-button" id="send" type="submit" aria-label="Send message">↑</button></span></div></div><div class="composer-hint">Enter to send · Shift+Enter for a new line</div></form></div>
</section>
</div>
</div>
<section class="folder-picker" id="folderPicker" hidden>
<div class="folder-dialog" role="dialog" aria-modal="true" aria-labelledby="folderTitle"><div class="folder-heading"><span id="folderTitle">Start a new chat in a working folder</span><button class="folder-close" id="closeFolder" type="button" aria-label="Close folder picker">×</button></div>
<label class="folder-location-label" for="folderPath">Starting folder</label><div class="folder-search"><input class="folder-search-input" id="folderPath" type="text" aria-label="Starting folder path" autocomplete="off"><button class="secondary-button" id="goFolderPath" type="button">Go</button></div>
<label class="folder-location-label" for="folderSearch">Optional: search inside this folder and its subfolders</label><div class="folder-search"><input class="folder-search-input" id="folderSearch" type="search" aria-label="Search within the starting folder" placeholder="Search within this folder" autocomplete="off"></div>
<div class="folder-list" id="folderList"></div>
<div class="hint" id="folderPickerHint">Choose a starting folder, then optionally search inside it. Chats stay tied to their working folder.</div><div class="folder-actions"><button class="secondary-button" id="cancelFolder" type="button">Cancel</button><button class="primary-button" id="useFolder" type="button">Start chat here</button></div></div>
</section>
<section class="folder-picker" id="settingsPicker" hidden>
<div class="folder-dialog" role="dialog" aria-modal="true" aria-labelledby="settingsTitle"><div class="folder-heading"><span id="settingsTitle">Settings</span><button class="folder-close" id="closeSettings" type="button" aria-label="Close settings">×</button></div>
<div class="settings-group"><label class="settings-label" for="settingsProvider">Provider</label><select class="settings-select" id="settingsProvider"></select><div class="settings-note" id="settingsKeyNote"></div><input class="settings-input" id="settingsApiKey" type="password" placeholder="Paste API key to save it for this provider" autocomplete="off" style="margin-top:8px"></div>
<div class="settings-group"><label class="settings-label" for="settingsModel">Model</label><select class="settings-select" id="settingsModel"></select></div>
<div class="settings-group"><span class="settings-label">Theme</span><div class="theme-picker" id="themePicker"><button type="button" data-theme-value="dark">Dark</button><button type="button" data-theme-value="light">Light</button><button type="button" data-theme-value="system">System</button></div></div>
<div class="folder-actions"><button class="secondary-button" id="cancelSettings" type="button">Cancel</button><button class="primary-button" id="saveSettings" type="button">Save</button></div></div>
</section>
<section class="model-picker" id="modelPicker" hidden>
<div class="model-picker-dialog" role="dialog" aria-modal="true" aria-labelledby="modelPickerTitle"><div class="folder-heading"><span id="modelPickerTitle">Select a model</span><button class="folder-close" id="closeModelPicker" type="button" aria-label="Close model picker">×</button></div>
<div class="model-picker-list" id="modelPickerList"></div></div>
</section>
<script>
const messages=document.querySelector("#messages"),notice=document.querySelector("#notice"),promptBox=document.querySelector("#prompt"),sendButton=document.querySelector("#send"),stopButton=document.querySelector("#stop"),folderPicker=document.querySelector("#folderPicker"),folderPath=document.querySelector("#folderPath"),folderSearch=document.querySelector("#folderSearch"),folderList=document.querySelector("#folderList"),folderPickerHint=document.querySelector("#folderPickerHint"),useFolderButton=document.querySelector("#useFolder"),welcome=document.querySelector("#welcome"),sidebar=document.querySelector("#sidebar"),conversationScroll=document.querySelector("#conversationScroll"),scrollBottomBtn=document.querySelector("#scrollBottom"),sessionSearch=document.querySelector("#sessionSearch");
let stickToBottom=true,lastSessions=[];
function nearBottom(){return conversationScroll.scrollHeight-conversationScroll.scrollTop-conversationScroll.clientHeight<140}
function scrollIfStuck(){if(stickToBottom){conversationScroll.scrollTop=conversationScroll.scrollHeight}updateScrollButton()}
function updateScrollButton(){scrollBottomBtn.hidden=nearBottom()||!messages.children.length}
conversationScroll.addEventListener("scroll",()=>{stickToBottom=nearBottom();updateScrollButton()},{passive:true});
scrollBottomBtn.onclick=()=>{stickToBottom=true;conversationScroll.scrollTop=conversationScroll.scrollHeight;updateScrollButton()};
function hashStr(s){let h=0;for(let i=0;i<s.length;i++){h=(Math.imul(h,31)+s.charCodeAt(i))|0}return (h>>>0).toString(36)}
async function copyText(btn,text){try{await navigator.clipboard.writeText(text);const o=btn.textContent;btn.textContent="Copied";setTimeout(()=>{btn.textContent=o},1200)}catch(e){const o=btn.textContent;btn.textContent="Failed";setTimeout(()=>{btn.textContent=o},1200)}}
let activeSessionId=null;
const toolCards=new Map();
const activeJobs=new Map();
let activeToolGroup=null;
let directoryRequestId=0;
let directorySearchTimer=null;
let currentDirectory=null;
let selectedFolderPath=null;
async function watchForReload(){let version=null;const check=async()=>{try{const response=await fetch("/api/dev-version",{cache:"no-store"});if(!response.ok)return;const data=await response.json();if(!data.enabled)return;if(version!==null&&version!==data.version){location.reload();return}version=data.version}catch{}setTimeout(check,1000)};check()}
watchForReload();
async function api(path,options={}){const r=await fetch(path,{...options,headers:{"Content-Type":"application/json",...(options.headers||{})}});const data=await r.json();if(!r.ok)throw Error(data.error||"Request failed");return data}
function updateWorkspace(path){const name=path.split(/[\\/]/).filter(Boolean).pop()||path;document.querySelector("#projectName").textContent=name;document.querySelector("#sidebarPath").textContent=path;document.querySelector("#topProject").textContent=name;document.querySelector("#sessionTitle").textContent="New session"}
function updateSessionWorkspace(path){const name=path.split(/[\\/]/).filter(Boolean).pop()||path;document.querySelector("#projectName").textContent=name;document.querySelector("#sidebarPath").textContent=path;document.querySelector("#topProject").textContent=name}
function appendInline(parent,text){const pattern=/<br\s*\/?>|`([^`\n]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)|\*\*([\s\S]+?)\*\*|__([\s\S]+?)__|~~([\s\S]+?)~~|\*([^*\n]+)\*|_([^_\n]+)_/gi;let cursor=0,match;while((match=pattern.exec(text))){if(match.index>cursor)parent.append(document.createTextNode(text.slice(cursor,match.index).replace(/\\([\\`*_{}[\]()#+.!>-])/g,"$1")));if(/^<br/i.test(match[0]))parent.append(document.createElement("br"));else if(match[1]){const code=document.createElement("code");code.textContent=match[1];parent.append(code)}else if(match[2]){let url;try{url=new URL(match[3],location.href)}catch{}if(url&&(url.protocol==="https:"||url.protocol==="http:"||url.protocol==="mailto:")){const link=document.createElement("a");link.href=url.toString();link.rel="noopener noreferrer";link.target="_blank";appendInline(link,match[2]);parent.append(link)}else parent.append(document.createTextNode(match[0]))}else{const tag=match[4]||match[5]?"strong":match[6]?"del":"em",value=match[4]||match[5]||match[6]||match[7]||match[8],node=document.createElement(tag);appendInline(node,value);parent.append(node)}cursor=pattern.lastIndex}if(cursor<text.length)parent.append(document.createTextNode(text.slice(cursor).replace(/\\([\\`*_{}[\]()#+.!>-])/g,"$1")))}
function tableCells(line){let value=line.trim();if(value.startsWith("|"))value=value.slice(1);if(value.endsWith("|"))value=value.slice(0,-1);const cells=[];let cell="",escaped=false;for(const char of value){if(char==="|"&&!escaped){cells.push(cell.trim());cell="";continue}cell+=char;if(char==="\\")escaped=!escaped;else escaped=false}cells.push(cell.trim());return cells.map(value=>value.replace(/\\\|/g,"|"))}
function isTableSeparator(line){const cells=tableCells(line);return cells.length>1&&cells.every(cell=>/^:?-{3,}:?$/.test(cell))}
function isListLine(line){return /^\s{0,3}(?:[-+*]|\d+[.)])\s+/.test(line)}
function isBlockStart(lines,index){const line=lines[index]||"",next=lines[index+1]||"";return /^\s{0,3}(?:```|~~~)/.test(line)||/^\s{0,3}#{1,6}\s+/.test(line)||/^\s{0,3}>/.test(line)||/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)||isListLine(line)||(line.includes("|")&&next.includes("|")&&isTableSeparator(next))}
function renderMarkdown(root,text){root.replaceChildren();const lines=text.replace(/\r\n?/g,"\n").split("\n");let index=0;while(index<lines.length){if(!lines[index].trim()){index++;continue}const line=lines[index],fence=line.match(/^\s{0,3}(```+|~~~+)\s*([^\s`]*)?.*$/);if(fence){const marker=fence[1][0],codeLines=[];index++;while(index<lines.length&&!new RegExp("^\\s{0,3}"+marker+"{"+fence[1].length+",}\\s*$").test(lines[index]))codeLines.push(lines[index++]);if(index<lines.length)index++;const pre=document.createElement("pre"),code=document.createElement("code");const lang=(fence[2]||"").trim();if(lang)code.dataset.language=lang;code.textContent=codeLines.join("\n");const header=document.createElement("div");header.className="code-header";const langEl=document.createElement("span");langEl.className="code-lang";langEl.textContent=lang||"code";header.append(langEl);const copyBtn=document.createElement("button");copyBtn.type="button";copyBtn.className="copy-button";copyBtn.textContent="Copy";copyBtn.onclick=async()=>{try{await navigator.clipboard.writeText(code.textContent);copyBtn.textContent="Copied"}catch(e){copyBtn.textContent="Failed"}setTimeout(()=>{copyBtn.textContent="Copy"},1200)};header.append(copyBtn);pre.append(header,code);try{highlightCode(code)}catch(hlErr){}root.append(pre);continue}const heading=line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);if(heading){const element=document.createElement("h"+heading[1].length);appendInline(element,heading[2]);root.append(element);index++;continue}if(index+1<lines.length&&/^\s*(?:=+|-+)\s*$/.test(lines[index+1])&&line.trim()){const level=lines[index+1].trim().startsWith("=")?1:2,element=document.createElement("h"+level);appendInline(element,line.trim());root.append(element);index+=2;continue}if(/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)){root.append(document.createElement("hr"));index++;continue}if(isListLine(line)){const ordered=/^\s{0,3}\d+[.)]\s+/.test(line),list=document.createElement(ordered?"ol":"ul");while(index<lines.length&&isListLine(lines[index])&&/^\s{0,3}\d+[.)]\s+/.test(lines[index])===ordered){const item=document.createElement("li"),value=lines[index++].replace(/^\s{0,3}(?:[-+*]|\d+[.)])\s+/,"");appendInline(item,value);list.append(item)}root.append(list);continue}if(/^\s{0,3}>/.test(line)){const quote=document.createElement("blockquote");while(index<lines.length&&/^\s{0,3}>/.test(lines[index])){const value=lines[index++].replace(/^\s{0,3}>\s?/,"");if(quote.childNodes.length)quote.append(document.createElement("br"));appendInline(quote,value)}root.append(quote);continue}if(line.includes("|")&&index+1<lines.length&&lines[index+1].includes("|")&&isTableSeparator(lines[index+1])){const headers=tableCells(line),alignments=tableCells(lines[index+1]).map(cell=>cell.startsWith(":")?(cell.endsWith(":")?"center":"left"):(cell.endsWith(":")?"right":"left")),wrap=document.createElement("div"),table=document.createElement("table"),thead=document.createElement("thead"),headRow=document.createElement("tr"),tbody=document.createElement("tbody");wrap.className="markdown-table-wrap";headers.forEach((value,column)=>{const cell=document.createElement("th");cell.style.textAlign=alignments[column]||"left";appendInline(cell,value);headRow.append(cell)});thead.append(headRow);table.append(thead,tbody);index+=2;while(index<lines.length&&lines[index].trim()&&lines[index].includes("|")){const row=document.createElement("tr");tableCells(lines[index++]).forEach((value,column)=>{const cell=document.createElement("td");cell.style.textAlign=alignments[column]||"left";appendInline(cell,value);row.append(cell)});tbody.append(row)}wrap.append(table);root.append(wrap);continue}const paragraph=[];while(index<lines.length&&lines[index].trim()&&!isBlockStart(lines,index)&&!(index+1<lines.length&&/^\s*(?:=+|-+)\s*$/.test(lines[index+1]))){paragraph.push(lines[index++])}if(!paragraph.length){paragraph.push(lines[index++])}const element=document.createElement("p");appendInline(element,paragraph.join(" "));root.append(element)}}
function makeMsgAction(icon,label,fn){const b=document.createElement("button");b.type="button";b.className="msg-action";b.textContent=icon;b.title=label;b.setAttribute("aria-label",label);b.onclick=fn;return b}
function addMessage(role,text,ts){welcome.hidden=true;const row=document.createElement("div");row.className="message-row "+role;if(ts){try{row.title=new Date(ts).toLocaleString()}catch(e){}}const content=document.createElement("div");content.className="message-content";if(role==="assistant"){const markdown=document.createElement("div");markdown.className="markdown-body";renderMarkdown(markdown,text);content.append(markdown);const activity=document.createElement("div");activity.className="assistant-activity";activity.hidden=true;activity.setAttribute("role","status");activity.setAttribute("aria-live","polite");const label=document.createElement("span");const dots=document.createElement("span");dots.className="activity-dots";dots.setAttribute("aria-hidden","true");for(let i=0;i<3;i++)dots.append(document.createElement("span"));activity.append(label,dots);content.append(activity);content._markdown=markdown;content._activity=activity;content._activityLabel=label}else content.textContent=text;row.append(content);const actions=document.createElement("div");actions.className="message-actions";const msgText=()=>role==="assistant"?(content._sourceText||""):text;actions.append(makeMsgAction("\u29c9","Copy message",e=>copyText(e.currentTarget,msgText())));if(role==="user"){actions.append(makeMsgAction("\u21bb","Redo this message",()=>{if(activeJobs.has(activeSessionId))return;submitChatText(text)}));actions.append(makeMsgAction("\u270e","Edit message",()=>editUserMessage(row,text)))}else{actions.append(makeMsgAction("\u27f3","Regenerate response",()=>regenerateResponse(row)));const likeBtn=makeMsgAction("\uD83D\uDC4D","Good response",()=>setFeedback(row,likeBtn,disBtn,"up"));const disBtn=makeMsgAction("\uD83D\uDC4E","Bad response",()=>setFeedback(row,disBtn,likeBtn,"down"));actions.append(likeBtn,disBtn);applyStoredFeedback(row,text,likeBtn,disBtn)}row.append(actions);messages.append(row);scrollIfStuck();if(role==="assistant")content._sourceText=text;return content}
function feedbackKey(text){return "clawd-feedback:"+(activeSessionId||"none")+":"+hashStr(text||"")}
function feedbackTextOf(row){const c=row.querySelector(".message-content");return (c&&c._sourceText)||""}
function setFeedback(row,btn,other,kind){const key=feedbackKey(feedbackTextOf(row));let cur=null;try{cur=localStorage.getItem(key)}catch(e){}const next=cur===kind?"":kind;try{if(next)localStorage.setItem(key,next);else localStorage.removeItem(key)}catch(e){}btn.classList.toggle("active",next===kind);other.classList.remove("active")}
function applyStoredFeedback(row,text,likeBtn,disBtn){let cur=null;try{cur=localStorage.getItem(feedbackKey(text))}catch(e){}likeBtn.classList.toggle("active",cur==="up");disBtn.classList.toggle("active",cur==="down")}
function removeRowsFrom(row){activeToolGroup=null;toolCards.clear();let n=row;while(n){const next=n.nextSibling;if(n.classList&&(n.classList.contains("message-row")||n.classList.contains("tool-group"))||n.id==="interaction")n.remove();n=next}stickToBottom=true;updateScrollButton()}
async function regenerateResponse(row){if(!row||activeJobs.has(activeSessionId))return;const userRows=[...messages.querySelectorAll(".message-row.user .message-content")];const lastUser=userRows[userRows.length-1];if(!lastUser)return;const text=lastUser.textContent;notice.textContent="";try{await api("/api/chat/retry",{method:"POST",body:JSON.stringify({session_id:activeSessionId,mode:"regenerate"})})}catch(e){notice.textContent=e.message;return}removeRowsFrom(row);submitChatText(text)}
function editUserMessage(row,text){if(activeJobs.has(activeSessionId))return;const content=row.querySelector(".message-content");content.replaceChildren();const ta=document.createElement("textarea");ta.className="message-edit";ta.value=text;ta.setAttribute("aria-label","Edit message");const btnRow=document.createElement("div");btnRow.className="message-edit-actions";const cancel=document.createElement("button");cancel.type="button";cancel.className="secondary-button";cancel.textContent="Cancel";const save=document.createElement("button");save.type="button";save.className="primary-button";save.textContent="Send";const restore=()=>{content.textContent=text};cancel.onclick=restore;const doSave=async()=>{const newText=ta.value.trim();if(!newText||newText===text){restore();return}try{await api("/api/chat/retry",{method:"POST",body:JSON.stringify({session_id:activeSessionId,mode:"edit",message:text,new_message:newText})})}catch(e){notice.textContent=e.message;restore();return}removeRowsFrom(row);submitChatText(newText)};save.onclick=doSave;ta.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();doSave()}else if(e.key==="Escape")restore()});btnRow.append(cancel,save);content.append(ta,btnRow);ta.focus();ta.setSelectionRange(ta.value.length,ta.value.length)}
async function exportChat(){if(!activeSessionId)return;try{const saved=await api("/api/session/load",{method:"POST",body:JSON.stringify({session_id:activeSessionId})});let md="# "+(saved.title||"Chat")+"\n\n";for(const m of saved.messages||[]){if(!m.text)continue;md+=(m.role==="user"?"## You\n\n":"## Clawd\n\n")+m.text+"\n\n"}const blob=new Blob([md],{type:"text/markdown"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=(saved.title||"chat").replace(/[^\w\-]+/g,"_").slice(0,60)+".md";document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),4000);notice.textContent="Chat exported."}catch(e){notice.textContent=e.message}}
document.querySelector("#exportChat").onclick=exportChat;
function setAssistantText(content,text){content._sourceText=text;renderMarkdown(content._markdown,text)}
function setAssistantActivity(content,activity){const label=content._activityLabel;content._activity.hidden=!["thinking","typing"].includes(activity);label.textContent=activity==="typing"?"Typing":"Thinking"}
function setThoughtFor(content,elapsedMs){content._activity.hidden=true;let el=content._thought;if(!el){el=document.createElement("div");el.className="assistant-thought";content.append(el);content._thought=el}el.hidden=false;const t=formatElapsed(elapsedMs);el.textContent=t?"Thought for "+t:""}
function clearAllActivities(){document.querySelectorAll(".assistant-activity").forEach(el=>{el.hidden=true})}
function selectSession(sessionId){activeSessionId=sessionId;for(const item of document.querySelectorAll(".session-item")){const active=item.dataset.sessionId===sessionId;item.classList.toggle("active",active);if(active)item.setAttribute("aria-current","page");else item.removeAttribute("aria-current")}setComposerBusy()}
async function refreshSessions(){try{const result=await api("/api/sessions");lastSessions=result.sessions||[]}catch(e){return}renderSessionList()}
function renderSessionList(){const list=document.querySelector("#sessionsList");const q=(sessionSearch.value||"").trim().toLowerCase();const sessions=q?lastSessions.filter(s=>(s.title||"").toLowerCase().includes(q)):lastSessions;list.replaceChildren();if(!sessions.length){const empty=document.createElement("div");empty.className="sessions-empty";empty.textContent=q?"No chats match":"No saved chats";list.append(empty);return}for(const session of sessions){const row=document.createElement("div");row.className="session-item";row.dataset.sessionId=session.session_id;const open=document.createElement("button");open.type="button";open.className="session-open";open.title=session.title;const title=document.createElement("span");title.className="session-title";title.textContent=session.title;open.append(title);open.onclick=()=>openSavedSession(session.session_id);row.append(open);const renameBtn=document.createElement("button");renameBtn.type="button";renameBtn.className="session-rename";renameBtn.textContent="\u270e";renameBtn.title="Rename chat";renameBtn.setAttribute("aria-label","Rename chat: "+session.title);renameBtn.onclick=e=>{e.stopPropagation();startSessionRename(row,session)};row.append(renameBtn);if(typeof session.session_cost_usd==="number"&&session.session_cost_usd>0){const scost=document.createElement("span");scost.className="session-cost";scost.textContent="$"+session.session_cost_usd.toFixed(4);scost.title="Session spend";row.append(scost)}if(session.running||activeJobs.has(session.session_id)){const status=document.createElement("span");status.className="session-state";status.textContent="Running";status.title="Chat is running";row.append(status)}const remove=document.createElement("button");remove.type="button";remove.className="session-delete";remove.textContent="×";remove.title="Delete saved chat";remove.setAttribute("aria-label","Delete saved chat: "+session.title);remove.disabled=session.running||activeJobs.has(session.session_id);remove.onclick=()=>deleteSavedSession(session.session_id);row.append(remove);if(session.session_id===activeSessionId)row.classList.add("active");list.append(row)}}
function resetDisplayedConversation(){messages.replaceChildren();toolCards.clear();activeToolGroup=null;welcome.hidden=true;document.querySelector("#interaction")?.remove()}
function restoreRunningJob(sessionId){const running=activeJobs.get(sessionId);if(!running)return;running.renderedText="";running.renderedEvents=0;running.assistant=null;running.interactionId=null;running.assistant=addMessage("assistant","");if(running.latestText){setAssistantText(running.assistant,running.latestText);running.renderedText=running.latestText}running.assistantText=running.latestText||"";setAssistantActivity(running.assistant,running.latestActivity);const pending=running.pendingDeltas||[];running.pendingDeltas=[];for(const d of pending){if(d.kind==="text"||d.kind==="tool")running.renderedTimeline++;if(d.kind!=="text")renderDelta(running,sessionId,d)}}
async function openSavedSession(sessionId){try{const saved=await api("/api/session/load",{method:"POST",body:JSON.stringify({session_id:sessionId})});updateSessionWorkspace(saved.workspace);resetDisplayedConversation();for(const message of saved.messages){if(message.text)addMessage(message.role,message.text,message.ts)}document.querySelector("#sessionTitle").textContent=saved.title;promptBox.value="";selectSession(saved.session_id);restoreRunningJob(saved.session_id);notice.textContent=activeJobs.has(saved.session_id)?"This chat is running in the background. You can keep working in another chat.":"";closeSidebar()}catch(e){notice.textContent=e.message}}
async function deleteSavedSession(sessionId){if(activeJobs.has(sessionId)){notice.textContent="This chat is still running and cannot be deleted yet.";return}if(!confirm("Delete this saved chat? This cannot be undone."))return;try{const result=await api("/api/session/delete",{method:"POST",body:JSON.stringify({session_id:sessionId})});if(activeSessionId===sessionId){resetConversation();selectSession(result.session_id)}await refreshSessions()}catch(e){notice.textContent=e.message}}
async function startSessionRename(row,session){const open=row.querySelector(".session-open");if(!open||open.querySelector("input"))return;const input=document.createElement("input");input.type="text";input.className="session-rename-input";input.value=session.title;input.maxLength=120;input.setAttribute("aria-label","Chat name");open.replaceChildren(input);input.focus();input.select();let done=false;const commit=async save=>{if(done)return;done=true;const value=input.value.trim();if(save&&value&&value!==session.title){try{await api("/api/session/rename",{method:"POST",body:JSON.stringify({session_id:session.session_id,title:value})})}catch(e){notice.textContent=e.message}}await refreshSessions()};input.addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();commit(true)}else if(e.key==="Escape")commit(false)});input.addEventListener("blur",()=>commit(true))}
function toolDisplayName(name){const key=(name||"Tool").toLowerCase();return ({bash:"Shell",powershell:"Shell",websearch:"Search",webfetch:"Fetch",askuserquestion:"Ask"})[key]||name||"Tool"}
function formatElapsed(ms){if(ms==null||isNaN(ms)||ms<0)return"";const s=ms/1000;if(s<60){const v=s<10?Math.round(s*10)/10:Math.round(s);return (""+v).replace(/\.0$/,"")+"s"}return Math.floor(s/60)+"m "+Math.round(s%60)+"s"}
function setToolDiffStat(el,stat){el.replaceChildren();if(!stat)return;const a=stat.added||0,d=stat.removed||0;if(!a&&!d)return;const sa=document.createElement("span");sa.className="add";sa.textContent="+"+a;const sd=document.createElement("span");sd.className="del";sd.textContent="-"+d;el.append(document.createTextNode("("),sa,document.createTextNode(", "),sd,document.createTextNode(")"))}
function setToolElapsed(el,ms){el.textContent=(ms==null||isNaN(ms))?"":" · "+formatElapsed(ms)}
/* Tiny inline syntax highlighter (no CDN). Regex tokenizer: comments, strings, numbers, keywords, function calls. */
const HL_KEYWORDS={python:["def","class","return","if","elif","else","for","while","import","from","as","with","try","except","finally","raise","lambda","pass","break","continue","in","is","not","and","or","None","True","False","async","await","yield","global","nonlocal","assert","del","print"],javascript:["function","return","if","else","for","while","const","let","var","import","from","export","default","new","class","extends","try","catch","finally","throw","async","await","typeof","instanceof","in","of","this","null","undefined","true","false","switch","case","break","continue","do"],typescript:["function","return","if","else","for","while","const","let","var","import","from","export","default","new","class","extends","implements","interface","type","enum","try","catch","finally","throw","async","await","typeof","instanceof","in","of","this","null","undefined","true","false","switch","case","break","continue","do","public","private","protected","readonly","static"],bash:["if","then","else","elif","fi","for","while","do","done","case","esac","function","return","exit","export","local","echo","in"],sh:["if","then","else","elif","fi","for","while","do","done","case","esac","function","return","exit","export","local","echo","in"],shell:["if","then","else","elif","fi","for","while","do","done","case","esac","function","return","exit","export","local","echo","in"],sql:["select","from","where","join","left","right","inner","outer","on","group","by","order","having","insert","into","values","update","set","delete","create","table","alter","drop","index","and","or","not","null","as","distinct","limit","offset","union","all"],css:["color","background","margin","padding","border","display","position","width","height","font","grid","flex"],json:[],html:["div","span","button","input","form","section","header","aside","main","script","style","html","head","body"],xml:[]};
const HL_ALIAS={js:"javascript",ts:"typescript",py:"python",yml:"yaml"};
function hlKeywords(lang){const key=HL_ALIAS[lang]||lang;return HL_KEYWORDS[key]}
function highlightCode(code){const rawLang=(code.dataset.language||"").toLowerCase();const keywords=hlKeywords(rawLang);const text=code.textContent;code.replaceChildren();const isHashComment=["python","bash","sh","shell","yaml","sql"].includes(HL_ALIAS[rawLang]||rawLang);const re=isHashComment?/(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g:/(\/\/[^\n]*|\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;let cursor=0,match;const kwSet=keywords?new Set(keywords):null;while((match=re.exec(text))){if(match.index>cursor)appendHlText(code,text.slice(cursor,match.index),kwSet);const span=document.createElement("span");if(match[1]){span.className="tok-com";span.textContent=match[0]}else if(match[2]){span.className="tok-str";span.textContent=match[0]}else if(match[3]){span.className="tok-num";span.textContent=match[0]}else{span.className="tok-fn";span.textContent=match[0]}code.append(span);cursor=match.index+match[0].length}if(cursor<text.length)appendHlText(code,text.slice(cursor),kwSet)}
function appendHlText(parent,text,kwSet){if(!kwSet||!kwSet.size){parent.append(document.createTextNode(text));return}const re=/\b([A-Za-z_]\w*)\b/g;let cursor=0,match;while((match=re.exec(text))){if(match.index>cursor)parent.append(document.createTextNode(text.slice(cursor,match.index)));if(kwSet.has(match[1])){const span=document.createElement("span");span.className="tok-kw";span.textContent=match[1];parent.append(span)}else parent.append(document.createTextNode(match[1]));cursor=match.index+match[1].length}if(cursor<text.length)parent.append(document.createTextNode(text.slice(cursor)))}
function addToolEvent(event){let card=toolCards.get(event.tool_use_id);if(!card){if(!activeToolGroup){activeToolGroup=document.createElement("details");activeToolGroup.className="tool-group";activeToolGroup.dataset.status="running";const summary=document.createElement("summary"),label=document.createElement("span"),count=document.createElement("span"),actions=document.createElement("span"),status=document.createElement("span"),content=document.createElement("div");label.className="tool-group-label";label.textContent="Used";count.className="tool-group-count";actions.className="tool-group-actions";status.className="tool-group-status";content.className="tool-group-content";summary.append(label,count,actions,status);activeToolGroup.append(summary,content);activeToolGroup._parts={count,actions,status,content,names:[],total:0,pending:0};messages.append(activeToolGroup)}card=document.createElement("details");card.className="tool-card";const summary=document.createElement("summary"),slim=document.createElement("span"),verb=document.createElement("span"),rest=document.createElement("span"),meta=document.createElement("span"),diffstat=document.createElement("span"),elapsed=document.createElement("span"),status=document.createElement("span"),detail=document.createElement("div"),inputSection=document.createElement("section"),inputLabel=document.createElement("div"),inputValue=document.createElement("pre"),outputSection=document.createElement("section"),outputLabel=document.createElement("div"),outputValue=document.createElement("pre");slim.className="tool-slim";verb.className="tool-verb";meta.className="tool-meta";diffstat.className="tool-diffstat";elapsed.className="tool-elapsed";status.className="tool-status";detail.className="tool-detail";inputLabel.className="tool-detail-label";inputLabel.textContent="Input";outputLabel.className="tool-detail-label";outputLabel.textContent="Output";inputSection.append(inputLabel,inputValue);outputSection.append(outputLabel,outputValue);detail.append(inputSection);if(event.diff){const diffSection=document.createElement("section"),diffLabel=document.createElement("div"),diffPre=document.createElement("pre");diffLabel.className="tool-detail-label";diffLabel.textContent="Diff";diffPre.className="diff-block";for(const line of String(event.diff).split("\n")){const diffRow=document.createElement("div");diffRow.textContent=line||" ";diffRow.className=line.startsWith("+")&&!line.startsWith("+++")?"diff-add":line.startsWith("-")&&!line.startsWith("---")?"diff-del":line.startsWith("@@")?"diff-hunk":"";diffPre.append(diffRow)}diffSection.append(diffLabel,diffPre);detail.append(diffSection)}detail.append(outputSection);slim.append(verb,rest);meta.append(diffstat,elapsed,status);summary.append(slim,meta);card.append(summary,detail);activeToolGroup._parts.content.append(card);activeToolGroup._parts.pending++;activeToolGroup._parts.total++;const action=toolDisplayName(event.tool_name),groupParts=activeToolGroup._parts;let entry=groupParts.names.find(n=>n.label===action);if(!entry){entry={label:action,category:(event.tool_name||"").toLowerCase(),failed:false};entry.category=["bash","powershell"].includes(entry.category)?"tool-shell":["webfetch","websearch"].includes(entry.category)?"tool-web":["read","write","edit","glob","grep"].includes(entry.category)?"tool-file":"tool-default";groupParts.names.push(entry)}toolCards.set(event.tool_use_id,card);card._parts={slim,verb,rest,diffstat,elapsed,status,inputValue,outputValue,group:activeToolGroup};card._actionEntry=entry;const normalized=(event.tool_name||"").toLowerCase();card.classList.add(["bash","powershell"].includes(normalized)?"tool-shell":["webfetch","websearch"].includes(normalized)?"tool-web":["read","write","edit","glob","grep"].includes(normalized)?"tool-file":"tool-default");updateToolGroup(activeToolGroup)}
const parts=card._parts,toolName=event.tool_name||"Tool";parts.verb.textContent=toolDisplayName(toolName);parts.rest.textContent=event.summary?" "+event.summary:"";parts.status.textContent=event.kind==="tool_use"?"Running":event.error?"Failed":"Done";setToolDiffStat(parts.diffstat,event.diff_stat);setToolElapsed(parts.elapsed,event.elapsed_ms);if(event.input)parts.inputValue.textContent=event.input;if(event.output)parts.outputValue.textContent=event.output;if(event.kind!=="tool_use"&&card.dataset.completed!=="true"){card.dataset.completed="true";parts.group._parts.pending=Math.max(0,parts.group._parts.pending-1);updateToolGroup(parts.group)}if(event.error){card.classList.add("tool-error");if(card._actionEntry){card._actionEntry.failed=true;updateToolGroup(card._parts.group)}}scrollIfStuck()}
function updateToolGroup(group){const parts=group._parts;parts.count.textContent=" "+parts.total;parts.actions.replaceChildren();parts.names.forEach((entry,index)=>{if(index)parts.actions.append(", ");const chip=document.createElement("span");chip.className="tool-group-action "+entry.category+(entry.failed?" tool-error":"");chip.textContent=entry.label;parts.actions.append(chip)});parts.status.textContent=parts.pending?"Working":"";group.dataset.status=parts.pending?"running":"completed"}
function renderTimelineItem(job,item){if(item.kind==="text"){if(!job.assistant||job.toolGroupOpen){if(job.assistant)setAssistantActivity(job.assistant,"");job.assistant=addMessage("assistant","");job.assistantText="";job.toolGroupOpen=false}job.assistantText+=item.text;setAssistantText(job.assistant,job.assistantText);job.renderedText=job.latestText;return}if(item.kind==="tool"){if(item.event.kind==="tool_use"&&!job.toolGroupOpen)job.toolGroupOpen=true;addToolEvent(item.event)}}
function resetConversation(){messages.replaceChildren();toolCards.clear();activeToolGroup=null;welcome.hidden=false;notice.textContent="";document.querySelector("#sessionTitle").textContent="New session";promptBox.value="";selectSession(null);promptBox.focus()}
function closeSidebar(){sidebar.classList.remove("open");document.querySelector("#mobileBackdrop").classList.remove("open")}
async function connect(){try{const s=await api("/api/state");updateWorkspace(s.workspace);document.querySelector("#composerModel").textContent=s.provider+" · "+s.model;selectSession(s.session_id);await refreshSessions()}catch(e){notice.textContent=e.message}}
connect();
function showFolderPickerMessage(text){folderList.replaceChildren();const message=document.createElement("div");message.className="hint";message.textContent=text;folderList.append(message)}
function makeFolderRow(name,path,description,icon,onClick){const button=document.createElement("button");button.type="button";button.className="folder-entry";button.title=path;const glyph=document.createElement("span"),copy=document.createElement("span"),label=document.createElement("span"),detail=document.createElement("span");glyph.className="folder-entry-icon";glyph.textContent=icon;copy.className="folder-entry-copy";label.className="folder-entry-name";label.textContent=name;detail.className="folder-entry-path";detail.textContent=description;copy.append(label,detail);button.append(glyph,copy);button.onclick=onClick;return button}
function selectFolder(path){selectedFolderPath=path;useFolderButton.disabled=!path;for(const row of folderList.querySelectorAll(".folder-entry")){const selected=row.dataset.path===path;row.classList.toggle("selected",selected);row.setAttribute("aria-pressed",String(selected))}}
async function browseDirectories(path){const requestId=++directoryRequestId;try{const query=new URLSearchParams({path});const result=await api("/api/directories?"+query);if(requestId!==directoryRequestId)return;currentDirectory=result.path;folderPath.value=currentDirectory;folderPickerHint.textContent="Choose this folder or browse into a subfolder. Search is limited to this location.";folderList.replaceChildren();folderList.scrollTop=0;if(result.parent){const row=makeFolderRow("Parent folder",result.parent,result.parent,"↰",()=>{folderSearch.value="";browseDirectories(result.parent)});row.dataset.path=result.parent;folderList.append(row)}for(const root of result.roots||[]){const row=makeFolderRow(root,root,root,"▣",()=>{folderSearch.value="";browseDirectories(root)});row.dataset.path=root;folderList.append(row)}for(const directory of result.directories){const row=makeFolderRow(directory.name,directory.path,directory.path,"▰",()=>{folderSearch.value="";browseDirectories(directory.path)});row.dataset.path=directory.path;folderList.append(row)}if(!result.directories.length&&!result.roots?.length&&!result.parent)showFolderPickerMessage("No subfolders in this location.");selectFolder(result.path);}catch(e){if(requestId===directoryRequestId){folderPickerHint.textContent=e.message;showFolderPickerMessage(e.message)}}}
async function searchDirectories(){const queryText=folderSearch.value.trim(),requestId=++directoryRequestId;if(!queryText){if(currentDirectory)browseDirectories(currentDirectory);return}if(queryText.length<2){selectFolder(null);folderPickerHint.textContent="Type at least 2 characters to search this folder.";showFolderPickerMessage("Keep typing to search.");return}if(!currentDirectory){folderPickerHint.textContent="Choose a starting folder first.";showFolderPickerMessage("Enter a folder path and select Go.");return}selectFolder(null);folderPickerHint.textContent="Searching within "+currentDirectory+"…";showFolderPickerMessage("Searching this folder and its subfolders…");try{const query=new URLSearchParams({path:currentDirectory,query:queryText});const result=await api("/api/directories/search?"+query);if(requestId!==directoryRequestId)return;folderList.replaceChildren();folderList.scrollTop=0;for(const directory of result.directories){const button=makeFolderRow(directory.name,directory.path,directory.parent,"▰",()=>selectFolder(directory.path));button.dataset.path=directory.path;folderList.append(button)}if(!result.directories.length)showFolderPickerMessage("No matching folders found here.");let summary=`${result.directories.length} matching folder${result.directories.length===1?"":"s"} in ${currentDirectory}`;if(result.skipped)summary+=`; ${result.skipped} inaccessible location${result.skipped===1?" was":"s were"} skipped`;if(result.truncated)summary+="; search stopped at the folder limit";folderPickerHint.textContent=summary+".";}catch(e){if(requestId===directoryRequestId){showFolderPickerMessage(e.message);folderPickerHint.textContent="Folder search failed."}}}
function scheduleDirectorySearch(){directoryRequestId++;if(directorySearchTimer)clearTimeout(directorySearchTimer);directorySearchTimer=setTimeout(searchDirectories,350)}
function openFolderPicker(){folderPicker.hidden=false;folderSearch.value="";selectedFolderPath=null;currentDirectory=null;useFolderButton.disabled=true;folderPath.value=document.querySelector("#sidebarPath").textContent;browseDirectories(folderPath.value)}
document.querySelector("#chooseWorkspace").onclick=openFolderPicker;
document.querySelector("#changeFolder").onclick=()=>{closeSidebar();openFolderPicker()};
document.querySelector("#goFolderPath").onclick=()=>{folderSearch.value="";selectedFolderPath=null;useFolderButton.disabled=true;browseDirectories(folderPath.value.trim())};
folderPath.addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();document.querySelector("#goFolderPath").click()}});
folderSearch.addEventListener("input",()=>{selectedFolderPath=null;useFolderButton.disabled=true;scheduleDirectorySearch()});
document.querySelector("#useFolder").onclick=async()=>{try{const result=await api("/api/workspace",{method:"POST",body:JSON.stringify({path:selectedFolderPath||currentDirectory||document.querySelector("#sidebarPath").textContent})});updateWorkspace(result.workspace);folderPicker.hidden=true;resetConversation();selectSession(result.session_id);closeSidebar();await refreshSessions()}catch(e){notice.textContent=e.message}};
document.querySelector("#closeFolder").onclick=()=>folderPicker.hidden=true;
document.querySelector("#cancelFolder").onclick=()=>folderPicker.hidden=true;
folderPicker.addEventListener("click",e=>{if(e.target===folderPicker)folderPicker.hidden=true});
const settingsPicker=document.querySelector("#settingsPicker"),settingsProvider=document.querySelector("#settingsProvider"),settingsModel=document.querySelector("#settingsModel"),settingsApiKey=document.querySelector("#settingsApiKey"),settingsKeyNote=document.querySelector("#settingsKeyNote"),themePicker=document.querySelector("#themePicker");
let settingsData=null;
function readStoredTheme(){try{return localStorage.getItem("clawd-theme")||"dark"}catch(e){return"dark"}}
function applyTheme(choice){const resolved=choice==="system"?(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"):choice;document.documentElement.dataset.theme=resolved;for(const button of themePicker.querySelectorAll("button"))button.classList.toggle("active",button.dataset.themeValue===choice)}
applyTheme(readStoredTheme());
matchMedia("(prefers-color-scheme: light)").addEventListener("change",()=>{if(readStoredTheme()==="system")applyTheme("system")});
for(const button of themePicker.querySelectorAll("button"))button.onclick=()=>{try{localStorage.setItem("clawd-theme",button.dataset.themeValue)}catch(e){}applyTheme(button.dataset.themeValue)};
function fillModelOptions(){const info=settingsData.providers.find(p=>p.name===settingsProvider.value);const models=info?info.models:[];const current=info?info.default_model:"";settingsModel.replaceChildren();for(const model of models){const option=document.createElement("option");option.value=model;option.textContent=model;settingsModel.append(option)}if(current&&models.includes(current))settingsModel.value=current}
function updateKeyNote(){const info=settingsData.providers.find(p=>p.name===settingsProvider.value);settingsKeyNote.textContent=info&&info.configured?"API key saved for "+info.label+". Paste a new one to replace it.":"No API key saved for "+(info?info.label:settingsProvider.value)+". Add one here or run `clawd login`."}
settingsProvider.onchange=()=>{fillModelOptions();updateKeyNote()};
async function openSettings(){try{settingsData=await api("/api/settings");settingsProvider.replaceChildren();for(const info of settingsData.providers){const option=document.createElement("option");option.value=info.name;option.textContent=info.label+(info.configured?"":" (no API key)");settingsProvider.append(option)}settingsProvider.value=settingsData.provider;fillModelOptions();if(settingsData.model&&!settingsModel.querySelector('option[value="'+CSS.escape(settingsData.model)+'"]')){const option=document.createElement("option");option.value=settingsData.model;option.textContent=settingsData.model;settingsModel.append(option)}if(settingsData.model)settingsModel.value=settingsData.model;updateKeyNote();settingsApiKey.value="";applyTheme(readStoredTheme());settingsPicker.hidden=false}catch(e){notice.textContent=e.message}}
document.querySelector("#openSettings").onclick=openSettings;
document.querySelector("#closeSettings").onclick=()=>settingsPicker.hidden=true;
document.querySelector("#cancelSettings").onclick=()=>settingsPicker.hidden=true;
settingsPicker.addEventListener("click",e=>{if(e.target===settingsPicker)settingsPicker.hidden=true});
document.querySelector("#saveSettings").onclick=async()=>{try{const body={provider:settingsProvider.value,model:settingsModel.value};const key=settingsApiKey.value.trim();if(key)body.api_key=key;const result=await api("/api/settings",{method:"POST",body:JSON.stringify(body)});const label=result.provider+" · "+result.model;document.querySelector("#composerModel").textContent=label;settingsPicker.hidden=true;notice.textContent="Settings saved · "+label}catch(e){notice.textContent=e.message}};
document.querySelector("#mobileNav").onclick=()=>{sidebar.classList.add("open");document.querySelector("#mobileBackdrop").classList.add("open")};
document.querySelector("#sidebarClose").onclick=closeSidebar;
document.querySelector("#mobileBackdrop").onclick=closeSidebar;
document.querySelector("#newSession").onclick=async()=>{try{const result=await api("/api/session/new",{method:"POST",body:JSON.stringify({start:true})});resetConversation();selectSession(result.session_id);closeSidebar();await refreshSessions()}catch(e){notice.textContent=e.message}};
sessionSearch.addEventListener("input",renderSessionList);
document.querySelectorAll("[data-prompt]").forEach(button=>button.addEventListener("click",()=>{promptBox.value=button.dataset.prompt||"";promptBox.focus()}));
async function answerInteraction(job,interaction,answer){try{await api("/api/jobs/"+job.jobId+"/interaction",{method:"POST",body:JSON.stringify({interaction_id:interaction.id,...answer})});if(activeSessionId===job.sessionId)document.querySelector("#interaction")?.remove();job.interactionId=null}catch(e){if(activeSessionId===job.sessionId)notice.textContent=e.message}}
function showInteraction(x,job){document.querySelector("#interaction")?.remove();const box=document.createElement("div");box.id="interaction";box.className="interaction";
if(x.kind==="permission"){const heading=document.createElement("strong");heading.textContent="Permission needed \u00b7 "+toolDisplayName(x.tool_name);box.append(heading);const message=document.createElement("p");message.textContent=x.message;box.append(message);if(x.suggestion){const suggestion=document.createElement("p");suggestion.className="hint";suggestion.textContent="Suggestion: "+x.suggestion;box.append(suggestion)}const note=document.createElement("p");note.className="hint";note.textContent="Review the tool card above to see exactly what will run.";box.append(note);for(const [label,allowed] of [["Allow",true],["Deny",false]]){const b=document.createElement("button");b.textContent=label;b.onclick=()=>answerInteraction(job,x,{allowed});box.append(b)}}
else{const title=document.createElement("strong");title.textContent="Clawd has a question";box.append(title);const answers={};for(const q of x.questions||[]){const p=document.createElement("p");p.textContent=q.question;box.append(p);const selected=new Set();for(const o of q.options||[]){const b=document.createElement("button");b.textContent=o.label;b.onclick=()=>{if(q.multiSelect){if(selected.has(o.label)){selected.delete(o.label);b.style.outline=""}else{selected.add(o.label);b.style.outline="2px solid white"}if(selected.size)answers[q.question]=Array.from(selected).join(", ");else delete answers[q.question]}else{answers[q.question]=o.label;for(const option of q.options||[]){if(option.label===o.label)b.style.outline="2px solid white"}for(const sibling of Array.from(box.querySelectorAll("button"))){if(sibling!==b&&q.options.some(option=>option.label===sibling.textContent))sibling.style.outline=""}}};box.append(b)}}
const submit=document.createElement("button");submit.textContent="Submit answers";submit.onclick=()=>answerInteraction(job,x,{answers});box.append(submit)}
messages.append(box);box.scrollIntoView({behavior:"smooth",block:"end"})}
function setComposerBusy(){const busy=activeJobs.has(activeSessionId);sendButton.disabled=busy;sendButton.hidden=busy;stopButton.hidden=!busy;stopButton.disabled=false;stopButton.classList.remove("stopping");if(!promptBox.value)promptBox.style.height="auto"}
stopButton.onclick=()=>{const job=activeJobs.get(activeSessionId);if(!job||!job.jobId)return;stopButton.disabled=true;stopButton.classList.add("stopping");notice.textContent="Stopping…";job.stopping=true;api("/api/jobs/"+job.jobId+"/cancel",{method:"POST",body:"{}"}).then(()=>{if(job.stopping&&activeJobs.get(job.sessionId)===job)notice.textContent="Stopping…" }).catch(e=>{stopButton.disabled=false;stopButton.classList.remove("stopping");job.stopping=false;if(activeSessionId===job.sessionId)notice.textContent=e.message})}
function formatTokens(usage){if(!usage)return"";const total=(usage.input_tokens||0)+(usage.output_tokens||0);if(!total)return"";return total>=1000?(total/1000).toFixed(1)+"k tokens":total+" tokens"}
function formatCost(usage){if(!usage||typeof usage.cost_usd!=="number")return"";return"$"+usage.cost_usd.toFixed(4)}
function applyJobState(job,sessionId,j){job.latestText=j.text;job.latestActivity=j.activity;if(typeof j.seq==="number")job.seq=j.seq;if(activeSessionId===sessionId){if(Array.isArray(j.timeline)){while(job.renderedTimeline<j.timeline.length)renderTimelineItem(job,j.timeline[job.renderedTimeline++]);if(!job.assistant&&j.activity==="thinking")job.assistant=addMessage("assistant","");if(job.assistant)setAssistantActivity(job.assistant,j.activity)}else{if(!job.assistant)job.assistant=addMessage("assistant","");if(job.renderedText!==j.text){setAssistantText(job.assistant,j.text);job.renderedText=j.text}setAssistantActivity(job.assistant,j.activity);while(job.renderedEvents<j.events.length){addToolEvent(j.events[job.renderedEvents++])}}if(j.interaction){if(job.interactionId!==j.interaction.id){showInteraction(j.interaction,job);job.interactionId=j.interaction.id}}else if(job.interactionId){document.querySelector("#interaction")?.remove();job.interactionId=null}}
if(j.status==="completed"||j.status==="error"||j.status==="stopped"){finishJob(job,sessionId,j);return true}return false}
function finishJob(job,sessionId,j){job.stopping=false;activeJobs.delete(sessionId);if(activeSessionId===sessionId){clearAllActivities();if(job.assistant&&j.elapsed_ms!=null)setThoughtFor(job.assistant,j.elapsed_ms);document.querySelector("#interaction")?.remove();job.interactionId=null;const tokens=formatTokens(j.usage),cost=formatCost(j.usage),spent=cost?" \u00b7 "+cost:"";if(j.status==="error")notice.textContent=j.error||"Request failed";else if(j.status==="stopped")notice.textContent="Stopped."+(tokens?" \u00b7 "+tokens:"")+spent;else notice.textContent=tokens+spent;setComposerBusy()}refreshSessions().catch(()=>{})}
function failJob(job,sessionId,e){activeJobs.delete(sessionId);if(activeSessionId===sessionId){clearAllActivities();notice.textContent=e.message;setComposerBusy()}refreshSessions().catch(()=>{})}
function renderDelta(job,sessionId,d){if(job.stopping&&(d.kind==="text"||d.kind==="tool"))return false;if(d.kind==="text"){renderTimelineItem(job,{kind:"text",text:d.text});return false}if(d.kind==="tool"){renderTimelineItem(job,{kind:"tool",event:d.event});return false}if(d.kind==="activity"){if(!job.assistant&&d.activity==="thinking")job.assistant=addMessage("assistant","");if(job.assistant)setAssistantActivity(job.assistant,d.activity);return false}if(d.kind==="interaction"){const x=d.interaction;if(x){if(job.interactionId!==x.id){showInteraction(x,job);job.interactionId=x.id}}else if(job.interactionId){document.querySelector("#interaction")?.remove();job.interactionId=null}return false}if(d.kind==="status"){finishJob(job,sessionId,{status:d.status,error:d.error,usage:d.usage,elapsed_ms:d.elapsed_ms});return true}return false}
function applyJobDelta(job,sessionId,d){job.seq=d.seq||job.seq;if(d.kind==="text")job.latestText=(job.latestText||"")+d.text;if(d.kind==="activity")job.latestActivity=d.activity;if(activeSessionId!==sessionId){if(d.kind==="text"||d.kind==="tool"||d.kind==="activity")(job.pendingDeltas=job.pendingDeltas||[]).push(d);return false}if(d.kind==="text"||d.kind==="tool")job.renderedTimeline++;return renderDelta(job,sessionId,d)}
function streamJob(job,sessionId){let source=null,retries=0;const connect=async()=>{if(activeJobs.get(sessionId)!==job)return;try{if(!job.seq){const snap=await api("/api/jobs/"+job.jobId);if(activeJobs.get(sessionId)!==job)return;if(applyJobState(job,sessionId,snap))return}source=new EventSource("/api/jobs/"+job.jobId+"/stream?last_seq="+job.seq);source.onmessage=e=>{if(activeJobs.get(sessionId)!==job){source.close();return}try{const d=JSON.parse(e.data);if(applyJobDelta(job,sessionId,d))source.close()}catch(err){}};source.onerror=()=>{source.close();source=null;if(activeJobs.get(sessionId)!==job)return;if(++retries>2)pollFallback(job,sessionId);else setTimeout(connect,800)}}catch(e){failJob(job,sessionId,e)}};connect()}
async function pollFallback(job,sessionId){while(activeJobs.get(sessionId)===job){try{const j=await api("/api/jobs/"+job.jobId);if(applyJobState(job,sessionId,j))break}catch(e){failJob(job,sessionId,e);break}await new Promise(r=>setTimeout(r,700))}}
async function poll(job,sessionId){streamJob(job,sessionId)}
async function submitChatText(text){const sessionId=activeSessionId;if(!text||!sessionId||activeJobs.has(sessionId))return;promptBox.value="";notice.textContent="";stickToBottom=true;addMessage("user",text);document.querySelector("#sessionTitle").textContent=text.length>48?text.slice(0,45)+"…":text;const job={jobId:null,sessionId,latestText:"",latestActivity:"thinking",renderedText:"",renderedEvents:0,renderedTimeline:0,seq:0,assistant:addMessage("assistant",""),assistantText:"",toolGroupOpen:false,interactionId:null};activeToolGroup=null;setAssistantActivity(job.assistant,job.latestActivity);activeJobs.set(sessionId,job);setComposerBusy();try{const started=await api("/api/chat",{method:"POST",body:JSON.stringify({message:text,session_id:sessionId})});job.jobId=started.job_id;try{await refreshSessions()}catch(refreshError){notice.textContent=refreshError.message}await poll(job,sessionId)}catch(e){activeJobs.delete(sessionId);clearAllActivities();if(activeSessionId===sessionId)notice.textContent=e.message;setComposerBusy();try{await refreshSessions()}catch(refreshError){notice.textContent=refreshError.message}}}
document.querySelector("#composer").addEventListener("submit",e=>{e.preventDefault();submitChatText(promptBox.value.trim())});
promptBox.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();document.querySelector("#composer").requestSubmit()}});
promptBox.addEventListener("input",()=>{promptBox.style.height="auto";promptBox.style.height=Math.min(promptBox.scrollHeight,180)+"px"});
const modelPicker=document.querySelector("#modelPicker"),modelPickerList=document.querySelector("#modelPickerList");
function closeModelPicker(){modelPicker.hidden=true}
async function openModelPicker(){try{const data=await api("/api/settings");modelPickerList.replaceChildren();const configured=data.providers.filter(p=>p.configured);if(!configured.length){const empty=document.createElement("div");empty.className="model-picker-empty";empty.textContent="No API keys saved yet. Add one in Settings (⚙).";modelPickerList.append(empty)}for(const info of configured){if(configured.length>1){const group=document.createElement("div");group.className="model-picker-group";group.textContent=info.label;modelPickerList.append(group)}const models=[...info.models];const currentHere=info.name===data.provider;if(currentHere&&!models.includes(data.model))models.unshift(data.model);for(const model of models){const item=document.createElement("button");item.type="button";item.className="model-picker-item"+(currentHere&&model===data.model?" active":"");const tick=document.createElement("span");tick.className="tick";tick.textContent=currentHere&&model===data.model?"✓":"";const name=document.createElement("span");name.textContent=model;item.append(tick,name);item.onclick=()=>chooseModel(info.name,model);modelPickerList.append(item)}}modelPicker.hidden=false}catch(e){notice.textContent=e.message}}
async function chooseModel(provider,model){try{const result=await api("/api/settings",{method:"POST",body:JSON.stringify({provider,model})});const label=result.provider+" · "+result.model;document.querySelector("#composerModel").textContent=label;closeModelPicker();notice.textContent=""}catch(e){notice.textContent=e.message}}
document.querySelector("#closeModelPicker").onclick=closeModelPicker;
modelPicker.addEventListener("click",e=>{if(e.target===modelPicker)closeModelPicker()});
document.querySelector("#composerModel").onclick=()=>openModelPicker();
</script>
</body>
</html>"""


@dataclass
class _Interaction:
    payload: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


@dataclass
class _Job:
    job_id: str
    session_id: str = ""
    session: Session | None = None
    tool_context: ToolContext | None = None
    prompt: str = ""
    message_added: bool = False
    status: str = "running"
    activity: str = "thinking"
    text: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    interaction: _Interaction | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    seq: int = 0
    stream: list[dict[str, Any]] = field(default_factory=list)
    stream_cond: threading.Condition = field(default_factory=threading.Condition)
    elapsed_ms: int | None = None


@dataclass
class _AppState:
    workspace: Path
    provider_name: str
    provider: Any
    session: Session
    registry: Any
    tool_context: ToolContext
    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, Session] = field(default_factory=dict)
    jobs: dict[str, _Job] = field(default_factory=dict)
    session_costs: dict[str, float] = field(default_factory=dict)
    dev_reload_version: str | None = None
    default_workspace: Path | None = None

    def __post_init__(self) -> None:
        if self.default_workspace is None:
            self.default_workspace = self.workspace
        if self.session.workspace is None:
            self.session.workspace = str(self.workspace)
        self.sessions.setdefault(self.session.session_id, self.session)


_cost_pricer = None


def _cost_pricer_instance():
    """Shared CostTracker used for pricing math only (it never records)."""
    global _cost_pricer
    if _cost_pricer is None:
        from src.cost_tracker import CostTracker

        _cost_pricer = CostTracker()
    return _cost_pricer


def _turn_cost_usd(state, usage):
    """Price one turn's usage dict in USD. Returns None on any problem."""
    try:
        if not usage:
            return None
        pricer = _cost_pricer_instance()
        return pricer.cost_for(
            provider=getattr(state, "provider_name", None),
            model=getattr(state.provider, "model", None),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cached_tokens=usage.get("cache_read_input_tokens", 0)
            or usage.get("cached_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        )
    except Exception:
        return None


def _create_state(workspace: str | Path) -> _AppState:
    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Workspace is not a directory: {root}")
    provider_name = get_default_provider()
    provider_config = get_provider_config(provider_name)
    if not provider_config.get("api_key"):
        raise ValueError("API key not configured. Run `clawd login` first.")
    provider = get_provider_class(provider_name)(
        api_key=provider_config["api_key"],
        base_url=provider_config.get("base_url"),
        model=provider_config.get("default_model"),
    )
    session = Session.create(provider_name, provider.model)
    session.workspace = str(root)
    tool_context = ToolContext(workspace_root=root)
    return _AppState(
        workspace=root,
        provider_name=provider_name,
        provider=provider,
        session=session,
        registry=build_default_registry(),
        tool_context=tool_context,
    )


def _filesystem_roots() -> list[Path]:
    if os.name != "nt":
        return [Path("/")]
    import ctypes

    drive_mask = ctypes.windll.kernel32.GetLogicalDrives()
    return [
        Path(f"{chr(ord('A') + index)}:\\")
        for index in range(26)
        if drive_mask & (1 << index)
    ]


def _drive_label(path: PurePath) -> str:
    """Display label for a path's volume anchor.

    Windows anchors look like ``C:\\`` and are shown as ``C:``; POSIX
    anchors are just ``/`` and must not be stripped down to an empty string.
    """
    return path.anchor.rstrip("\\/") or path.anchor


def _list_directories(workspace: Path, requested_path: str | None) -> dict[str, Any]:
    candidate = Path(requested_path).expanduser() if requested_path else workspace
    if not candidate.is_absolute():
        candidate = workspace / candidate
    path = candidate.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    directories = sorted(
        (
            {"name": entry.name, "path": str(entry.resolve())}
            for entry in path.iterdir()
            if entry.is_dir()
        ),
        key=lambda entry: entry["name"].casefold(),
    )
    parent = path.parent if path.parent != path else None
    roots = [str(root) for root in _filesystem_roots()] if parent is None else []
    return {
        "path": str(path),
        "parent": str(parent) if parent is not None else None,
        "roots": roots,
        "directories": directories,
    }


def _search_directories(
    workspace: Path,
    requested_path: str | None,
    query: str,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Search query must not be empty")
    root = Path(requested_path).expanduser() if requested_path else workspace
    if not root.is_absolute():
        root = workspace / root
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    needle = query.strip().casefold()
    directories: list[tuple[int, str, dict[str, str]]] = []
    visited = 0
    skipped = 0
    truncated = False

    def handle_walk_error(_: OSError) -> None:
        nonlocal skipped
        skipped += 1

    for current, names, _ in os.walk(root, onerror=handle_walk_error, followlinks=False):
        names.sort(key=str.casefold)
        current_path = Path(current)
        parent = str(current_path)
        for name in names:
            if visited >= _MAX_DIRECTORY_SEARCH_VISITED:
                truncated = True
                break
            visited += 1
            directory_path = current_path / name
            full_path = str(directory_path)
            name_lower = name.casefold()
            if needle == name_lower:
                relevance = 0
            elif name_lower.startswith(needle):
                relevance = 1
            elif any(part.startswith(needle) for part in re.split(r"[\\W_-]+", name_lower)):
                relevance = 2
            elif needle in name_lower:
                relevance = 3
            else:
                continue
            directories.append((
                relevance,
                full_path.casefold(),
                {
                    "name": name,
                    "path": full_path,
                    "parent": parent,
                    "drive": _drive_label(directory_path),
                },
            ))
        if truncated:
            break

    directories.sort(key=lambda item: (item[0], item[1]))
    matching_count = len(directories)
    results = [item[2] for item in directories[:_MAX_DIRECTORY_SEARCH_RESULTS]]
    if matching_count > _MAX_DIRECTORY_SEARCH_RESULTS:
        truncated = True
    return {
        "directories": results,
        "truncated": truncated,
        "visited": min(visited, _MAX_DIRECTORY_SEARCH_VISITED),
        "skipped": skipped,
    }


def _change_workspace(state: _AppState, workspace: str) -> Path:
    if not workspace.strip():
        raise ValueError("path must be a non-empty directory path")
    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    with state.lock:
        session = Session.create(state.provider_name, state.provider.model)
        session.session_id = f"{session.session_id}_{uuid.uuid4().hex[:8]}"
        session.workspace = str(state.workspace)
        session.workspace = str(root)
        state.workspace = root
        state.session = session
        state.tool_context = ToolContext(workspace_root=root)
        state.sessions[session.session_id] = session
    return root


def _session_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if hasattr(block, "text"):
                text = block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
            else:
                continue
            if isinstance(text, str):
                text_blocks.append(text)
        return "\n".join(text_blocks)
    return ""


def _session_title(session: Session, fallback: str = "") -> str:
    if session.title and session.title.strip():
        title = session.title.strip()
        return title[:48] + ("…" if len(title) > 48 else "")
    for message in session.conversation.messages:
        if message.role == "user" and not message._is_internal:
            text = _session_text(message.content).strip()
            if text:
                return text[:48] + ("…" if len(text) > 48 else "")
    if fallback:
        return fallback[:48] + ("…" if len(fallback) > 48 else "")
    return f"Saved chat · {session.session_id}"


def _resolve_session_workspace(session: Session, fallback: Path) -> Path:
    path = Path(session.workspace).expanduser() if session.workspace else fallback
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Saved chat working folder is not a directory: {resolved}")
    session.workspace = str(resolved)
    return resolved


def _saved_sessions(state: _AppState | None = None) -> list[Session]:
    session_dir = Path.home() / ".clawd" / "sessions"
    sessions_by_id: dict[str, Session] = {}
    if session_dir.exists():
        for session_file in session_dir.glob("*.json"):
            if not _SESSION_ID_PATTERN.fullmatch(session_file.stem):
                continue
            try:
                session = Session.load(session_file.stem)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                # A truncated or corrupt file must not hide the other saved chats.
                continue
            if session is not None:
                sessions_by_id[session.session_id] = session
    running_prompts: dict[str, str] = {}
    if state is not None:
        with state.lock:
            fallback_workspace = state.default_workspace or state.workspace
            running_prompts = {
                job.session_id: job.prompt
                for job in state.jobs.values()
                if job.status in {"running", "waiting"}
            }
            sessions_by_id.update({
                session_id: session
                for session_id, session in state.sessions.items()
                if session.conversation.messages or session_id in running_prompts
            })
        for session in sessions_by_id.values():
            if session.workspace is None:
                session.workspace = str(fallback_workspace)
    sessions = list(sessions_by_id.values())
    return sorted(sessions, key=lambda session: session.updated_at, reverse=True)


def _session_transcript(session: Session) -> list[dict[str, str]]:
    transcript = []
    for message in session.conversation.messages:
        if message.role not in {"user", "assistant"} or message._is_internal:
            continue
        text = _session_text(message.content).strip()
        if text:
            transcript.append({"role": message.role, "text": text, "ts": message.timestamp})
    return transcript


def _truncate_conversation_for_retry(
    session: Session, mode: str, message: str = ""
) -> str:
    """Truncate the conversation so a turn can be re-run. Returns the prompt.

    - "regenerate": drop trailing assistant content after the last user text
      message; the caller re-sends that user message.
    - "edit": drop the last user message matching `message` and everything
      after it; the caller submits the edited text as a new message.

    Raises ValueError when there is nothing to retry.
    """
    messages = session.conversation.messages
    if mode == "regenerate":
        idx = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if (
                m.role == "user"
                and not m._is_internal
                and isinstance(m.content, str)
                and m.content.strip()
            ):
                idx = i
                break
        if idx is None:
            raise ValueError("Nothing to regenerate: no user message found")
        prompt = messages[idx].content.strip()
        del messages[idx + 1 :]
        return prompt
    if mode == "edit":
        want = message.strip()
        if not want:
            raise ValueError("edit mode needs the original message text")
        idx = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if (
                m.role == "user"
                and not m._is_internal
                and isinstance(m.content, str)
                and m.content.strip() == want
            ):
                idx = i
                break
        if idx is None:
            raise ValueError("Message to edit not found")
        del messages[idx:]
        return want
    raise ValueError(f"Unknown retry mode: {mode!r}")


def _start_new_session(state: _AppState) -> None:
    with state.lock:
        session = Session.create(state.provider_name, state.provider.model)
        session.session_id = f"{session.session_id}_{uuid.uuid4().hex[:8]}"
        session.workspace = str(state.workspace)
        state.session = session
        state.tool_context = ToolContext(workspace_root=state.workspace)
        state.sessions[session.session_id] = session


def _session_has_running_job(state: _AppState, session_id: str) -> bool:
    return any(
        job.session_id == session_id and job.status in {"running", "waiting"}
        for job in state.jobs.values()
    )


def _tool_detail(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError, RecursionError) as exc:
        text = f"Unable to display tool data: {exc}"
    if len(text) > _MAX_TOOL_DETAIL_CHARS:
        text = text[:_MAX_TOOL_DETAIL_CHARS] + "\n… (truncated)"
    return text


_STREAM_BUFFER_SIZE = 2000
_DIFF_MAX_CHARS = 8000


def _job_emit(job: _Job, kind: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append a sequenced event to a job's SSE stream.

    Thread-safe: takes the job's stream condition (never state.lock), so it
    can be called from worker callbacks that already hold state.lock.
    """
    event = {"seq": 0, "kind": kind, **(data or {})}
    with job.stream_cond:
        job.seq += 1
        event["seq"] = job.seq
        job.stream.append(event)
        if len(job.stream) > _STREAM_BUFFER_SIZE:
            del job.stream[: _STREAM_BUFFER_SIZE // 2]
        job.stream_cond.notify_all()
    return event


def _tool_input_diff(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Best-effort unified diff for file-editing tools, for review in the UI."""
    name = (tool_name or "").lower()
    old: str | None = None
    new: str | None = None
    if name == "edit":
        old = tool_input.get("old_string")
        new = tool_input.get("new_string")
    elif name == "write":
        old = ""
        new = tool_input.get("content")
    if old is None or not isinstance(new, str) or not isinstance(old, str):
        return None
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile="before", tofile="after", lineterm="",
    )
    body = [line for line in lines if not line.startswith(("---", "+++"))]
    text = "\n".join(body).strip("\n")
    if not text:
        return None
    if len(text) > _DIFF_MAX_CHARS:
        text = text[:_DIFF_MAX_CHARS] + "\n… (diff truncated)"
    return text


def _diff_stat(diff_text: str) -> dict[str, int]:
    """Count added/removed content lines in a unified diff (excludes headers)."""
    added = removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"added": added, "removed": removed}


def _apply_tool_timing(
    timings: dict[str, float], payload: dict[str, Any], event: ToolEvent
) -> None:
    """Attach elapsed_ms to a tool_result payload from tool_use start times.

    Pure apart from reading the clock; safe to call for every event kind.
    """
    if event.kind == "tool_use":
        if event.tool_use_id:
            timings[event.tool_use_id] = time.monotonic()
    elif event.tool_use_id and event.tool_use_id in timings:
        start = timings.pop(event.tool_use_id)
        payload["elapsed_ms"] = max(0, int((time.monotonic() - start) * 1000))


def _tool_event_payload(event: ToolEvent) -> dict[str, Any]:
    """Build the browser payload for one agent-loop ToolEvent (pure)."""
    if event.kind == "tool_use":
        payload = {
            "kind": event.kind,
            "tool_use_id": event.tool_use_id,
            "tool_name": event.tool_name,
            "summary": summarize_tool_use(event.tool_name, event.tool_input or {}),
            "input": _tool_detail(event.tool_input or {}),
            "output": "",
            "error": False,
        }
        diff = _tool_input_diff(event.tool_name, event.tool_input or {})
        if diff:
            payload["diff"] = diff
            payload["diff_stat"] = _diff_stat(diff)
        return payload
    if event.kind == "tool_result":
        return {
            "kind": event.kind,
            "tool_use_id": event.tool_use_id,
            "tool_name": event.tool_name,
            "summary": summarize_tool_result(event.tool_name, event.tool_output),
            "input": "",
            "output": _tool_detail(event.tool_output),
            "error": event.is_error,
        }
    return {
        "kind": event.kind,
        "tool_use_id": event.tool_use_id,
        "tool_name": event.tool_name,
        "summary": "Tool call failed",
        "input": _tool_detail(event.tool_input or {}),
        "output": event.error or f"{event.tool_name} failed",
        "error": True,
    }


def _wait_for_interaction(state: _AppState, job: _Job, payload: dict[str, Any]) -> dict[str, Any]:
    interaction = _Interaction(payload={"id": uuid.uuid4().hex, **payload})
    with state.lock:
        if job.cancel_event.is_set():
            raise AgentLoopCancelled("Stopped by the user.")
        job.interaction = interaction
        job.activity = "waiting"
    _job_emit(job, "interaction", {"interaction": interaction.payload})
    _job_emit(job, "activity", {"activity": "waiting"})
    if not interaction.event.wait(_INTERACTION_TIMEOUT_SECONDS):
        with state.lock:
            job.interaction = None
        _job_emit(job, "interaction", {"interaction": None})
        raise TimeoutError("Timed out waiting for a response in the browser.")
    with state.lock:
        job.interaction = None
        job.activity = "tool"
    _job_emit(job, "interaction", {"interaction": None})
    _job_emit(job, "activity", {"activity": "tool"})
    if job.cancel_event.is_set():
        raise AgentLoopCancelled("Stopped by the user.")
    return interaction.response or {}


def _run_job(state: _AppState, job: _Job, message: str) -> None:
    session = job.session
    tool_context = job.tool_context
    if session is None or tool_context is None:
        raise ValueError("Chat job is missing its isolated session or tool context.")
    job_started = time.monotonic()
    tool_timings: dict[str, float] = {}

    def on_text(chunk: str) -> None:
        with state.lock:
            job.text += chunk
            job.timeline.append({"kind": "text", "text": chunk})
            job.activity = "typing"
        _job_emit(job, "text", {"text": chunk})

    def on_activity(activity: str) -> None:
        with state.lock:
            job.activity = activity
        _job_emit(job, "activity", {"activity": activity})

    def on_event(event: ToolEvent) -> None:
        payload = _tool_event_payload(event)
        _apply_tool_timing(tool_timings, payload, event)
        with state.lock:
            job.events.append(payload)
            job.timeline.append({"kind": "tool", "event": payload})
            job.activity = "tool"
        _job_emit(job, "tool", {"event": payload})

    def permission_handler(tool_name: str, prompt: str, suggestion: str | None) -> tuple[bool, bool]:
        response = _wait_for_interaction(
            state,
            job,
            {
                "kind": "permission",
                "tool_name": tool_name,
                "message": prompt,
                "suggestion": suggestion,
            },
        )
        return bool(response.get("allowed")), False

    def ask_user(questions: list[dict[str, Any]]) -> dict[str, str]:
        response = _wait_for_interaction(
            state,
            job,
            {"kind": "questions", "questions": questions},
        )
        answers = response.get("answers", {})
        if not isinstance(answers, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in answers.items()
        ):
            raise ValueError("Browser response to AskUserQuestion must be a string-to-string object.")
        return answers

    tool_context.permission_handler = permission_handler
    tool_context.ask_user = ask_user
    try:
        if not job.message_added:
            session.conversation.add_user_message(message)
        result = run_agent_loop(
            conversation=session.conversation,
            provider=state.provider,
            tool_registry=state.registry,
            tool_context=tool_context,
            stream=True,
            on_event=on_event,
            on_text_chunk=on_text,
            on_activity=on_activity,
            should_stop=job.cancel_event.is_set,
        )
        if not job.text and result.response_text:
            with state.lock:
                job.text = result.response_text
                job.timeline.append({"kind": "text", "text": result.response_text})
        session.save()
        cancelled = job.cancel_event.is_set() or getattr(result, "cancelled", False)
        usage = getattr(result, "usage", None)
        turn_cost = _turn_cost_usd(state, usage)
        with state.lock:
            job.status = "stopped" if cancelled else "completed"
            job.activity = ""
            job.elapsed_ms = int((time.monotonic() - job_started) * 1000)
            job.usage = dict(usage) if usage else None
            if turn_cost is not None and job.usage is not None:
                job.usage["cost_usd"] = round(turn_cost, 6)
                state.session_costs[job.session_id] = round(
                    state.session_costs.get(job.session_id, 0.0) + turn_cost, 6
                )
        _job_emit(job, "status", {"status": job.status, "error": None, "usage": job.usage,
                                   "elapsed_ms": job.elapsed_ms})
    except AgentLoopCancelled:
        # Keep the partial turn so the chat history matches what the user saw.
        try:
            session.save()
        except OSError:
            pass
        with state.lock:
            job.status = "stopped"
            job.activity = ""
            job.elapsed_ms = int((time.monotonic() - job_started) * 1000)
        _job_emit(job, "status", {"status": "stopped", "error": None, "usage": job.usage,
                                  "elapsed_ms": job.elapsed_ms})
    except Exception as exc:
        with state.lock:
            job.error = str(exc)
            job.status = "error"
            job.activity = ""
            job.elapsed_ms = int((time.monotonic() - job_started) * 1000)
        _job_emit(job, "status", {"status": "error", "error": job.error, "usage": job.usage,
                                  "elapsed_ms": job.elapsed_ms})


def _make_handler(state: _AppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ClawdWeb/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            # The browser UI polls the server constantly (job status every
            # 700ms, directory searches, session refreshes), so request logs
            # would flood the terminal. Keep the server quiet by default.
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _check_auth(self) -> bool:
            """Optional shared-token gate. Set CLAWD_WEB_TOKEN to require it.

            Uses HTTP Basic auth so browsers handle it natively (login prompt
            once, credentials sent on every request including fetch/SSE).
            The username is ignored; the password must equal the token.
            """
            token = os.environ.get("CLAWD_WEB_TOKEN")
            if not token:
                return True
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
                except Exception:
                    decoded = ""
                password = decoded.split(":", 1)[1] if ":" in decoded else ""
                # constant-time compare
                if hmac.compare_digest(password, token):
                    return True
            body = b"Unauthorized"
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="clawd"')
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def do_GET(self) -> None:
            if not self._check_auth():
                return
            path = urlsplit(self.path).path
            if path == "/api/dev-version":
                with state.lock:
                    payload = {
                        "enabled": state.dev_reload_version is not None,
                        "version": state.dev_reload_version,
                        "busy": any(job.status in {"running", "waiting"} for job in state.jobs.values()),
                    }
                self._json(200, payload)
                return
            if path == "/":
                self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                with state.lock:
                    workspace = str(state.workspace)
                self._json(200, {
                    "workspace": workspace,
                    "provider": state.provider_name,
                    "model": state.provider.model,
                    "session_id": state.session.session_id,
                })
                return
            if path == "/api/settings":
                config = load_config()
                configured_providers = config.get("providers", {})
                providers = []
                for name, info in PROVIDER_INFO.items():
                    provider_config = configured_providers.get(name, {})
                    api_key = provider_config.get("api_key", "")
                    providers.append({
                        "name": name,
                        "label": info["label"],
                        "models": list(info["available_models"]),
                        "configured": bool(api_key),
                        "default_model": provider_config.get("default_model") or info["default_model"],
                    })
                with state.lock:
                    current_provider = state.provider_name
                    current_model = state.provider.model
                self._json(200, {
                    "providers": providers,
                    "provider": current_provider,
                    "model": current_model,
                })
                return
            if path == "/api/sessions":
                try:
                    sessions = _saved_sessions(state)
                    for session in sessions:
                        if session.workspace is None:
                            session.workspace = str(state.default_workspace or state.workspace)
                    with state.lock:
                        running_prompts = {
                            job.session_id: job.prompt
                            for job in state.jobs.values()
                            if job.status in {"running", "waiting"}
                        }
                        session_costs = dict(state.session_costs)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._json(500, {"error": f"Could not list saved chats: {exc}"})
                    return
                self._json(200, {
                    "sessions": [
                        {
                            "session_id": session.session_id,
                            "title": _session_title(session, running_prompts.get(session.session_id, "")),
                            "updated_at": session.updated_at,
                            "running": _session_has_running_job(state, session.session_id),
                            "workspace": session.workspace,
                            "session_cost_usd": round(session_costs.get(session.session_id, 0.0), 4),
                        }
                        for session in sessions
                    ],
                })
                return
            if path == "/api/directories":
                requested = parse_qs(urlsplit(self.path).query).get("path", [None])[0]
                try:
                    listing = _list_directories(state.workspace, requested)
                except (OSError, RuntimeError, ValueError) as exc:
                    self._json(400, {"error": f"Could not browse directory: {exc}"})
                    return
                self._json(200, listing)
                return
            if path == "/api/directories/search":
                params = parse_qs(urlsplit(self.path).query)
                query = params.get("query", [""])[0]
                requested_path = params.get("path", [str(state.workspace)])[0]
                try:
                    results = _search_directories(state.workspace, requested_path, query)
                except (OSError, RuntimeError, ValueError) as exc:
                    self._json(400, {"error": f"Could not search folders: {exc}"})
                    return
                self._json(200, results)
                return
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stream":
                with state.lock:
                    job = state.jobs.get(parts[2])
                if job is None:
                    self._json(404, {"error": "Job not found"})
                    return
                query = parse_qs(urlsplit(self.path).query)
                try:
                    last_seq = int(query.get("last_seq", ["0"])[0])
                except (TypeError, ValueError):
                    last_seq = 0
                self._serve_job_stream(job, max(0, last_seq))
                return
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                with state.lock:
                    job = state.jobs.get(parts[2])
                    if job is None:
                        self._json(404, {"error": "Job not found"})
                        return
                    payload = {
                        "job_id": job.job_id,
                        "session_id": job.session_id,
                        "status": job.status,
                        "activity": job.activity,
                        "text": job.text,
                        "events": list(job.events),
                        "timeline": list(job.timeline),
                        "interaction": job.interaction.payload if job.interaction else None,
                        "error": job.error,
                        "usage": job.usage,
                        "seq": job.seq,
                        "elapsed_ms": job.elapsed_ms,
                    }
                self._json(200, payload)
                return
            self._json(404, {"error": "Not found"})

        def _serve_job_stream(self, job: _Job, last_seq: int) -> None:
            """Stream job deltas as server-sent events with seq-based replay."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 2000\n\n")
                self.wfile.flush()
                cursor = last_seq
                while True:
                    with job.stream_cond:
                        pending = [event for event in job.stream if event["seq"] > cursor]
                        while not pending:
                            if job.status in {"completed", "error", "stopped"}:
                                return
                            job.stream_cond.wait(timeout=20)
                            pending = [event for event in job.stream if event["seq"] > cursor]
                        cursor = pending[-1]["seq"]
                    for event in pending:
                        line = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                        self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                    if pending[-1]["kind"] == "status":
                        return
            except OSError:
                # Browser navigated away or the connection dropped; the job
                # itself is unaffected.
                pass

        def _read_json(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "Invalid Content-Length"})
                return None
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                self._json(413, {"error": "Request body is empty or too large"})
                return None
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "Request body must be valid JSON"})
                return None
            if not isinstance(payload, dict):
                self._json(400, {"error": "Request body must be a JSON object"})
                return None
            return payload

        def do_POST(self) -> None:
            if not self._check_auth():
                return
            path = urlsplit(self.path).path
            payload = self._read_json()
            if payload is None:
                return
            if path == "/api/chat/retry":
                # Conversation surgery for regenerate/edit: truncates the
                # conversation, the browser then re-submits via /api/chat.
                session_id = payload.get("session_id")
                mode = payload.get("mode")
                if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
                    self._json(400, {"error": "session_id must be a valid session ID"})
                    return
                if mode not in {"regenerate", "edit"}:
                    self._json(400, {"error": "mode must be 'regenerate' or 'edit'"})
                    return
                with state.lock:
                    session = state.sessions.get(session_id)
                    if session is None:
                        try:
                            session = Session.load(session_id)
                        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                            self._json(500, {"error": f"Could not load chat session: {exc}"})
                            return
                        if session is not None:
                            state.sessions[session_id] = session
                    if session is None:
                        self._json(404, {"error": "Chat session not found"})
                        return
                    if _session_has_running_job(state, session_id):
                        self._json(409, {"error": "This chat already has a turn running"})
                        return
                    try:
                        prompt = _truncate_conversation_for_retry(
                            session, mode, payload.get("message") or ""
                        )
                    except ValueError as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    try:
                        session.save()
                    except OSError as exc:
                        self._json(500, {"error": f"Could not save chat: {exc}"})
                        return
                self._json(200, {"ok": True, "prompt": prompt})
                return
            if path == "/api/chat":
                message = payload.get("message")
                if not isinstance(message, str) or not message.strip():
                    self._json(400, {"error": "message must be a non-empty string"})
                    return
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
                    self._json(400, {"error": "session_id must be a valid session ID"})
                    return
                with state.lock:
                    session = state.sessions.get(session_id)
                    if session is None:
                        try:
                            session = Session.load(session_id)
                        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                            self._json(500, {"error": f"Could not load chat session: {exc}"})
                            return
                        if session is not None:
                            state.sessions[session_id] = session
                    if session is None:
                        self._json(404, {"error": "Chat session not found"})
                        return
                    if _session_has_running_job(state, session_id):
                        self._json(409, {"error": "This chat already has a turn running"})
                        return
                    try:
                        session_workspace = _resolve_session_workspace(
                            session,
                            state.default_workspace or state.workspace,
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        self._json(400, {"error": f"Could not use this chat's working folder: {exc}"})
                        return
                    state.sessions[session_id] = session
                    tool_context = ToolContext(workspace_root=session_workspace)
                    job = _Job(
                        job_id=uuid.uuid4().hex,
                        session_id=session_id,
                        session=session,
                        tool_context=tool_context,
                        prompt=message.strip(),
                        message_added=True,
                    )
                    session.conversation.add_user_message(message.strip())
                    state.jobs[job.job_id] = job
                thread = threading.Thread(
                    target=_run_job,
                    args=(state, job, message.strip()),
                    daemon=True,
                    name=f"clawd-web-{job.job_id[:8]}",
                )
                try:
                    thread.start()
                except RuntimeError as exc:
                    with state.lock:
                        job.status = "error"
                        job.activity = ""
                        job.error = f"Could not start chat worker: {exc}"
                    self._json(503, {"error": job.error})
                    return
                self._json(202, {"job_id": job.job_id, "session_id": job.session_id})
                return
            if path == "/api/session/new":
                _start_new_session(state)
                self._json(200, {"ok": True, "session_id": state.session.session_id})
                return
            if path == "/api/session/load":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
                    self._json(400, {"error": "session_id must be a valid saved session ID"})
                    return
                with state.lock:
                    try:
                        session = state.sessions.get(session_id) or Session.load(session_id)
                    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                        self._json(500, {"error": f"Could not load saved chat: {exc}"})
                        return
                    if session is None:
                        self._json(404, {"error": "Saved chat not found"})
                        return
                    try:
                        session_workspace = _resolve_session_workspace(
                            session,
                            state.default_workspace or state.workspace,
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        self._json(400, {"error": f"Could not open saved chat working folder: {exc}"})
                        return
                    state.workspace = session_workspace
                    state.session = session
                    state.tool_context = ToolContext(workspace_root=session_workspace)
                    state.sessions[session.session_id] = session
                self._json(200, {
                    "session_id": session.session_id,
                    "title": _session_title(session),
                    "workspace": str(session_workspace),
                    "messages": _session_transcript(session),
                })
                return
            if path == "/api/session/delete":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
                    self._json(400, {"error": "session_id must be a valid saved session ID"})
                    return
                with state.lock:
                    if _session_has_running_job(state, session_id):
                        self._json(409, {"error": "This chat is still running and cannot be deleted yet."})
                        return
                    session_file = Path.home() / ".clawd" / "sessions" / f"{session_id}.json"
                    try:
                        session_file.unlink()
                    except FileNotFoundError:
                        self._json(404, {"error": "Saved chat not found"})
                        return
                    except OSError as exc:
                        self._json(500, {"error": f"Could not delete saved chat: {exc}"})
                        return
                    state.sessions.pop(session_id, None)
                    if state.session.session_id == session_id:
                        session = Session.create(state.provider_name, state.provider.model)
                        session.session_id = f"{session.session_id}_{uuid.uuid4().hex[:8]}"
                        state.session = session
                        state.sessions[session.session_id] = session
                        state.tool_context = ToolContext(workspace_root=state.workspace)
                self._json(200, {
                    "ok": True,
                    "session_id": state.session.session_id,
                })
                return
            if path == "/api/session/rename":
                session_id = payload.get("session_id")
                title = payload.get("title")
                if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
                    self._json(400, {"error": "session_id must be a valid saved session ID"})
                    return
                if not isinstance(title, str) or not title.strip():
                    self._json(400, {"error": "title must be a non-empty string"})
                    return
                title = title.strip()
                if len(title) > 120:
                    self._json(400, {"error": "title must be at most 120 characters"})
                    return
                with state.lock:
                    session = state.sessions.get(session_id)
                    if session is None:
                        try:
                            session = Session.load(session_id)
                        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                            self._json(500, {"error": f"Could not load saved chat: {exc}"})
                            return
                        if session is None:
                            self._json(404, {"error": "Saved chat not found"})
                            return
                        state.sessions[session_id] = session
                    session.title = title
                    try:
                        session.save()
                    except OSError as exc:
                        self._json(500, {"error": f"Could not save chat: {exc}"})
                        return
                self._json(200, {"ok": True, "session_id": session_id, "title": title})
                return
            if path == "/api/workspace":
                workspace = payload.get("path")
                if not isinstance(workspace, str):
                    self._json(400, {"error": "path must be a directory path string"})
                    return
                try:
                    selected = _change_workspace(state, workspace)
                except RuntimeError as exc:
                    self._json(409, {"error": str(exc)})
                    return
                except (OSError, ValueError) as exc:
                    self._json(400, {"error": f"Could not use directory: {exc}"})
                    return
                self._json(200, {"workspace": str(selected), "session_id": state.session.session_id})
                return
            if path == "/api/settings":
                provider_name = payload.get("provider")
                model = payload.get("model")
                api_key = payload.get("api_key")
                base_url = payload.get("base_url")
                if provider_name is not None and (
                    not isinstance(provider_name, str) or provider_name not in PROVIDER_INFO
                ):
                    self._json(400, {"error": "provider must be a known provider name"})
                    return
                if model is not None and (not isinstance(model, str) or not model.strip()):
                    self._json(400, {"error": "model must be a non-empty string"})
                    return
                if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
                    self._json(400, {"error": "api_key must be a non-empty string"})
                    return
                if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
                    self._json(400, {"error": "base_url must be a non-empty string"})
                    return
                selected_provider = provider_name or state.provider_name
                selected_model = (model or state.provider.model).strip()
                try:
                    provider_config = get_provider_config(selected_provider)
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                new_api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
                effective_key = new_api_key or provider_config.get("api_key", "")
                if not effective_key:
                    self._json(400, {
                        "error": (
                            f"No API key saved for {selected_provider}. "
                            "Add one in Settings or run `clawd login`."
                        )
                    })
                    return
                effective_base_url = (
                    base_url.strip()
                    if isinstance(base_url, str) and base_url.strip()
                    else provider_config.get("base_url")
                )
                try:
                    provider = get_provider_class(selected_provider)(
                        api_key=effective_key,
                        base_url=effective_base_url,
                        model=selected_model,
                    )
                except (TypeError, ValueError) as exc:
                    self._json(400, {"error": f"Could not create provider: {exc}"})
                    return
                try:
                    if new_api_key is not None:
                        set_api_key(
                            selected_provider,
                            new_api_key,
                            base_url=effective_base_url,
                            default_model=selected_model,
                        )
                    else:
                        config = load_config()
                        saved_config = config.setdefault("providers", {}).setdefault(selected_provider, {})
                        saved_config["default_model"] = selected_model
                        if isinstance(base_url, str) and base_url.strip():
                            saved_config["base_url"] = base_url.strip()
                        config["default_provider"] = selected_provider
                        save_config(config)
                except (OSError, ValueError) as exc:
                    self._json(400, {"error": f"Could not save settings: {exc}"})
                    return
                with state.lock:
                    state.provider_name = selected_provider
                    state.provider = provider
                    state.session.provider = selected_provider
                    state.session.model = selected_model
                self._json(200, {
                    "ok": True,
                    "provider": selected_provider,
                    "model": selected_model,
                })
                return
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                with state.lock:
                    job = state.jobs.get(parts[2])
                    if job is None:
                        self._json(404, {"error": "Job not found"})
                        return
                    job.cancel_event.set()
                    interaction = job.interaction
                    if interaction is not None:
                        # Unblock the worker; it will see the cancel and stop.
                        if interaction.payload.get("kind") == "permission":
                            interaction.response = {"allowed": False}
                        else:
                            interaction.response = {"answers": {}}
                        job.interaction = None
                        interaction.event.set()
                    status = job.status
                if interaction is not None:
                    _job_emit(job, "interaction", {"interaction": None})
                self._json(200, {"ok": True, "status": status})
                return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "interaction":
                with state.lock:
                    job = state.jobs.get(parts[2])
                    interaction = job.interaction if job else None
                    if interaction is None or payload.get("interaction_id") != interaction.payload["id"]:
                        self._json(409, {"error": "No matching pending interaction"})
                        return
                    if interaction.payload["kind"] == "permission":
                        if not isinstance(payload.get("allowed"), bool):
                            self._json(400, {"error": "allowed must be a boolean"})
                            return
                        interaction.response = {"allowed": payload["allowed"]}
                    else:
                        if not isinstance(payload.get("answers"), dict):
                            self._json(400, {"error": "answers must be an object"})
                            return
                        interaction.response = {"answers": payload["answers"]}
                    job.interaction = None
                    interaction.event.set()
                _job_emit(job, "interaction", {"interaction": None})
                self._json(200, {"ok": True})
                return
            self._json(404, {"error": "Not found"})

    return Handler


def _local_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def _source_signature(root: Path = _SOURCE_ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _reload_worker_busy(port: int) -> bool | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/dev-version", timeout=1) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("busy"), bool):
            raise ValueError("Invalid reload worker status response")
        return bool(payload.get("busy"))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        print(f"Waiting for the reload worker status endpoint: {exc}")
        return None


def _stop_reload_worker(worker: subprocess.Popen) -> None:
    if worker.poll() is not None:
        return
    worker.terminate()
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()


def _run_reload_supervisor(port: int) -> int:
    signature = _source_signature()
    module_root = str(_SOURCE_ROOT.parent)
    worker_environment = os.environ.copy()
    existing_pythonpath = worker_environment.get("PYTHONPATH")
    worker_environment["PYTHONPATH"] = (
        module_root + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else module_root
    )
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "web",
        "--port",
        str(port),
        "--_reload-worker",
    ]
    print("Clawd web reload mode is watching Python files under src/.")
    print("Connected browsers refresh automatically after the server reloads.")
    worker: subprocess.Popen | None = None
    try:
        while True:
            worker = subprocess.Popen(command, cwd=os.getcwd(), env=worker_environment)
            pending_signature: str | None = None
            stable_polls = 0
            busy_notice_shown = False
            while worker.poll() is None:
                time.sleep(_RELOAD_POLL_SECONDS)
                try:
                    current_signature = _source_signature()
                except OSError as exc:
                    print(f"Could not inspect source files for web reload: {exc}")
                    continue
                if current_signature == signature:
                    pending_signature = None
                    stable_polls = 0
                    busy_notice_shown = False
                    continue
                if current_signature == pending_signature:
                    stable_polls += 1
                else:
                    pending_signature = current_signature
                    stable_polls = 1
                if stable_polls < _RELOAD_STABLE_POLLS:
                    continue
                busy = _reload_worker_busy(port)
                if busy is None:
                    continue
                if busy:
                    if not busy_notice_shown:
                        print("Source change detected; waiting for active chat turns to finish before reloading.")
                        busy_notice_shown = True
                    continue
                print("Source change detected. Restarting the Clawd web server...")
                _stop_reload_worker(worker)
                signature = current_signature
                break
            else:
                return worker.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping Clawd web reload mode.")
        if worker is not None:
            _stop_reload_worker(worker)
        return 0


def run_web_app(
    workspace: str | Path,
    port: int = 8765,
    *,
    reload: bool = False,
    reload_worker: bool = False,
) -> int:
    """Start the browser UI, bound to LAN interfaces for phone access."""
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    if reload and reload_worker:
        raise ValueError("Reload mode cannot be enabled for the reload worker.")
    if reload:
        return _run_reload_supervisor(port)
    try:
        state = _create_state(workspace)
        if reload_worker:
            state.dev_reload_version = _source_signature()
        server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(state))
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    except OSError as exc:
        print(f"Error starting Clawd web app: {exc}")
        return 1

    server.daemon_threads = True
    print(f"Clawd web app is using {state.provider_name} ({state.provider.model}).")
    print(f"Workspace: {state.workspace}")
    if os.environ.get("CLAWD_WEB_TOKEN"):
        print("Access token is set: browsers will prompt for credentials (any username, token as password).")
    else:
        print("Warning: Clawd web has no access token and is available to anyone who can reach this port.")
    print("Open http://127.0.0.1:%d on this computer." % port)
    for address in _local_addresses():
        print(f"On your phone, open http://{address}:{port} on the same Wi-Fi.")
    print("Keep this terminal open. Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Clawd web app.")
    finally:
        server.server_close()
    return 0
