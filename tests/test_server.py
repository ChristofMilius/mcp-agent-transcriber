"""tests/test_server.py — server assembly."""
from __future__ import annotations

import asyncio

import mcp.server.mcpserver
from conftest import build_ctx

from mcp_agent_transcriber.context import AppContext
from mcp_agent_transcriber.server import SERVER_NAME, create_server


def _ctx(tmp_project):
    root, env = tmp_project
    base = build_ctx(root, env)
    return AppContext(cfg=base.cfg, engine=base.engine)


class TestCreateServer:
    def test_uses_mcpserver_contract(self, tmp_project):
        server = create_server(_ctx(tmp_project))
        assert isinstance(server, mcp.server.mcpserver.MCPServer)
        assert server.name == SERVER_NAME

    def test_known_tool_names_registered(self, tmp_project):
        server = create_server(_ctx(tmp_project))
        names = {t.name for t in asyncio.run(server.list_tools())}
        assert {
            "video_info",
            "fetch_transcript",
            "download_audio",
            "transcribe_video",
        } <= names
