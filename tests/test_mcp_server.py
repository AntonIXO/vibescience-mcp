"""MCP-level smoke test: drive the real FastMCP server in-memory over the
Client transport, exercising the full §12 loop through the tool interface."""

import asyncio
import json
import os
import tempfile

from fastmcp import Client


def _data(res):
    """Extract the tool's structured dict from a FastMCP CallToolResult."""
    if getattr(res, "structured_content", None):
        sc = res.structured_content
        # FastMCP wraps non-dict returns under 'result'; ours already return dict
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    # fallback: parse text content
    return json.loads(res.content[0].text)


async def _run(tmp):
    os.environ["VIBESCIENCE_VAULT"] = tmp
    # import AFTER env is set so the module-level VAULT picks it up
    import importlib
    import vibescience_mcp.server as srv
    importlib.reload(srv)

    async with Client(srv.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
        assert {"recall", "propose_hypothesis", "close_experiment",
                "causal_map", "calibration"} <= tools, tools

        # guide resource present
        res = await client.read_resource("vibescience://guide")
        guide_text = res[0].text
        assert "pre-mortem gate" in guide_text

        # full loop
        assert _data(await client.call_tool("create_problem",
                    {"title": "smoke problem", "id": "sp"}))["ok"]

        # recall empty first time
        r = _data(await client.call_tool("recall", {"problem_id": "sp"}))
        assert r["ok"] and r["data"]["results"] == []

        assert _data(await client.call_tool("register_diagnostic",
                    {"name": "m", "id": "m", "direction": "higher_better"}))["ok"]

        # prediction gate: rejected without a predicted_effect
        bad = _data(await client.call_tool("propose_hypothesis",
                    {"problem_id": "sp", "statement": "no pred", "predicted_effects": []}))
        assert bad["ok"] is False and "predicted_effect" in bad["error"]

        # proper hypothesis
        assert _data(await client.call_tool("propose_hypothesis", {
            "problem_id": "sp", "statement": "m goes up", "id": "h",
            "predicted_effects": [{"diagnostic_id": "m", "direction": "up"}],
        }))["ok"]

        e = _data(await client.call_tool("start_experiment",
                    {"hypothesis_id": "h", "id": "e"}))
        assert e["ok"]

        assert _data(await client.call_tool("record_diagnostics", {
            "experiment_id": "e",
            "measurements": [{"diagnostic_id": "m", "before": 0.1, "after": 0.9}],
        }))["ok"]

        closed = _data(await client.call_tool("close_experiment", {"experiment_id": "e"}))
        assert closed["ok"]
        assert closed["data"]["verdict"] == "supports"
        assert "committing" in closed["data"]["suggested_next_action"]

        cm = _data(await client.call_tool("causal_map", {"problem_id": "sp"}))
        assert cm["ok"] and cm["data"]["edges"] == []  # no interventions applied

        cal = _data(await client.call_tool("calibration", {"diagnostic_id": "m"}))
        assert cal["ok"] and cal["data"]["accuracy"] == 1.0

    print("MCP smoke test OK — tools:", len(tools))


def test_mcp_server_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_run(os.path.join(tmp, "vault")))
