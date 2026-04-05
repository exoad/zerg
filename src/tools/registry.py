from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
import warnings
from typing import Any

_USING_LEGACY_DDG = False

try:
    import ddgs
except ImportError:
    _USING_LEGACY_DDG = True
    import duckduckgo_search as ddgs

logger = logging.getLogger(__name__)

_DOCKER_AVAILABLE: bool | None = None

def _is_docker_available() -> bool:
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is None:
        if shutil.which("docker") is None:
            _DOCKER_AVAILABLE = False
        else:
            try:
                result = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                _DOCKER_AVAILABLE = result.returncode == 0
            except Exception:
                _DOCKER_AVAILABLE = False
    return _DOCKER_AVAILABLE

async def web_search(query: str) -> str:
    """Perform a web search using DuckDuckGo."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return "WEB_SEARCH_ERROR: query must not be empty"

    try:
        if _USING_LEGACY_DDG:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"This package \(`duckduckgo_search`\) has been renamed to `ddgs`",
            )
        ddgs_client = ddgs.DDGS()
        # run in thread since ddgs / duckduckgo_search is synchronous
        results = await asyncio.to_thread(ddgs_client.text, cleaned_query, max_results=5)
        if not isinstance(results, list):
            results = list(results)
        if not results:
            return f"WEB_SEARCH_NO_RESULTS: no results found for query '{cleaned_query}'"
        
        output = [f"Found {len(results)} results:"]
        for idx, r in enumerate(results, 1):
            output.append(f"{idx}. {r.get('title', 'No Title')} - {r.get('href', 'No URL')}")
            output.append(f"   {r.get('body', 'No snippet available')}")
        
        return "\n".join(output)
    except Exception as exc:
        logger.warning(f"Web search failed: {exc}")
        return f"WEB_SEARCH_ERROR: {exc}"


async def execute_python(code: str) -> str:
    """Execute Python code in an isolated Docker container with a short timeout."""
    if not _is_docker_available():
        return "Docker is not available on this system. The execute_python tool requires Docker to be installed and running."

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(code)

        try:
            cmd = (
                f"docker run --rm --network none -v {temp_dir}:/app -w /app "
                "python:3.11-alpine python script.py"
            )
            
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                return "Execution timed out after 10 seconds."
            
            output = ""
            if stdout:
                output += "STDOUT:\n" + stdout.decode().strip() + "\n"
            if stderr:
                output += "STDERR:\n" + stderr.decode().strip() + "\n"
            
            if not output:
                output = "Code executed successfully without producing output."
                
            return output.strip()[:1000]

        except Exception as exc:
            return f"Failed to execute sandboxed python code: {exc}"


async def execute_javascript(code: str) -> str:
    """Execute Javascript code in an isolated Docker Node container."""
    if not _is_docker_available():
        return "Docker is not available on this system. The execute_javascript tool requires Docker to be installed and running."

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.js")
        with open(script_path, "w") as f:
            f.write(code)

        try:
            cmd = (
                f"docker run --rm --network none -v {temp_dir}:/app -w /app "
                "node:20-alpine node script.js"
            )
            
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                return "Execution timed out after 10 seconds."
            
            output = ""
            if stdout:
                output += "STDOUT:\n" + stdout.decode().strip() + "\n"
            if stderr:
                output += "STDERR:\n" + stderr.decode().strip() + "\n"
            
            if not output:
                output = "Code executed successfully without producing output."
                
            return output.strip()[:1000]

        except Exception as exc:
            return f"Failed to execute sandboxed JS code: {exc}"


def get_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a real-time web search for information about current events, code documentation, or general queries using DuckDuckGo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to search the internet for."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute arbitrary sandboxed Python code to compute math, test algorithms, or run logic. Code is isolated, standard libraries only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The python 3.11 script to execute. Must include print() statements to view output."}
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_javascript",
                "description": "Execute arbitrary sandboxed Javascript via NodeJS 20. Code is isolated, core modules only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The javascript code to execute. Use console.log() to view output."}
                    },
                    "required": ["code"],
                },
            },
        }
    ]


async def dispatch_tool(tool_name: str, arguments: dict[str, Any] | str) -> str:
    """Dispatches execution to the relevant tool handler."""
    kwargs = {}
    if isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments)
        except Exception:
            return f"Argument parsing failed. Arguments must be valid JSON: {arguments}"
    elif isinstance(arguments, dict):
        kwargs = arguments
        
    try:
        if tool_name == "web_search":
            result = await web_search(kwargs.get("query", ""))
        elif tool_name == "execute_python":
            result = await execute_python(kwargs.get("code", ""))
        elif tool_name == "execute_javascript":
            result = await execute_javascript(kwargs.get("code", ""))
        else:
            result = f"Unknown tool: {tool_name}"
        args_text = json.dumps(kwargs, indent=2, ensure_ascii=False)
        result_text = str(result).strip()
        logger.info(
            'Tool call "%s"\nargs:\n```json\n%s\n```\nresult:\n```\n%s\n```',
            tool_name,
            args_text,
            result_text,
        )
        return result
    except Exception as exc:
        return f"Tool execution crashed: {exc}"
