"""Llm against a fake LiteLLM gateway: schema errors are fed back, failures become LlmUnavailable."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from librarian.agent.llm import Llm, LlmUnavailable


class Pick(BaseModel):
    slug: str
    count: int


def completion(content: str) -> dict:
    """A chat completion in the shape LiteLLM returns."""
    return {"id": "chatcmpl-0", "object": "chat.completion", "model": "librarian-chat",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def gateway(monkeypatch):
    """Fake gateway; yields (queued replies, received request bodies)."""
    replies, seen = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            """Answer with the next queued completion."""
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(completion(replies.pop(0))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """Keep pytest output quiet."""

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LITELLM_URL", f"http://127.0.0.1:{server.server_port}")
    yield replies, seen
    server.shutdown()


def test_schema_error_is_fed_back(settings, gateway):
    replies, seen = gateway
    replies += ['{"slug": "printer"}', '{"slug": "printer", "count": 2}']
    result = Llm(settings).chat_json("system", "user", Pick, model="librarian-chat")
    assert result == Pick(slug="printer", count=2)
    feedback = seen[-1]["messages"][-1]["content"]
    assert "count" in feedback and "Field required" in feedback
    assert seen[-1]["model"] == "librarian-chat" and seen[-1]["temperature"] == 0.1


def test_unusable_replies_raise_llm_unavailable(settings, gateway):
    replies, _ = gateway
    replies += ["no json"] * (1 + settings.cfg.llm_retries)
    with pytest.raises(LlmUnavailable):
        Llm(settings).chat_json("system", "user", Pick, model="librarian-chat")


def test_dead_gateway_raises_llm_unavailable(settings, monkeypatch):
    monkeypatch.setenv("LITELLM_URL", "http://127.0.0.1:9")
    with pytest.raises(LlmUnavailable):
        Llm(settings).chat_json("system", "user", Pick, model="librarian-chat")
