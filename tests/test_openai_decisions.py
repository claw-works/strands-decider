"""The OpenAI-compatible `POST /v1/decisions` surface, against a stub engine.

These pin the wire contract the same way test_server.py does for System One: the point of
matching OpenAI's shape is that `openai.OpenAI(base_url=...).decisions.create(...)` works
unchanged, so the last tests drive the real SDK through the app.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_server import _StubEngine, _StubModel

from strands_decider import server
from strands_decider.openai_schema import DecisionCreateRequest, to_systemone
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


class _RecordingEngine(_StubEngine):
    """The stub engine, keeping the System One request it was handed."""

    last = None

    def evaluate(self, request):
        _RecordingEngine.last = request
        return super().evaluate(request)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _StubModel()))
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _RecordingEngine())
    return TestClient(server.create_app("checkpoints/strands-decider-test", device="cpu"))


DOCUMENTED = {
    "model": "gpt-6-luna",
    "input": "I was charged twice for my order.",
    "questions": [
        {"type": "predicate", "name": "is_urgent", "instructions": "Does this convey urgency?"},
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this complaint?",
            "choices": [
                {"value": "billing", "description": "Payments, invoices, and refunds."},
                {"value": "technical"},
                {"value": "other", "description": "Requests outside these categories."},
            ],
        },
        {
            "type": "score",
            "name": "frustration",
            "instructions": "How frustrated is the customer?",
            "levels": [
                {"label": "calm"},
                {"label": "frustrated", "description": "Annoyed but polite."},
                {"label": "angry"},
            ],
        },
    ],
}


def test_documented_payload_round_trips(client):
    r = client.post("/v1/decisions", json=DOCUMENTED)
    assert r.status_code == 200, r.text
    body = r.json()

    assert set(body) == {"answers", "model", "usage"}
    assert body["model"] == "strands-decider-test"  # the checkpoint served, not the one asked for
    assert [a["name"] for a in body["answers"]] == ["is_urgent", "department", "frustration"]

    predicate, choice, score = body["answers"]
    assert predicate["type"] == "predicate"
    assert 0.0 <= predicate["probability"] <= 1.0

    assert choice["type"] == "choice"
    assert choice["choice"] in {"billing", "technical", "other"}
    assert [p["value"] for p in choice["probabilities"]] == ["billing", "technical", "other"]
    assert sum(p["probability"] for p in choice["probabilities"]) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= choice["confidence"] <= 1.0

    assert score["type"] == "score"
    assert score["probabilities"] == [
        {"label": "calm", "probability": pytest.approx(score["probabilities"][0]["probability"]), "value": 0},
        {"label": "frustrated", "probability": pytest.approx(score["probabilities"][1]["probability"]), "value": 1},
        {"label": "angry", "probability": pytest.approx(score["probabilities"][2]["probability"]), "value": 2},
    ]
    assert 0.0 <= score["score"] <= 2.0

    usage = body["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens_details"] == {"cache_write_tokens": 0, "cached_tokens": 0}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}


def test_questions_map_onto_systemone():
    native = to_systemone(DecisionCreateRequest.model_validate(DOCUMENTED))
    assert native.state == "I was charged twice for my order."
    assert list(native.questions) == ["is_urgent", "department", "frustration"]
    assert native.questions["is_urgent"] == NoulQuestion(instructions="Does this convey urgency?")
    assert native.questions["department"] == ChoiceQuestion(
        instructions="Which department should handle this complaint?",
        criteria={
            "billing": "Payments, invoices, and refunds.",
            "technical": None,
            "other": "Requests outside these categories.",
        },
    )
    assert native.questions["frustration"] == ScoreQuestion(
        instructions="How frustrated is the customer?",
        criteria=["calm", "frustrated — Annoyed but polite.", "angry"],
    )


def test_unnamed_questions_answer_with_null_names_in_order(client):
    body = client.post("/v1/decisions", json={
        "model": "m",
        "input": "x",
        "questions": [
            {"type": "predicate", "instructions": "a?"},
            {"type": "predicate", "name": "__question_0", "instructions": "b?"},
            {"type": "predicate", "instructions": "c?"},
        ],
    }).json()
    assert [a["name"] for a in body["answers"]] == [None, "__question_0", None]
    assert len(_RecordingEngine.last.questions) == 3  # a caller's name never shadows a generated key


def test_boolean_choice_values_come_back_as_booleans(client):
    body = client.post("/v1/decisions", json={
        "model": "m",
        "input": "x",
        "questions": [{"type": "choice", "name": "ok", "instructions": "ok?",
                       "choices": [{"value": True}, {"value": False}]}],
    }).json()
    (answer,) = body["answers"]
    assert answer["choice"] is True  # the stub puts its mass on the first option
    assert [p["value"] for p in answer["probabilities"]] == [True, False]


def test_message_input_joins_text_and_collects_images(client):
    image = "data:image/png;base64,iVBORw0KGgo="
    r = client.post("/v1/decisions", json={
        "model": "m",
        "input": [
            {"role": "user", "content": "first"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "second", "prompt_cache_breakpoint": {"mode": "explicit"}},
                {"type": "input_image", "image_url": image, "detail": "auto"},
            ]},
        ],
        "questions": [{"type": "predicate", "name": "q", "instructions": "y?"}],
        "prompt_cache_key": "k",
        "safety_identifier": "user-1",
    })
    assert r.status_code == 200, r.text
    assert _RecordingEngine.last.state == "first\n\nsecond"
    assert _RecordingEngine.last.images == [image]


@pytest.mark.parametrize(
    "change",
    [
        {"questions": []},
        {"questions": [{"type": "noul", "instructions": "System One's name, not OpenAI's"}]},
        {"questions": [{"type": "choice", "instructions": "x", "choices": [{"value": "a"}]}]},
        {"questions": [{"type": "choice", "instructions": "x", "choices": [{"value": "a"}, {"value": "a"}]}]},
        {"questions": [{"type": "predicate", "name": "q", "instructions": "x"}] * 2},
        {"input": [{"role": "system", "content": "x"}]},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}]},
    ],
)
def test_bad_requests_get_openai_errors(client, change):
    r = client.post("/v1/decisions", json={**DOCUMENTED, **change})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"]


def test_engine_rejections_get_openai_errors(client):
    """Errors only the engine (or the System One schema) can see are still a 400."""
    r = client.post("/v1/decisions", json={
        **DOCUMENTED,
        "questions": [{"type": "score", "instructions": "x", "levels": [{"label": "only one"}]}],
    })
    assert r.status_code == 400
    assert "levels" in r.json()["error"]["message"]

    r = client.post("/v1/decisions", json={
        **DOCUMENTED,
        "questions": [{"type": "choice", "instructions": "x",
                       "choices": [{"value": "true"}, {"value": True}]}],
    })
    assert r.status_code == 400
    assert "collide" in r.json()["error"]["message"]


def test_systemone_keeps_its_own_errors(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 422
    assert "detail" in r.json()


# ---- the real SDK -------------------------------------------------------------------


@pytest.fixture
def openai_client(client):
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(api_key="unused", base_url="http://testserver/v1", http_client=client)
    if not hasattr(sdk, "decisions"):
        pytest.skip("this openai SDK predates the Decisions API")
    return openai, sdk


def test_openai_sdk_parses_the_response(openai_client):
    _, sdk = openai_client
    decision = sdk.decisions.create(**DOCUMENTED)
    assert decision.model == "strands-decider-test"
    predicate, choice, score = decision.answers
    assert predicate.type == "predicate" and predicate.name == "is_urgent"
    assert choice.type == "choice" and choice.choice in {"billing", "technical", "other"}
    assert score.type == "score" and [p.value for p in score.probabilities] == [0, 1, 2]
    assert decision.usage.total_tokens == decision.usage.input_tokens + decision.usage.output_tokens


def test_openai_sdk_raises_bad_request(openai_client):
    openai, sdk = openai_client
    with pytest.raises(openai.BadRequestError):
        sdk.decisions.create(model="m", input="x", questions=[])
