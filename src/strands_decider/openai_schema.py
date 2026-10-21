"""Wire types for the OpenAI Decisions API, and their mapping onto System One.

Follows the request and response shape of OpenAI's `POST /v1/decisions` (the
`client.decisions.create(...)` method of the `openai` SDKs), so an OpenAI client can be
repointed at this server by changing its base URL:

    POST /v1/decisions
    {"model": "...", "input": "...", "questions": [{"type": "predicate", "name": "...", ...}]}

The two APIs ask the same three kinds of question under different names:

    OpenAI     System One   answer
    predicate  noul         P(true)
    choice     choice       one of N values + per-value probabilities
    score      score        expected level index + per-level probabilities

The differences are in shape, not meaning. OpenAI lists questions in order and names them
optionally; System One keys them by name. OpenAI choice values may be strings or booleans;
System One option names are strings. OpenAI levels carry a label and a description;
System One levels are one rubric line each. The functions below translate each way and
keep the caller's values, names and order intact.

Fields this server has no use for (`prompt_cache_key`, `prompt_cache_options`,
`safety_identifier`, an input part's `detail` and `prompt_cache_breakpoint`) are accepted
and ignored: the engine caches the shared state on its own, per request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictStr, field_validator, model_validator

from .schema import (
    MAX_CHOICE_OPTIONS,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

MAX_QUESTIONS = 200
MAX_INPUT_IMAGES = 128

ChoiceValue = StrictStr | StrictBool

# ---- request ------------------------------------------------------------------------


class PromptCacheBreakpoint(BaseModel):
    mode: Literal["explicit"]


class InputText(BaseModel):
    type: Literal["input_text"]
    text: str
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


class InputImage(BaseModel):
    type: Literal["input_image"]
    # OpenAI also takes a public http(s) URL. This server does not fetch URLs on a
    # caller's behalf (that would let any caller make it request arbitrary hosts), so
    # only inline `data:image/...;base64,` URLs are accepted.
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None

    @field_validator("image_url")
    @classmethod
    def _inline_only(cls, v: str) -> str:
        if not v.strip().startswith("data:image/"):
            raise ValueError(
                "image_url must be a base64 data URL (data:image/...;base64,...); "
                "this server does not fetch remote images"
            )
        return v


InputPart = InputText | InputImage


class InputMessage(BaseModel):
    role: Literal["user"]
    content: str | list[InputPart]
    type: Literal["message"] = "message"


class PredicateQuestion(BaseModel):
    """Estimate how likely it is that a statement about the input is true."""

    type: Literal["predicate"]
    instructions: str
    name: str | None = None


class Choice(BaseModel):
    # A string and a boolean with the same text are distinct values.
    value: ChoiceValue
    description: str | None = None


class OpenAIChoiceQuestion(BaseModel):
    """Choose from the supplied options based on the input."""

    type: Literal["choice"]
    instructions: str
    choices: list[Choice] = Field(..., min_length=2, max_length=MAX_CHOICE_OPTIONS)
    name: str | None = None

    @field_validator("choices")
    @classmethod
    def _unique(cls, v: list[Choice]) -> list[Choice]:
        seen = [(type(c.value), c.value) for c in v]
        if len(set(seen)) != len(seen):
            raise ValueError("choice values must be unique")
        return v


class Level(BaseModel):
    label: str
    description: str | None = None


class OpenAIScoreQuestion(BaseModel):
    """Rate the input against the supplied ordered levels, lowest first."""

    type: Literal["score"]
    instructions: str
    levels: list[Level]
    name: str | None = None


OpenAIQuestion = PredicateQuestion | OpenAIChoiceQuestion | OpenAIScoreQuestion


class PromptCacheOptions(BaseModel):
    mode: Literal["implicit", "explicit"] | None = None
    ttl: Literal["30m"] | None = None


class DecisionCreateRequest(BaseModel):
    model: str
    input: str | list[InputMessage]
    questions: list[OpenAIQuestion] = Field(..., min_length=1, max_length=MAX_QUESTIONS)
    prompt_cache_key: str | None = None
    prompt_cache_options: PromptCacheOptions | None = None
    safety_identifier: str | None = None

    @model_validator(mode="after")
    def _unique_names(self) -> DecisionCreateRequest:
        names = [q.name for q in self.questions if q.name is not None]
        if len(set(names)) != len(names):
            raise ValueError("question names must be unique")
        return self


# ---- response -----------------------------------------------------------------------


class PredicateAnswer(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str | None
    probability: float


class ChoiceProbability(BaseModel):
    value: ChoiceValue
    probability: float


class OpenAIChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    name: str | None
    choice: ChoiceValue
    confidence: float
    probabilities: list[ChoiceProbability]


class ScoreProbability(BaseModel):
    label: str
    probability: float
    value: int  # 0-based level index


class OpenAIScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    name: str | None
    score: float
    confidence: float
    probabilities: list[ScoreProbability]


class RefusalAnswer(BaseModel):
    """The model declined this question. Never produced here: a decision model always
    answers, so a low confidence is how it says it does not know. Listed so the wire
    union matches OpenAI's."""

    type: Literal["refusal"] = "refusal"
    name: str | None


DecisionAnswer = PredicateAnswer | OpenAIChoiceAnswer | OpenAIScoreAnswer | RefusalAnswer


class InputTokensDetails(BaseModel):
    cache_write_tokens: int = 0
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class DecisionUsage(BaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int
    output_tokens_details: OutputTokensDetails = Field(default_factory=OutputTokensDetails)
    total_tokens: int


class Decision(BaseModel):
    answers: list[DecisionAnswer]
    model: str
    usage: DecisionUsage


# ---- translation --------------------------------------------------------------------


def _choice_key(value: str | bool) -> str:
    """System One option name for an OpenAI choice value. Booleans read as JSON does."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _level_rubric(level: Level) -> str:
    return f"{level.label} — {level.description}" if level.description else level.label


def _to_systemone_question(question: OpenAIQuestion) -> Question:
    if isinstance(question, PredicateQuestion):
        return NoulQuestion(instructions=question.instructions)
    if isinstance(question, OpenAIChoiceQuestion):
        keys = [_choice_key(c.value) for c in question.choices]
        if len(set(keys)) != len(keys):
            # e.g. "true" and true in one question: distinct to OpenAI, one name to us.
            raise ValueError(
                "choice values collide once rendered as text (a string and a boolean "
                "with the same text); rename one of them"
            )
        return ChoiceQuestion(
            instructions=question.instructions,
            criteria={k: c.description for k, c in zip(keys, question.choices, strict=True)},
        )
    return ScoreQuestion(
        instructions=question.instructions,
        criteria=[_level_rubric(level) for level in question.levels],
    )


def _split_input(input_: str | list[InputMessage]) -> tuple[str, list[str]]:
    """The input as one System One state plus its images, in the order they were sent."""
    if isinstance(input_, str):
        return input_, []
    texts: list[str] = []
    images: list[str] = []
    for message in input_:
        if isinstance(message.content, str):
            texts.append(message.content)
            continue
        for part in message.content:
            if isinstance(part, InputText):
                texts.append(part.text)
            else:
                images.append(part.image_url)
    if len(images) > MAX_INPUT_IMAGES:
        raise ValueError(f"{len(images)} images; at most {MAX_INPUT_IMAGES} are allowed")
    return "\n\n".join(texts), images


def _keys(request: DecisionCreateRequest) -> list[str]:
    """A System One key per question. Unnamed questions get a positional key that cannot
    collide with a caller's name."""
    taken = {q.name for q in request.questions if q.name is not None}
    keys: list[str] = []
    for i, q in enumerate(request.questions):
        if q.name is not None:
            keys.append(q.name)
            continue
        key = f"__question_{i}"
        while key in taken:
            key = f"_{key}"
        keys.append(key)
    return keys


def to_systemone(request: DecisionCreateRequest) -> SystemOneRequest:
    """An OpenAI decision request as the System One request the engine evaluates."""
    state, images = _split_input(request.input)
    return SystemOneRequest(
        state=state,
        model=request.model,
        questions={
            key: _to_systemone_question(q)
            for key, q in zip(_keys(request), request.questions, strict=True)
        },
        images=images,
    )


def _to_decision_answer(question: OpenAIQuestion, answer: object) -> DecisionAnswer:
    name = question.name
    if isinstance(question, PredicateQuestion) and isinstance(answer, NoulAnswer):
        return PredicateAnswer(name=name, probability=answer.noul)
    if isinstance(question, OpenAIChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        # Back from option names to the caller's values, in the order they were listed.
        values = {_choice_key(c.value): c.value for c in question.choices}
        return OpenAIChoiceAnswer(
            name=name,
            choice=values[answer.choice],
            confidence=answer.confidence,
            probabilities=[
                ChoiceProbability(value=values[k], probability=answer.probabilities[k])
                for k in values
            ],
        )
    if isinstance(question, OpenAIScoreQuestion) and isinstance(answer, ScoreAnswer):
        return OpenAIScoreAnswer(
            name=name,
            score=answer.score,
            confidence=answer.confidence,
            probabilities=[
                ScoreProbability(
                    label=level.label, probability=answer.probabilities[str(i)], value=i
                )
                for i, level in enumerate(question.levels)
            ],
        )
    raise TypeError(  # pragma: no cover - the engine answers each question in its own kind
        f"{type(answer).__name__} does not answer a {question.type} question"
    )


def from_systemone(request: DecisionCreateRequest, response: SystemOneResponse) -> Decision:
    """The engine's answers as an OpenAI decision, in question order."""
    answers = [
        _to_decision_answer(q, response.answers[key])
        for key, q in zip(_keys(request), request.questions, strict=True)
    ]
    usage = response.usage
    return Decision(
        answers=answers,
        model=response.model,
        usage=DecisionUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        ),
    )


def error_body(message: str, *, param: str | None = None) -> dict[str, object]:
    """OpenAI's error envelope, so the SDKs raise their usual `BadRequestError`."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": None,
        }
    }
