"""Catch a tool call the agent *shouldn't* make yet, and send it back to ask first.

**Experimental.** This is a demo -- a sample of the kind of judgement you can put in this
seam, written to be read rather than deployed. The questions, the thresholds and the policy
are all illustrative: choose your own, and fit them to your own traffic. None of it is
recommended for production as it stands.

This gates a tool call on something subtle: whether the call is even well-formed given
the conversation so far. The agent proposes a tool call; the strands-decider model answers
two narrow yes/no questions about it -- are the arguments grounded in what the user
actually said, and is the call premature -- and a plain-Python policy turns those
probabilities into `Guide` (send the model back with feedback) or `Proceed` (let it run).

The seam is `InterventionHandler.before_tool_call`, which runs before Strands executes the
tool. The verdict is what makes this different from a hard allow/deny gate: `Guide` does
not refuse the call, it *corrects the course*, the model gets the feedback and takes
another turn, typically asking the user the question it should have asked in the first
place.

    strands-decider serve <checkpoint> --port 8099
    python examples/strands/tool_call_intervention.py

The scenario is deliberately small: one `get_weather` tool and one incomplete request
("What's the weather?" with no city). The agent is given an eager system prompt so it
guesses a location instead of asking, exactly the failure mode the gate is meant to
catch. strands-decider flags the guessed argument as ungrounded, and the policy returns a
`Guide` that sends the model back to ask the user which city they meant.

Why classify instead of letting the agent self-check? The check is cheap and it
has to be honest, a general chat model asked "are your own arguments grounded?" is possible
to talk out of a no. The gate is a local forward pass in a couple of hundred
milliseconds, the conversation never leaves the machine, and each verdict is a number your
`if` statements branch on rather than prose the agent negotiates with.

Needs AWS credentials for Bedrock to drive the agent. The gate needs only the local
server, so swap `model=` for any Strands model provider and the strands-decider half is
unchanged. STRANDS_DECIDER_URL overrides the endpoint (default http://127.0.0.1:8099).
"""

from __future__ import annotations

import json
import sys

from _client import NAME, Decider, ServerUnavailable
from strands import Agent, tool
from strands.interventions import Guide, InterventionHandler, Proceed

# --------------------------------------------------------------------------- the questions
#
# Two narrow yes/no (predicate) questions, each a single judgement: one about the arguments, one
# about the timing. strands-decider answers each with P(true) in 0..1; it never decides the
# action. The `true`/`false` descriptions carry most of the signal, so they are worth
# writing as carefully as the questions themselves.

QUESTIONS = {
    "args_grounded": Decider.predicate(
        "Are the tool's argument values grounded in facts the user actually provided?",
        {
            "true": "every argument value traces back to something the user said",
            "false": "an argument value was guessed or invented, not stated by the user",
        },
    ),
    "premature": Decider.predicate(
        "Is it premature to call this tool now, before clarifying with the user?",
        {
            "true": "the assistant should ask a clarifying question before calling the tool",
            "false": "there is nothing left to clarify; calling now is appropriate",
        },
    ),
}

# A strands-decider probability at or above this reads as a confident "yes". This is the
# policy knob, owned by the code, not the model. Raise it to block less, lower it to block
# more. 0.45 is illustrative -- pick a value that suits your own traffic.
YES = 0.45


# --------------------------------------------------------------------------- the gate


class ToolCallReviewer(InterventionHandler):
    """Review each proposed tool call before it runs, using strands-decider.

    The conversation so far plus the proposed call go to the server as one state; the four
    questions come back under their own names. The policy below is ordinary control flow
    over those numbers -- that separation is the whole point: the model classifies, the
    code decides.
    """

    name = f"{NAME}-tool-call-reviewer"

    def __init__(self, decider: Decider) -> None:
        self._decider = decider
        self.log: list[dict[str, object]] = []

    @staticmethod
    def _render(messages: list[dict]) -> str:
        """Flatten Strands' message list to plain text: only the human-readable blocks."""
        lines: list[str] = []
        for message in messages:
            role = message.get("role", "unknown")
            for block in message.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    lines.append(f"{role}: {block['text']}")
        return "\n".join(lines) if lines else "(no conversation yet)"

    def before_tool_call(self, event, **kwargs):
        name = event.tool_use["name"]
        arguments = event.tool_use.get("input") or {}
        state = (
            "A conversation between a user and an AI assistant is below, followed by a "
            "tool call the assistant now wants to make.\n\n"
            f"--- CONVERSATION ---\n{self._render(event.agent.messages)}\n\n"
            f"--- PROPOSED TOOL CALL ---\ntool: {name}\n"
            f"arguments: {json.dumps(arguments)}"
        )

        answers = self._decider.ask(state, QUESTIONS)
        args_grounded = answers["args_grounded"]["probability"]
        premature = answers["premature"]["probability"]

        print("[intervention] classifications (probability of 'yes'):")
        print(f"    args_grounded  : {args_grounded:.2f}")
        print(f"    premature      : {premature:.2f}")

        # Ordinary control flow over typed numbers. Each branch maps one classification
        # pattern to a typed action; the first that fires wins.
        if args_grounded < YES:
            verdict, action = "guided", Guide(
                feedback=(
                    "The arguments are not grounded in anything the user said -- they look "
                    "guessed. Ask the user to confirm the values instead of inventing them."
                )
            )
        elif premature >= YES:
            verdict, action = "guided", Guide(
                feedback=(
                    "It is too early to call this tool. Clarify with the user first, then "
                    "try again."
                )
            )
        else:
            verdict, action = "proceeded", Proceed()

        self.log.append(
            {"tool": name, "args_grounded": round(args_grounded, 2),
             "premature": round(premature, 2), "verdict": verdict}
        )
        arrow = {"proceeded": "->", "guided": "~>"}[verdict]
        print(f"    {arrow} {verdict}\n")
        return action


# --------------------------------------------------------------------------- the tool
#
# One tool, fake data. The point is not the weather -- it is that the gate catches a call
# with a guessed argument before it ever runs.


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific city.

    Args:
        location: The city to look up, e.g. "Seattle" or "Paris".
    """
    return f"It's 21C and sunny in {location}."


# An eager system prompt models the failure mode: an agent that guesses rather than asks.
# It is instructed to assume Seattle when no city is given, so it proposes get_weather with
# an ungrounded argument -- which is exactly what the gate is there to catch.
SYSTEM_PROMPT = (
    "You are an eager weather assistant. Always answer weather questions by calling the "
    "get_weather tool immediately. Never ask the user for clarification -- if no city is "
    "given, just assume Seattle and call the tool with location='Seattle'."
)

# Deliberately incomplete: no city. The eager model guesses "Seattle"; strands-decider
# flags the argument as ungrounded, and the policy returns a Guide telling it to ask first.
USER_REQUEST = "What's the weather?"


def main() -> int:
    decider = Decider()
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    gate = ToolCallReviewer(decider)
    agent = Agent(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tools=[get_weather],
        interventions=[gate],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,  # the gate's own log is the interesting output
    )

    print(f"the agent proposes each tool call; {NAME} decides whether it is ready to run\n")
    print(f"USER: {USER_REQUEST}\n")
    result = agent(USER_REQUEST)

    guided = sum(1 for row in gate.log if row["verdict"] == "guided")
    print(f"{decider}; guided {guided} of {len(gate.log)} proposed calls back for clarification")
    print(f"\n--- what the agent finally replied ---\n{result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
