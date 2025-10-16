"""
Signal prompt templates for trajectory-driven analysis (plain text output).

These prompts ask for concise normal text. The raw text becomes the signal content
ingested by the optimization stage. Do NOT require XML or YAML in the output.
"""

# 1) Dynamics-focused analysis
# Goal: infer environment rules, rewards, transitions, hazards from trajectories.
DYNAMICS_OPTIMIZATION_PROMPT = """
You are an expert at reverse-engineering environment dynamics from agent trajectories.

Input trajectories (human-readable):
{trajectories}

Optional current component (prompt or agent code excerpt):
{component_content}

Write a concise analysis (plain text) that covers:
- Environment: key state variables, observations, and action space the agent appears to have.
- Transitions: common preconditions → effects; typical progress vs. dead-ends; termination cues.
- Rewards: which actions/events correlate with reward changes; signs of sparse/dense reward.
- Failures: frequent mistakes and likely causes, with brief evidence from the trajectories.
- Strategies: practical heuristics/rules to increase reward and reduce mistakes.
- Uncertainties: what remains unclear and what evidence would disambiguate it.
- Confidence: your overall confidence (0.0–1.0).
"""


# 2) Instruction-focused analysis
# Goal: diagnose why the agent underperforms and propose instruction changes.
INSTRUCTION_OPTIMIZATION_PROMPT = """
You evaluate agent trajectories to improve the agent's instruction/policy prompt.

Input trajectories (human-readable):
{trajectories}

Optional current instruction/code excerpt:
{component_content}

Write a concise diagnosis and proposal (plain text) that covers:
- Diagnosis: concrete failure patterns (perception, action choice, planning, termination misuse, etc.).
- Principles: short, general rules the agent should follow (imperative and checkable).
- Step Guidelines: when-then style rules for common situations.
- Guardrails: behaviors the agent must avoid, with conditions.
- Mini Examples (optional): tiny templates that illustrate correct handling.
- Measurement: how success should be measured and expected direction of change.
- Confidence: your overall confidence (0.0–1.0).
"""
