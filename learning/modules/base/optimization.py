"""
Basic abstractions and default wiring for optimization in agent learning.

This module defines:
- OptimizationSignalGenerator: produces analysis signals from trajectories/components
- Optimization: consumes signals and applies concrete updates to a Candidate

Two optimization targets are supported:
  - PROMPT: update a single prompt variable inside prompt.py
  - AGENT:  optionally update prompt variables (set/rename/delete) and/or replace agent.py
"""

from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Any, Dict, List, Optional
import re
import yaml
import os

from .candidate import Candidate, ComponentType
from .selection import SelectionType
from base.engine.async_llm import AsyncLLM
from base.engine.utils import parse_xml_content, read_file_content, write_file_content
from base.engine.logs import logger_to_optimize

PROMPT_TRAGET_INSTRUCTION = """
<task>
  You will update ONLY the instruction prompt text.
  Output strictly using XML tags as follows:
  <prompt> ...your new prompt text... </prompt>

  Rules:
  - Do NOT include any extra commentary outside the tag.
  - Do NOT add code fences.
  - Keep the content safe for Python triple-quoted string embedding.
</task>
"""

AGENT_TARGET_INSTURCTION = """
You may modify the agent implementation and optionally adjust prompt variables.

Modify Instruction:
1. You can optimize both the prompt and agent code. 
2. When receive signal about the environment itself, you can modify, add new prompt to make your agent aware of environments. And if you add new prompt and want to use it in the agent, just from .prompt import <prompt_var_name>
3. You can add more llm calling to extend thinking process, such as ensemble to choose action, or add llm to compress memory.
4. The llm calling function is self.llm(prompt, max_tokens), and you can use the function parse_xml_content to parse the llm response (only you ask llm output with xml tags).
5. Make sure to understand the action parsing function.

Basic Class: 
Basic Class is BaseAgent, you can refer to it, but you can't modify BaseAgent

class BaseAgent(BaseAction, BaseModel):
    
    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    # Make LLM optional; concrete agents may initialize it from config.
    llm: Optional[AsyncLLM] = Field(default=None, description="Language model instance")

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    # Agent-As-An-Action
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        
    @abstractmethod
    async def step(self) -> str:
        pass
    
    @abstractmethod
    async def run(self, request: Optional[str] = None) -> str:
        pass

    async def __call__(self, **kwargs) -> Any:
        return await self.run(**kwargs)
    
    def to_param(self) -> Dict[str, Any]:
        return {
            "type": "agent-as-function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }



Output format:
1) A YAML block describing prompt updates (optional), and
2) A Python code block for the full agent (optional; omit to keep current agent code).

YAML block (fenced):
<agent_yaml>
type: agent_update
prompts:
  set:
    - name: LEARNED_INSTRUCTION_PROMPT   # variable name to create/update
      content: |-
        ... new prompt content ...
  rename:
    - old: OLD_VAR
      new: NEW_VAR
  delete:
    - UNUSED_VAR
</agent_yaml>

Python block (fenced):
<agent_code>
# full agent code here (replace agent.py). Omit block to keep existing code.
</agent_code>

Notes:
- If you change prompt variable names, ensure the provided agent code uses the new names.
- Omit any section you don't intend to change.
"""

OPTIMIZE_PROMPT = """
Given some optimization signal generated from agent trajectoris from a specific environment, your task is optimize the agent to get better performance.

<metric goal>
{metric_goal}
</metric goal>

<context>
Signals:
{signals_content}

Current component content:
{candidate_content}
</context>

<instruction>
{optimize_target}
</instruction>
"""

PERFORMANCE_TARGET_INSTRUCTION = """
Primary objective: maximize task performance (accuracy / total reward).
"""

PARETO_TARGET_INSTRUCTION = """
Objective: achieve a Pareto improvement on Accuracy vs. Cost.

Acceptable outcomes (priority order):
- Increase accuracy while reducing or maintaining cost.
- Maintain accuracy while reducing cost.
"""



class OptSignalMode(Enum):
    DYNAMICS = "dynamics"
    INSTRUCTION = "instruction"

class OptimizationSignal(BaseModel):
    signal_type: OptSignalMode
    content: str


class OptimizationSignalGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    Different Optimization Signal Method get different optimization signal from data.
    """
    analyze_prompt: str = Field(description="Prompt for analyzing the trajectory")
    analyze_llm: AsyncLLM = Field(description="LLM for analyzing the trajectory")

    async def generate_signal(self, candidate: Candidate, component_type: ComponentType) -> OptimizationSignal:
        trajectories = candidate.get_trajectories()
        component_content = candidate.get_component(component_type).strip()
        analyze_prompt = self.analyze_prompt.format(
            trajectories=trajectories,
            component_content=component_content,
        )
        resp = await self.analyze_llm(analyze_prompt, max_tokens=16384)
        logger_to_optimize(str(resp), os.path.join(candidate.candidate_folder, "optimize.log"))

        content = str(resp)
        # Default to INSTRUCTION if subclass does not specify
        mode = getattr(self, "signal_type", OptSignalMode.INSTRUCTION)
        return OptimizationSignal(signal_type=mode, content=content)



class Optimization(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    signal_generators: List[OptimizationSignalGenerator] = Field(default_factory=list)
    prompt_var_name: str = Field(default="LEARNED_INSTRUCTION_PROMPT")
    optimize_llm: Optional[AsyncLLM] = Field(default=None, description="LLM used to produce concrete updates from signals")
    # Initialization behavior
    initial_prompt: bool = Field(default=False, description="Use env agent_instruction.txt as initial prompt for round 1")
    env_folder_path: Optional[str] = Field(default=None, description="Env folder containing agent_instruction.txt")
    # Selection strategy wiring (affects metric goal guidance)
    selection_type: SelectionType = Field(default=SelectionType.CURRENT_BEST)

    async def optimize_candidate(
        self,
        candidate: Candidate,
        component_types: List[ComponentType],
    ) -> Candidate:
        # Initialize prompt once from env if requested
        try:
            if self.initial_prompt:
                existing = candidate.get_component(ComponentType.PROMPT)
                var = self.prompt_var_name
                has_triple = False
                has_none = False
                if existing:
                    has_triple = re.search(rf"^(\s*){re.escape(var)}\s*=\s*([\"\']{{3}})", existing, flags=re.M) is not None
                    has_none = re.search(rf"^(\s*){re.escape(var)}\s*=\s*None\b", existing, flags=re.M) is not None
                is_first = (getattr(candidate, "round", 0) <= 1) or (getattr(candidate, "parent", None) is None)
                need_seed = is_first or has_none or not has_triple
                if need_seed:
                    base = self.env_folder_path
                    if base:
                        inst_path = os.path.join(base, "agent_instruction.txt")
                        if os.path.isfile(inst_path):
                            with open(inst_path, "r", encoding="utf-8") as f:
                                init_prompt = f.read()
                            candidate.save_component(ComponentType.PROMPT, self.prompt_var_name, init_prompt)
        except Exception:
            pass
        # For each requested component type, generate signals and apply.
        for component_type in component_types:
            signals: List[OptimizationSignal] = []
            for gen in self.signal_generators:
                try:
                    sig = await gen.generate_signal(candidate, component_type)
                except Exception:
                    continue
                signals.append(sig)
            try:
                await self.leverage_siganl(signals, candidate, component_type)
            except Exception:
                continue
        return candidate


    async def leverage_siganl(
        self,
        signals: List[OptimizationSignal],
        candidate: Candidate,
        component_type: ComponentType,
    ) -> Candidate:
        # 1) Compose optimization instruction using signals
        signals_content = "\n".join(f"{sig.signal_type.value}: {sig.content}" for sig in signals)
        if component_type == ComponentType.AGENT:
            optimize_target = AGENT_TARGET_INSTURCTION
            candidate_content = (
                f"Agent Code:\n{candidate.get_component(ComponentType.AGENT)}\n\n"
                f"PROMPT:\n{candidate.get_component(ComponentType.PROMPT)}"
            )
        else:
            optimize_target = PROMPT_TRAGET_INSTRUCTION
            candidate_content = (
                f"PROMPT:\n{candidate.get_component(ComponentType.PROMPT)}\n"
                f"And You can only modify this prompt:{self.prompt_var_name}, "
                f"the AGENT_ACT_PROMPT is a default prompt, you can't modify it."
            )
        
        # Inject metric-goal guidance according to selection strategy
        try:
            st = getattr(self, "selection_type", SelectionType.CURRENT_BEST)
        except Exception:
            st = SelectionType.CURRENT_BEST
        metric_goal = PARETO_TARGET_INSTRUCTION if st == SelectionType.PARETO_FRONT else PERFORMANCE_TARGET_INSTRUCTION

        optimize_prompt = OPTIMIZE_PROMPT.format(
            signals_content=signals_content,
            candidate_content=candidate_content,
            metric_goal=metric_goal,
            optimize_target=optimize_target,
        )

        # 2) Ask LLM to produce update payload
        if self.optimize_llm is None:
            # No-op if no optimize llm is provided
            return candidate
        resp = await self.optimize_llm(optimize_prompt, max_tokens=20000)
        # Minimal optimization logging: record raw LLM response only
        try:
            logger_to_optimize(str(resp), os.path.join(candidate.candidate_folder, "optimize.log"))
        except Exception:
            pass

        # 3) Apply updates depending on target type
        if component_type == ComponentType.PROMPT:
            self._apply_prompt_update_from_xml(candidate, resp)
            return candidate

        if component_type == ComponentType.AGENT:
            self._apply_agent_update_from_yaml_and_code(candidate, resp)
            # Keep optimization lightweight. Any fix attempts will be done after smoke test in runner.
            return candidate

        # Unsupported types: no-op
        return candidate

    # ---------------- internal helpers ----------------
    def _apply_prompt_update_from_xml(self, candidate: Candidate, text: str) -> None:
        """Extract <prompt>...</prompt> and write to configured variable name."""
        data = parse_xml_content(text, "prompt")
        content = None
        if isinstance(data, dict):
            v = data.get("prompt")
            if isinstance(v, list):
                content = "\n\n".join([str(x) for x in v])
            else:
                content = v
        if not content:
            # Fallback: use the raw text if parsing failed
            content = text.strip()
        candidate.save_component(ComponentType.PROMPT, self.prompt_var_name, str(content))

    def _apply_agent_update_from_yaml_and_code(self, candidate: Candidate, text: str) -> None:
        """
        Parse a YAML directive (prompt edits) and an optional Python code block (agent replacement).
        """
        # Extract YAML update (if provided)
        yaml_directive: Optional[Dict[str, Any]] = None
        yaml_data = parse_xml_content(text, "agent_yaml")
        if isinstance(yaml_data, dict) and "agent_yaml" in yaml_data:
            try:
                yaml_directive = yaml.safe_load(yaml_data["agent_yaml"]) or {}
            except Exception:
                yaml_directive = None

        # Apply prompt directive if present
        if isinstance(yaml_directive, dict):
            prompts = yaml_directive.get("prompts") or {}
            # rename first (to keep content for set phase if same names conflict)
            for item in prompts.get("rename", []) or []:
                try:
                    self._rename_prompt_variable(candidate, str(item.get("old")), str(item.get("new")))
                except Exception:
                    continue
            # delete
            for name in prompts.get("delete", []) or []:
                try:
                    self._delete_prompt_variable(candidate, str(name))
                except Exception:
                    continue
            # set/create
            for item in prompts.get("set", []) or []:
                try:
                    name = str(item.get("name"))
                    content = str(item.get("content", ""))
                    candidate.save_component(ComponentType.PROMPT, name, content)
                except Exception:
                    continue

        # Extract/replace agent code if provided
        code_data = parse_xml_content(text, "agent_code")
        if isinstance(code_data, dict) and "agent_code" in code_data:
            code = code_data["agent_code"]
            if code and isinstance(code, str) and code.strip():
                candidate.save_component(ComponentType.AGENT, code)

    def _prompt_file_path(self, candidate: Candidate) -> Optional[str]:
        try:
            return candidate.components_path.get(ComponentType.PROMPT)  # type: ignore[return-value]
        except Exception:
            return None

    def _rename_prompt_variable(self, candidate: Candidate, old: str, new: str) -> None:
        """Rename a top-level triple-quoted assignment variable in prompt.py."""
        path = self._prompt_file_path(candidate)
        if not path or not old or not new or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new):
            return
        try:
            src = read_file_content(path)
        except Exception:
            return
        pattern = rf"^(\s*){re.escape(old)}\s*=\s*([\"\']{{3}})([\s\S]*?)\2"  # multiline, triple-quote block
        def repl(m: re.Match) -> str:
            indent, quote, body = m.group(1), m.group(2), m.group(3)
            return f"{indent}{new} = {quote}{body}{quote}"
        new_src, n = re.subn(pattern, repl, src, flags=re.M)
        if n > 0:
            write_file_content(path, new_src)

    def _delete_prompt_variable(self, candidate: Candidate, name: str) -> None:
        """Delete a top-level triple-quoted assignment variable from prompt.py."""
        path = self._prompt_file_path(candidate)
        if not path or not name:
            return
        try:
            src = read_file_content(path)
        except Exception:
            return
        pattern = rf"^(\s*){re.escape(name)}\s*=\s*([\"\']{{3}})[\s\S]*?\2\s*\n?"  # remove the whole block
        new_src, n = re.subn(pattern, "", src, flags=re.M)
        if n > 0:
            write_file_content(path, new_src)
