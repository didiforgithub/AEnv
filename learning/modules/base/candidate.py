
from __future__ import annotations

import json
import os
import shutil
import re
from enum import Enum
from typing import Dict, Optional, List, Tuple, Union

from pydantic import BaseModel, Field

from base.engine.utils import read_file_content, write_file_content
from .evaluation import EvaluationMetric


class ComponentType(Enum):
    """
    Coarse-grained component types.
    """
    PROMPT = "prompt"  # optimize variables inside prompt.py
    AGENT = "agent"    # replace whole agent.py with provided content
    MODEL = "model"    # reserved


class  Candidate(BaseModel):
    round: int = Field(description="Round of the candidate")
    parent: Optional[int] = Field(default=None, description="Parent of the candidate")
    metrics: Dict[EvaluationMetric, float] = Field(default_factory=dict, description="Metrics of the candidate")
    candidate_folder: str = Field(description="Folder path of the candidate")
    basic_info_path: str = Field(description="Path to basic info json")
    components_path: Dict[ComponentType, str] = Field(default_factory=dict, description="Component file paths")
    trajectory_path: str = Field(description="Path to trajectories json")

    # ---------- Component access ----------
    def get_component(self, component_type: ComponentType) -> str:
        path = self.components_path.get(component_type)
        if not path or not os.path.isfile(path):
            return ""
        return read_file_content(path)

    # ---------- Trajectory access -----------
    def get_trajectories(self) -> str:
        """
        Render stored trajectories.json as a human-readable English summary.

        Format:
          Trajectory 1 | World: <world_id> | Agent: <agent>
          - Total Reward: <total_reward> | Steps: <n>
          - Initial Observation: <short>
            Step 1:
              - Thought: <short>
              - Action: <short>
              - Observation: <short>
              - Result: <short>
              - Reward: <r>  Events: e1, e2
          (blank line between trajectories)
        """
        import json
        from typing import Any

        def _short(v: Any, limit: int = 240) -> str:
            try:
                if isinstance(v, (dict, list)):
                    s = json.dumps(v, ensure_ascii=False)
                else:
                    s = str(v)
            except Exception:
                s = str(v)
            s = s.strip()
            return (s if len(s) <= limit else s[: limit] + "…") if s else "(empty)"

        if not os.path.isfile(self.trajectory_path or ""):
            return f"Trajectory file not found: {self.trajectory_path}"

        try:
            with open(self.trajectory_path, "r", encoding="utf-8") as f:
                traj_list = json.load(f) or []
        except Exception as e:
            return f"Failed to read trajectories: {e}"

        if not isinstance(traj_list, list) or not traj_list:
            return "No trajectories"

        lines: list[str] = []
        for idx, t in enumerate(traj_list, 1):
            world_id = t.get("world_id")
            agent_name = t.get("agent_name", "agent")
            total_reward = t.get("total_reward", 0)
            steps = t.get("steps", []) or []
            meta = t.get("metadata", {}) or {}
            init_obs = meta.get("initial_observation")

            lines.append(f"Trajectory {idx} | World: {world_id} | Agent: {agent_name}")
            lines.append(f"- Total Reward: {total_reward} | Steps: {len(steps)}")
            if init_obs is not None:
                lines.append(f"- Initial Observation: {_short(init_obs)}")

            for j, s in enumerate(steps, 1):
                thought = s.get("thinking")
                action = s.get("action", {})
                obs = s.get("obs")
                result = s.get("result")
                reward = s.get("reward", 0)
                events = s.get("events", []) or []

                # Prefer concise action if present
                action_str = None
                if isinstance(action, dict):
                    action_str = action.get("action")
                if not action_str:
                    action_str = _short(action, limit=160)

                lines.append(f"  Step {j}:")
                if thought is not None:
                    lines.append(f"    - Thought: {_short(thought)}")
                lines.append(f"    - Action: {action_str}")
                if obs is not None:
                    lines.append(f"    - Observation: {_short(obs)}")
                if result is not None:
                    lines.append(f"    - Result: {_short(result)}")
                lines.append(f"    - Reward: {reward}  Events: {', '.join(map(str, events)) if events else 'None'}")

            # blank line between trajectories
            lines.append("")

        return "\n".join(lines).rstrip()

    # ---------- Prompt editing (variable write-in) ----------
    def _update_prompt_variable(self, prompt_path: str, var_name: str, content: str) -> None:
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        existing = ""
        if os.path.isfile(prompt_path):
            try:
                existing = read_file_content(prompt_path)
            except Exception:
                existing = ""

        # triple-quote safe content
        safe = content.replace('"""', '\\"\\"\\"')
        new_block = f'{var_name} = """\n{safe}\n"""'

        if existing:
            # Replace if variable exists; else append
            pattern_triple = rf"{re.escape(var_name)}\s*=\s*([\"\']{{3}})([\s\S]*?)\1"
            pattern_none = rf"{re.escape(var_name)}\s*=\s*None\b"
            if re.search(pattern_triple, existing, flags=re.S):
                updated = re.sub(pattern_triple, new_block, existing, flags=re.S)
            elif re.search(pattern_none, existing, flags=re.S):
                updated = re.sub(pattern_none, new_block, existing, flags=re.S)
            else:
                updated = existing.rstrip() + "\n\n" + new_block + "\n"
        else:
            updated = new_block + "\n"

        write_file_content(prompt_path, updated)

    # ---------- Agent editing (full replace) ----------
    def _replace_agent_file(self, agent_path: str, new_content: str) -> None:
        os.makedirs(os.path.dirname(agent_path), exist_ok=True)
        write_file_content(agent_path, new_content)

    def save_component(self, component_type: ComponentType, *params):
        """
        Save/modify a specific component.

        - PROMPT: params = (var_name: str, content: str)
            Update or append a triple-quoted variable assignment in prompt.py
        - AGENT: params = (content_str: str,)           -> replace whole file

        basic_info.json holds round/parent/metrics. Component files live under candidate_folder.
        trajectories.json records all trajectories path (not written here).
        """
        if component_type not in self.components_path:
            # Pre-assign default paths under candidate folder
            if component_type == ComponentType.PROMPT:
                self.components_path[component_type] = os.path.join(self.candidate_folder, "prompt.py")
            elif component_type == ComponentType.AGENT:
                self.components_path[component_type] = os.path.join(self.candidate_folder, "agent.py")

        if component_type == ComponentType.PROMPT:
            assert len(params) == 2, "PROMPT update requires (var_name, content)"
            var_name, content = params[0], params[1]
            if not isinstance(var_name, str):
                raise TypeError("var_name must be str")
            self._update_prompt_variable(self.components_path[ComponentType.PROMPT], var_name, str(content))
            return

        if component_type == ComponentType.AGENT:
            assert len(params) == 1, "AGENT update requires exactly one parameter"
            payload = params[0]
            path = self.components_path[ComponentType.AGENT]
            if not isinstance(payload, str):
                raise TypeError("Unsupported AGENT payload; use str to replace the whole file")
            # Replace with full text
            self._replace_agent_file(path, payload)
            return

        raise NotImplementedError(f"Unsupported component type: {component_type}")

    # ---------- Basic info IO ----------
    def save_basic_info(self) -> None:
        os.makedirs(os.path.dirname(self.basic_info_path), exist_ok=True)
        metrics_json = {k.value if isinstance(k, EvaluationMetric) else str(k): float(v) for k, v in self.metrics.items()}
        data = {
            "round": self.round,
            "parent": self.parent,
            "metrics": metrics_json,
            "components": {ct.value: self.components_path.get(ct) for ct in self.components_path},
            "trajectory_path": self.trajectory_path,
        }
        with open(self.basic_info_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_folder(cls, folder: str) -> Candidate:
        basic = os.path.join(folder, "basic_info.json")
        # Fallback compatibility paths
        metrics_json: Dict[str, float] = {}
        parent = None
        traj_path = os.path.join(folder, "trajectories.json")
        if os.path.isfile(basic):
            with open(basic, "r", encoding="utf-8") as f:
                info = json.load(f)
            round_id = int(info.get("round", 0))
            parent = info.get("parent")
            metrics_json = info.get("metrics", {}) or {}
            traj_path = info.get("trajectory_path", traj_path)
            comp_map_raw = info.get("components", {})
        else:
            # Try legacy files
            name = os.path.basename(folder).split("_")[-1]
            round_id = int(name) if name.isdigit() else 0
            comp_map_raw = {}
            mfile = os.path.join(folder, "metrics.json")
            if os.path.isfile(mfile):
                try:
                    with open(mfile, "r", encoding="utf-8") as f:
                        metrics_json = json.load(f)
                except Exception:
                    metrics_json = {}

        # Build components map
        comp_map: Dict[ComponentType, str] = {}
        p = comp_map_raw.get("prompt") or os.path.join(folder, "prompt.py")
        a = comp_map_raw.get("agent") or os.path.join(folder, "agent.py")
        if os.path.isfile(p):
            comp_map[ComponentType.PROMPT] = p
        if os.path.isfile(a):
            comp_map[ComponentType.AGENT] = a

        # Cast metrics keys back to enum
        metrics: Dict[EvaluationMetric, float] = {}
        for k, v in (metrics_json or {}).items():
            try:
                metrics[EvaluationMetric(k)] = float(v)
            except Exception:
                continue

        return cls(
            round=round_id,
            parent=parent,
            metrics=metrics,
            candidate_folder=folder,
            basic_info_path=basic,
            components_path=comp_map,
            trajectory_path=traj_path,
        )


class CandidateManager(BaseModel):
    candidates: Dict[int, Candidate] = Field(default_factory=dict, description="rounds and candidates")
    workspace_path: str = Field(description="The path of workspace")
    seed_file_path: str = Field(description="The path of seed files (dir with agent.py/prompt.py)")

    def _candidate_dir(self, round_id: int) -> str:
        return os.path.join(self.workspace_path, "candidates", f"candidate_{round_id}")

    def _copy_seed_into(self, target_dir: str) -> None:
        os.makedirs(target_dir, exist_ok=True)
        for name in ("agent.py", "prompt.py"):
            src = os.path.join(self.seed_file_path, name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(target_dir, name))

    def _copy_parent_into(self, parent_dir: str, target_dir: str) -> None:
        os.makedirs(target_dir, exist_ok=True)
        for name in ("agent.py", "prompt.py"):
            src = os.path.join(parent_dir, name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(target_dir, name))

    def _load(self) -> None:
        """
        Load existing candidates from workspace. If none, bootstrap round 1 by copying seed files.
        """
        root = os.path.join(self.workspace_path, "candidates")
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                if not name.startswith("candidate_"):
                    continue
                try:
                    round_id = int(name.split("_")[-1])
                except ValueError:
                    continue
                folder = os.path.join(root, name)
                cand = Candidate.load_from_folder(folder)
                self.candidates[round_id] = cand

        if not self.candidates:
            # Bootstrap round 1
            r1_dir = self._candidate_dir(1)
            self._copy_seed_into(r1_dir)
            cand = Candidate(
                round=1,
                parent=None,
                metrics={},
                candidate_folder=r1_dir,
                basic_info_path=os.path.join(r1_dir, "basic_info.json"),
                components_path={
                    ComponentType.PROMPT: os.path.join(r1_dir, "prompt.py"),
                    ComponentType.AGENT: os.path.join(r1_dir, "agent.py"),
                },
                trajectory_path=os.path.join(r1_dir, "trajectories.json"),
            )
            cand.save_basic_info()
            self.candidates[1] = cand

    def create_candidate(self, round_id: int, parent_round: Optional[int] = None) -> Candidate:
        """
        Create a new candidate folder by copying from seed (if first) or from parent.
        Returns the Candidate object.
        """
        target = self._candidate_dir(round_id)
        if parent_round is None and not self.candidates:
            # first candidate -> seed
            self._copy_seed_into(target)
        elif parent_round is not None and parent_round in self.candidates:
            self._copy_parent_into(self.candidates[parent_round].candidate_folder, target)
        else:
            # No parent provided; default to last candidate as base if exists else seed
            if self.candidates:
                base_parent = self.candidates[max(self.candidates.keys())]
                self._copy_parent_into(base_parent.candidate_folder, target)
            else:
                self._copy_seed_into(target)

        cand = Candidate(
            round=round_id,
            parent=parent_round,
            metrics={},
            candidate_folder=target,
            basic_info_path=os.path.join(target, "basic_info.json"),
            components_path={
                ComponentType.PROMPT: os.path.join(target, "prompt.py"),
                ComponentType.AGENT: os.path.join(target, "agent.py"),
            },
            trajectory_path=os.path.join(target, "trajectories.json"),
        )
        cand.save_basic_info()
        self.candidates[round_id] = cand
        return cand

    def modify_candidate(self, round_id: int, changes: List[Tuple[ComponentType, Union[str, Tuple[str, str]]]]) -> Candidate:
        """
        Apply a batch of changes to a candidate.

        Each change is a tuple of (component_type, payload). Payload formats:
        - PROMPT: (var_name: str, content: str)
        - AGENT:  str (full file content)
        """
        cand = self.candidates[round_id]
        for ctype, payload in changes:
            if ctype == ComponentType.PROMPT:
                assert isinstance(payload, tuple) and len(payload) == 2
                var, content = payload
                cand.save_component(ComponentType.PROMPT, var, content)  # type: ignore[arg-type]
            elif ctype == ComponentType.AGENT:
                cand.save_component(ComponentType.AGENT, payload)  # type: ignore[arg-type]
            else:
                raise NotImplementedError(f"Unsupported component: {ctype}")
        cand.save_basic_info()
        return cand

    def save_components(self, round_id: int) -> None:
        cand = self.candidates[round_id]
        cand.save_basic_info()
