from __future__ import annotations

from typing import Dict, List, Optional
import os
import re

from pydantic import BaseModel, Field
from base.engine.logs import logger

from learning.modules.base.candidate import CandidateManager, Candidate, ComponentType
from learning.modules.base.selection import Selection, SelectionType, CurrentBestSelection, ParetoFrontSelection
from learning.modules.base.optimization import Optimization, OptSignalMode
from learning.modules.base.evaluation import Evaluation, EvaluationMetric, EvaluationResult


class LearnerConfig(BaseModel):
    """Config schema for the base learner.

    This intentionally focuses on wiring and leaves concrete modules optional.
    """

    # Workspace and seeds
    workspace_path: str = Field(default="workspace/experiments/learning/base")
    seed_file_path: str = Field(default="workspace/seed_components")

    # Rounds and targets
    max_rounds: int = Field(default=3)
    target_components: List[ComponentType] = Field(default_factory=lambda: [ComponentType.PROMPT])

    # Selection
    selection_type: SelectionType = Field(default=SelectionType.CURRENT_BEST)
    selection_metrics: List[EvaluationMetric] = Field(default_factory=lambda: [EvaluationMetric.ACCURACY])

    # Optional evaluator / optimization (pluggable)
    enable_evaluation: bool = Field(default=False)


class Learner(BaseModel):
    """Base learner orchestrating selection → optimization → candidate creation.

    Provide concrete Evaluation/Optimization implementations externally and set on the instance.
    """

    config: LearnerConfig

    # Pluggable modules (optional)
    evaluator: Optional[Evaluation] = Field(default=None)
    optimization: Optional[Optimization] = Field(default=None)

    # Internal state
    candidate_manager: Optional[CandidateManager] = Field(default=None)
    selection: Optional[Selection] = Field(default=None)

    @classmethod
    def from_yaml(cls, cfg: Dict) -> "Learner":
        """Build a learner from a parsed YAML dict."""
        # Flatten nested structure if present
        ws = (cfg or {}).get("workspace", {})
        st = (cfg or {}).get("settings", {})
        sel = (cfg or {}).get("selection", {})

        # Enable evaluation by default; no separate 'enabled' flag required.
        flat = {
            "workspace_path": ws.get("workspace_path", "workspace/experiments/learning/base"),
            "seed_file_path": ws.get("seed_file_path", "workspace/seed_components"),
            "max_rounds": st.get("max_rounds", 3),
            "enable_evaluation": True,
        }
        # Target components
        target = (cfg or {}).get("optimization", {}).get("target_components", ["prompt"])
        comp_map = []
        for x in target:
            try:
                comp_map.append(ComponentType(x))
            except Exception:
                pass
        if not comp_map:
            comp_map = [ComponentType.PROMPT]
        flat["target_components"] = comp_map

        # Selection
        sel_type = sel.get("type", "current_best")
        try:
            flat["selection_type"] = SelectionType(sel_type)
        except Exception:
            flat["selection_type"] = SelectionType.CURRENT_BEST
        metrics = sel.get("metrics", ["accuracy"])
        met_list: List[EvaluationMetric] = []
        for m in metrics:
            try:
                met_list.append(EvaluationMetric(m))
            except Exception:
                pass
        if not met_list:
            met_list = [EvaluationMetric.ACCURACY]
        flat["selection_metrics"] = met_list

        learner = cls(config=LearnerConfig(**flat))
        logger.info(f"Creating candidate manager with workspace: {learner.config.workspace_path}")
        learner.candidate_manager = CandidateManager(
            workspace_path=learner.config.workspace_path,
            seed_file_path=learner.config.seed_file_path,
        )
        # Selection strategy instance
        if learner.config.selection_type == SelectionType.CURRENT_BEST:
            learner.selection = CurrentBestSelection()
        elif learner.config.selection_type == SelectionType.PARETO_FRONT:
            learner.selection = ParetoFrontSelection()
        else:
            learner.selection = CurrentBestSelection()
        return learner

    def _select_parent(self) -> Candidate:
        assert self.candidate_manager is not None, "candidate_manager not initialized"
        assert self.candidate_manager.candidates, "No candidates available to select from"
        assert self.selection is not None, "selection not initialized"
        
        parent = self.selection.select(self.candidate_manager.candidates, self.config.selection_metrics)
        logger.info(f"Selected parent candidate from round {parent.round}")
        return parent

    async def _evaluate(self, cand: Candidate) -> Optional[EvaluationResult]:
        if not self.config.enable_evaluation or self.evaluator is None:
            logger.debug("Evaluation disabled or evaluator not configured")
            return None
        
        logger.info(f"Evaluating candidate from round {cand.round}")
        result = await self.evaluator.evaluate_candidate(cand)
        
        if result and result.metrics:
            metrics_str = ", ".join([f"{k.value}: {v:.4f}" for k, v in result.metrics.items()])
            logger.info(f"Evaluation completed - {metrics_str}")
        else:
            logger.warning("Evaluation completed but no metrics returned")
        # Persist metrics and trajectories to candidate folder
        cand.metrics = result.metrics
        try:
            # Transform runner trajectories into learning-compatible trajectory list
            from learning.trajectory import Trajectory, TrajectoryStep  # lazy import

            traj_items = []
            for t in (result.trajectories or []):
                actions = t.get("actions", [])
                steps = []
                for idx, a in enumerate(actions):
                    try:
                        steps.append(
                            TrajectoryStep(
                                step_index=idx,
                                obs=a.get("observation"),
                                action=a.get("action", {}),
                                thinking=a.get("thought"),
                                result=a.get("result"),
                                reward=float(a.get("reward", 0.0)),
                                events=a.get("events", []) or [],
                            ).model_dump()
                        )
                    except Exception:
                        # Best-effort; skip malformed step
                        continue
                try:
                    traj = Trajectory(
                        world_id=str(t.get("world_id")),
                        agent_name=str(t.get("agent_name", "agent")),
                        metadata={
                            "initial_observation": t.get("initial_observation"),
                        },
                        steps=steps,
                        total_reward=float(t.get("total_reward", 0.0)),
                        finished=True,
                    ).model_dump()
                except Exception:
                    # Fallback to a minimal dict if model fails
                    traj = {
                        "world_id": str(t.get("world_id")),
                        "agent_name": str(t.get("agent_name", "agent")),
                        "metadata": {"initial_observation": t.get("initial_observation")},
                        "steps": steps,
                        "total_reward": float(t.get("total_reward", 0.0)),
                        "finished": True,
                    }
                traj_items.append(traj)

            # Ensure path ends with trajectories.json
            try:
                import os

                cand.trajectory_path = os.path.join(cand.candidate_folder, "trajectories.json")
                os.makedirs(cand.candidate_folder, exist_ok=True)
                import json

                with open(cand.trajectory_path, "w", encoding="utf-8") as f:
                    json.dump(traj_items, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        finally:
            cand.save_basic_info()
        return result

    async def run(self, start_round: Optional[int] = None) -> None:
        """Run the generic learning loop without concrete optimization specifics.

        Steps per round:
          - select parent candidate via self.selection
          - create new candidate (copying files from parent)
          - if optimization provided, it will modify components (not implemented here)
          - optional evaluation via self.evaluator
        """
        # Load or bootstrap
        assert self.candidate_manager is not None, "candidate_manager not initialized"
        logger.info("Loading existing candidates...")
        self.candidate_manager._load()

        # Determine starting round (the last completed round index). We will create from start+1.
        existing_max = max(self.candidate_manager.candidates.keys()) if self.candidate_manager.candidates else 0
        # If user provides start_round (first new round id), prefer it only if it's ahead of existing
        if start_round is not None and start_round > existing_max:
            start_idx = int(start_round) - 1
        else:
            start_idx = existing_max
        logger.info(f"Starting from round {start_idx + 1}, max rounds: {self.config.max_rounds}")

        for r in range(start_idx + 1, start_idx + 1 + self.config.max_rounds):
            logger.info(f"=== Starting learning round {r} ===")
            # choose parent according to selection
            parent = self._select_parent()

            # If parent has no metrics yet, evaluate it first (seed initial prompt if configured)
            if (not parent.metrics) and self.config.enable_evaluation and self.evaluator is not None:
                logger.info("Parent candidate has no metrics, evaluating first...")
                try:
                    if self.optimization is not None and getattr(self.optimization, "initial_prompt", False):
                        # Seed LEARNED_INSTRUCTION_PROMPT from env/agent_instruction.txt for the parent if absent
                        var_name = getattr(self.optimization, "prompt_var_name", "LEARNED_INSTRUCTION_PROMPT")
                        prompt_src = parent.get_component(ComponentType.PROMPT)
                        need_seed = True
                        if prompt_src:
                            has_triple = re.search(rf"^(\s*){re.escape(var_name)}\s*=\s*([\"\']{{3}})", prompt_src, flags=re.M) is not None
                            has_none = re.search(rf"^(\s*){re.escape(var_name)}\s*=\s*None\b", prompt_src, flags=re.M) is not None
                            # If triple exists and not None, no need to seed
                            need_seed = (not has_triple) or has_none
                        if need_seed:
                            env_path = getattr(self.optimization, "env_folder_path", None)
                            if env_path:
                                inst_path = os.path.join(env_path, "agent_instruction.txt")
                                if os.path.isfile(inst_path):
                                    with open(inst_path, "r", encoding="utf-8") as f:
                                        init_prompt = f.read()
                                    parent.save_component(ComponentType.PROMPT, var_name, init_prompt)
                                    parent.save_basic_info()
                except Exception:
                    pass

                await self._evaluate(parent)

            # create new candidate from parent
            logger.info(f"Creating new candidate for round {r} from parent round {parent.round}")
            cand = self.candidate_manager.create_candidate(round_id=r, parent_round=parent.round)

            # Apply optimization if provided (simple wiring)
            if self.optimization is not None:
                logger.info("Applying optimization to candidate...")
                try:
                    # Default to instruction signal if not otherwise specified
                    await self.optimization.optimize_candidate(
                        cand,
                        self.config.target_components,
                    )
                    logger.info("Optimization applied successfully")
                except NotImplementedError:
                    # Allow running without concrete optimization
                    logger.warning("Optimization not implemented, skipping")
                    pass
                except Exception as e:
                    logger.error(f"Optimization failed: {e}")
                    raise

            # Optional evaluation to attach metrics
            await self._evaluate(cand)

        # end
        logger.info(f"Learning loop completed after {self.config.max_rounds} rounds")
        return None
