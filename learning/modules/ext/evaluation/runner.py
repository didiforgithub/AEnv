from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field
from pydantic import ConfigDict

from evaluation.benchmark import Benchmark
from base.engine.logs import logger
from base.agent.base_agent import SolverAgent
from base.agent.base_agent import SolverAgent as _BaseSolver
from base.agent.base_agent import BaseAgent as _BaseAgent
from learning.modules.base.candidate import Candidate
from learning.modules.base.evaluation import EvaluationMetric


class CandidateRunner(BaseModel):
    """Run a candidate across benchmark worlds.

    - Dynamically loads agent class from candidate.candidate_folder/agent.py when present.
    - Falls back to learning.solver.SolverAgent when no candidate agent is provided.
    - Uses evals.Benchmark helpers to list/load worlds.
    - Computes mean reward as ACCURACY and per-run LLM cost delta as COST.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_folder_path: str = Field(...)
    # Persistent LLM instance for accurate cost deltas
    llm: Any = Field(...)
    world_ids: Optional[List[str]] = Field(default=None)
    world_concurrency: int = Field(default=4)
    val_world_count: Optional[int] = Field(default=None)
    repeats_per_world: int = Field(default=1)
    enable_candidate_agent: bool = Field(default=True)
    strict_agent: bool = Field(default=True)
    # Smoke test configuration: run a tiny precheck before full eval
    precheck_smoke_test: bool = Field(default=True)
    smoke_world_count: int = Field(default=1)
    smoke_steps: int = Field(default=3)
    smoke_fail_fast: bool = Field(default=True)
    smoke_llm: Optional[Any] = Field(default=None, description="LLM used for smoke test; defaults to main llm")

    # --- Benchmark helpers ---
    def list_worlds(self) -> List[str]:
        bench = Benchmark(env_folder_path=self.env_folder_path)
        worlds = bench._list_env_worlds(self.env_folder_path, mode="val")
        if self.val_world_count is not None and self.val_world_count > 0:
            worlds = worlds[: self.val_world_count]
        logger.debug(f"Listed {len(worlds)} validation worlds")
        return worlds

    def load_env_world(self, world_id: str):
        bench = Benchmark(env_folder_path=self.env_folder_path)
        return bench._load_env_world(self.env_folder_path, world_id)

    # --- Core execution ---
    async def run_candidate(self, candidate: Candidate) -> Tuple[List[Dict], Dict[EvaluationMetric, float]]:
        logger.info(f"Running candidate from round {candidate.round}")
        # Resolve agent class; under strict mode, propagate load failure to zero out this candidate
        try:
            solver_cls = self._resolve_solver_cls(candidate)
            logger.debug(f"Resolved solver class: {solver_cls.__name__}")
        except Exception as e:
            logger.error(f"Failed to resolve solver class: {e}")
            ids = list(self.world_ids or self.list_worlds())
            trajectories: List[Dict] = []
            rewards: List[float] = []
            for world_id in ids:
                rewards.append(0.0)
                trajectories.append(
                    {
                        "world_id": world_id,
                        "agent_name": "load_error",
                        "actions": [],
                        "total_reward": 0.0,
                        "steps": 0,
                        "events_count": {"error": 1},
                        "initial_observation": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
            metrics = {
                EvaluationMetric.ACCURACY: 0.0,
                EvaluationMetric.COST: 0.0,
            }
            return trajectories, metrics

        # Optional precheck: run a single short world to catch early runtime errors
        if self.precheck_smoke_test:
            try:
                ids_pre = list(self.world_ids or self.list_worlds())
            except Exception:
                ids_pre = []
            if ids_pre:
                world_id = ids_pre[0]
                logger.info(f"Smoke test on world: {world_id} (steps <= {self.smoke_steps})")
                error_msg: Optional[str] = None
                try:
                    env, env_info = self.load_env_world(world_id)
                    # Limit steps for smoke run
                    try:
                        env_info = dict(env_info)
                    except Exception:
                        env_info = {"world_id": world_id, "action_space": "", "agent_instruction": ""}
                    env_info["max_step"] = int(self.smoke_steps)
                    solver = solver_cls(llm=(self.smoke_llm or self.llm))
                    # Run smoke
                    _ = await solver.run(env, env_info)
                except Exception as e:
                    error_msg = f"Smoke test failed: {type(e).__name__}: {e}"
                if error_msg and self.smoke_fail_fast:
                    # Directly short-circuit on smoke failure without auto-fix
                    try:
                        log_path = os.path.join(candidate.candidate_folder, "smoke_test.log")
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(error_msg + "\n")
                    except Exception:
                        pass
                    logger.error(error_msg)
                    return [
                        {
                            "world_id": world_id,
                            "agent_name": "smoke_error",
                            "actions": [],
                            "total_reward": 0.0,
                            "steps": 0,
                            "events_count": {"error": 1},
                            "initial_observation": None,
                            "error": error_msg,
                        }
                    ], {
                        EvaluationMetric.ACCURACY: 0.0,
                        EvaluationMetric.COST: 0.0,
                    }

        trajectories: List[Dict] = []
        rewards: List[float] = []

        ids = list(self.world_ids or self.list_worlds())
        logger.info(f"Running on {len(ids)} worlds with concurrency {self.world_concurrency}")
        semaphore = asyncio.Semaphore(max(1, int(self.world_concurrency)))

        # Snapshot cost (after smoke test, so smoke usage isn't counted)
        start_cost = 0.0
        try:
            summary0 = self.llm.get_usage_summary()
            if isinstance(summary0, dict):
                start_cost = float(summary0.get("total_cost", 0.0))
            logger.debug(f"Starting cost: {start_cost:.6f}")
        except Exception:
            start_cost = 0.0

        async def run_one(world_id: str):
            async with semaphore:
                solver = None
                try:
                    env, env_info = self.load_env_world(world_id)
                    solver = solver_cls(llm=self.llm)
                    result = await solver.run(env, env_info)
                except Exception as e:
                    # Gracefully degrade: mark this world as failed with zero reward
                    result = {
                        "total_reward": 0.0,
                        "step": 0,
                        "events_count": {"error": 1},
                        "initial_observation": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                return world_id, solver, result

        # Schedule repeats per world
        tasks: List[asyncio.Task] = []
        for w in ids:
            for _ in range(max(1, int(self.repeats_per_world))):
                tasks.append(asyncio.create_task(run_one(w)))
        
        total_runs = len(tasks)
        logger.info(f"Scheduled {total_runs} total runs ({len(ids)} worlds × {self.repeats_per_world} repeats)")

        # Collect results and aggregate by world
        from collections import defaultdict
        world_runs: Dict[str, List[Tuple[Optional[Any], Dict]]] = defaultdict(list)
        logger.info("Executing all runs...")
        for world_id, solver, result in await asyncio.gather(*tasks):
            world_runs[world_id].append((solver, result))
        logger.info("All runs completed")

        # Load per-world max rewards for normalization
        bench = Benchmark(env_folder_path=self.env_folder_path)
        max_rewards_map = bench._load_max_rewards(self.env_folder_path)

        # Build per-world representative trajectory and reward mean (raw and normalized)
        world_means_raw: List[float] = []
        world_means_norm: List[float] = []
        for world_id in ids:
            runs = world_runs.get(world_id, [])
            if not runs:
                world_means_raw.append(0.0)
                world_means_norm.append(0.0)
                continue
            # Mean reward for this world
            rlist = [float(r.get("total_reward", 0.0)) for (_, r) in runs]
            mean_r = sum(rlist) / len(rlist)
            world_means_raw.append(mean_r)
            # Normalize by max reward if available
            mr = float(max_rewards_map.get(world_id, 0.0) or 0.0)
            if mr > 0:
                norm = max(0.0, min(1.0, mean_r / mr))
            else:
                norm = 0.0
            world_means_norm.append(norm)
            # Choose best run as representative for trajectory
            best_idx = max(range(len(runs)), key=lambda i: float(runs[i][1].get("total_reward", 0.0)))
            solver, result = runs[best_idx]
            # Safe agent name when solver is None
            agent_name = "error" if solver is None else (getattr(solver, "name", None) or solver.__class__.__name__)
            trajectories.append(
                {
                    "world_id": world_id,
                    "agent_name": agent_name,
                    "actions": list(getattr(solver, "past_actions", [])) if solver is not None else [],
                    "total_reward": result.get("total_reward", 0),
                    "steps": result.get("step", 0),
                    "events_count": result.get("events_count", {}),
                    "initial_observation": result.get("initial_observation"),
                }
            )

        # Compute metrics
        # Accuracy as mean of per-world normalized mean rewards
        acc = (sum(world_means_norm) / len(world_means_norm)) if world_means_norm else 0.0
        total_cost = 0.0
        try:
            summary = self.llm.get_usage_summary()
            if isinstance(summary, dict):
                end_cost = float(summary.get("total_cost", 0.0))
                total_cost = max(0.0, end_cost - start_cost)
        except Exception:
            total_cost = 0.0

        metrics = {
            EvaluationMetric.ACCURACY: float(acc),
            EvaluationMetric.COST: float(total_cost),
        }
        
        logger.info(f"Candidate evaluation completed - Accuracy: {acc:.4f}, Cost: {total_cost:.6f}")
        return trajectories, metrics

    # --- Candidate agent loading ---
    def _resolve_solver_cls(self, candidate: Candidate) -> Type:
        if not self.enable_candidate_agent:
            logger.debug("Candidate agent disabled, using default solver")
            return self._default_solver()
        # Prefer agent.py in candidate folder
        cdir = getattr(candidate, "candidate_folder", None)
        if cdir:
            ap = os.path.join(cdir, "agent.py")
            if os.path.isfile(ap):
                try:
                    logger.debug(f"Loading candidate agent from {ap}")
                    return self._load_solver_from_file(ap)
                except Exception as e:
                    logger.warning(f"Failed to load candidate agent: {e}")
                    if self.strict_agent:
                        raise
                    return self._default_solver()
            else:
                logger.debug(f"No agent.py found in {cdir}")
                if self.strict_agent:
                    raise FileNotFoundError(f"agent.py missing in {cdir}")
        return self._default_solver()

    def _default_solver(self) -> Type:
        logger.info("Using default solver: SolverAgent")

        return SolverAgent

    def _select_solver_class(self, module: Any) -> Optional[Type]:

        preferred = (
            "ReActAgent",
            "Agent",
            "CustomAgent",
            "InstructionalSolverAgent",
            "SolverAgent",
        )
        for name in preferred:
            cls = getattr(module, name, None)
            if inspect.isclass(cls) and (issubclass(cls, _BaseSolver) or issubclass(cls, _BaseAgent)):
                return cls
        for _, cls in inspect.getmembers(module, inspect.isclass):
            try:
                if (issubclass(cls, _BaseSolver) and cls is not _BaseSolver) or (
                    issubclass(cls, _BaseAgent) and cls.__name__ != "BaseAgent"
                ):
                    return cls
            except Exception:
                continue
        return None

    def _load_solver_from_file(self, path: str) -> Type:
        """Load agent.py as a namespaced module to allow relative imports (e.g., from .prompt import *)."""
        dirpath = os.path.dirname(os.path.abspath(path))
        pkg_name = f"candidate_pkg_{abs(hash(dirpath))}"

        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [dirpath]  # type: ignore[attr-defined]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

        # Preload prompt.py if present
        prompt_path = os.path.join(dirpath, "prompt.py")
        if os.path.isfile(prompt_path):
            mod_name = f"{pkg_name}.prompt"
            if mod_name not in sys.modules:
                spec_p = importlib.util.spec_from_file_location(mod_name, prompt_path)
                if spec_p and spec_p.loader:
                    mod_p = importlib.util.module_from_spec(spec_p)
                    mod_p.__package__ = pkg_name
                    sys.modules[mod_name] = mod_p
                    spec_p.loader.exec_module(mod_p)  # type: ignore

        # Load agent module
        mod_name = f"{pkg_name}.agent"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Invalid module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = pkg_name
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore

        cls = self._select_solver_class(module)
        if not cls:
            raise ImportError("No compatible agent class found in agent.py")
        return cls
