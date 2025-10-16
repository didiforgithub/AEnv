import argparse
import asyncio
import yaml
import os
from datetime import datetime

from base.engine.logs import logger
from base.engine.utils import summarize_candidates
from base.engine.async_llm import LLMsConfig, create_llm_instance

from learning.learner import Learner
from learning.modules.ext.evaluation.benchmark import BenchmarkEvaluation
from learning.modules.ext.evaluation.runner import CandidateRunner
from learning.modules.base.candidate import ComponentType
from learning.modules.base.optimization import Optimization, OptimizationSignalGenerator
from learning.modules.ext.optimization.signal_prompt import (
    DYNAMICS_OPTIMIZATION_PROMPT,
    INSTRUCTION_OPTIMIZATION_PROMPT,
)


def _get_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/learning/learner_dynamics_agent.yaml")
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "Explicit workspace path to use (resume or write). "
            "If provided, disables timestamped stamping."
        ),
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=None,
        help=(
            "First new round number to create. "
            "If omitted, auto-continues from the last existing round + 1."
        ),
    )
    args = parser.parse_args()

    logger.info("Starting learning process...")
    logger.info(f"Using config file: {args.config}")
    
    cfg = _get_cfg(args.config)

    # Resolve environment path early and stamp workspace_path with timestamp + env name
    ws = (cfg or {}).get("workspace", {})
    runner_cfg = (cfg or {}).get("runner", {})
    env_folder_path = (
        runner_cfg.get("env_folder_path")
        or ws.get("env_folder_path")
        or "benchmarks/4_BackwardTimes"
    )
    # If user specifies --workspace, honor it and skip stamping. Otherwise stamp.
    if args.workspace:
        cfg.setdefault("workspace", {})["workspace_path"] = str(args.workspace)
        stamped_workspace = str(args.workspace)
        logger.info("Using explicit workspace from CLI; timestamp stamping disabled")
    else:
        base_workspace = ws.get("workspace_path", "workspace/experiments/learning/base")
        env_name = os.path.basename(str(env_folder_path).rstrip(os.sep)) or "env"
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        stamped_workspace = os.path.join(base_workspace, f"{ts}_{env_name}")
        cfg.setdefault("workspace", {})["workspace_path"] = stamped_workspace

    logger.info(f"Environment folder: {env_folder_path}")
    logger.info(f"Workspace path: {stamped_workspace}")

    # Build learner with the stamped workspace path
    learner = Learner.from_yaml(cfg)
    logger.info("Learner initialized successfully")

    # Resolve LLM config once; runner will construct and hold the LLM instance
    llms = (cfg or {}).get("llms", {})
    exec_llm_name = llms.get("exec_llm_name")
    exec_llm = None
    if exec_llm_name:
        logger.info(f"Creating execution LLM: {exec_llm_name}")
        exec_llm_cfg = LLMsConfig.default().get(exec_llm_name)
        exec_llm = create_llm_instance(exec_llm_cfg)

    # Auto-configure benchmark evaluation from config (no 'enabled' flag required)
    runner = None
    evaluator_cfg = (cfg or {}).get("evaluator", {})
    if str(evaluator_cfg.get("type", "")).lower() == "benchmark":
        # Resolve runner configuration
        world_concurrency = int(
            runner_cfg.get("world_concurrency", 4)
        )
        repeats_per_world = int((cfg or {}).get("settings", {}).get("repeats_per_world", 1))
        # Use validation sample count from sample section only
        sample_cfg = (cfg or {}).get("sample", {})
        val_world_count = sample_cfg.get("val_world_count")

        runner = CandidateRunner(
            env_folder_path=env_folder_path,
            llm=exec_llm,
            world_concurrency=world_concurrency,
            val_world_count=val_world_count,
            repeats_per_world=repeats_per_world,
            enable_candidate_agent=True,
        )
        # Only run smoke tests when optimizing the AGENT component
        try:
            runner.precheck_smoke_test = (ComponentType.AGENT in learner.config.target_components)
        except Exception:
            runner.precheck_smoke_test = False
        learner.evaluator = BenchmarkEvaluation(runner=runner)
        learner.config.enable_evaluation = True
        logger.info(f"Benchmark evaluation configured with {world_concurrency} world concurrency")




    # Optimization: default wiring (can be extended later)
    opt_cfg = (cfg or {}).get("optimization", {})
    mode = str((opt_cfg or {}).get("mode", "default")).lower()
    optimization_target = opt_cfg.get("target_components")
    signal_generator_names = opt_cfg.get("signal_generators")
    # Wire optimization with plain-text signal generators if configured
    prompt_var_name = (opt_cfg or {}).get("default", {}).get("prompt_var_name", "LEARNED_INSTRUCTION_PROMPT")
    sig_names = list((opt_cfg or {}).get("signal_generators", []) or [])
    sig_gens = []

    # Optimization LLM (fallback to exec if opt not provided)
    opt_llm_name = llms.get("opt_llm_name") if isinstance(llms, dict) else None
    opt_llm = None
    if opt_llm_name:
        try:
            logger.info(f"Creating optimization LLM: {opt_llm_name}")
            opt_llm_cfg = LLMsConfig.default().get(opt_llm_name)
            opt_llm = create_llm_instance(opt_llm_cfg)
        except Exception:
            logger.warning(f"Failed to create optimization LLM {opt_llm_name}, falling back to execution LLM")
            opt_llm = exec_llm
    else:
        opt_llm = exec_llm

    for name in sig_names:
        if str(name).lower() == "dynamics":
            sig_gens.append(OptimizationSignalGenerator(analyze_prompt=DYNAMICS_OPTIMIZATION_PROMPT, analyze_llm=opt_llm))
        elif str(name).lower() == "instruction":
            sig_gens.append(OptimizationSignalGenerator(analyze_prompt=INSTRUCTION_OPTIMIZATION_PROMPT, analyze_llm=opt_llm))

    initial_prompt_flag = bool((opt_cfg or {}).get("initial_prompt", False))
    if sig_gens or initial_prompt_flag:
        logger.info(f"Configuring optimization with {len(sig_gens)} signal generators")
        learner.optimization = Optimization(
            signal_generators=sig_gens,
            prompt_var_name=prompt_var_name,
            optimize_llm=opt_llm,
            initial_prompt=initial_prompt_flag,
            env_folder_path=env_folder_path,
            selection_type=learner.config.selection_type,
        )

    # After opt_llm resolved, set smoke_llm for runner if available
    if runner is not None:
        runner.smoke_llm = opt_llm

    logger.info("Starting learning loop...")
    # start_round argument semantics: create the first new candidate with this round id
    await learner.run(start_round=args.start_round)
    logger.info("Learning loop completed")

    # Post-run summary: parent-child relations and improvements
    try:
        logger.info("Generating candidate summary...")
        summarize_candidates(learner.candidate_manager.workspace_path)  # type: ignore[arg-type]
        logger.info("Candidate summary completed")
    except Exception as e:
        logger.warning(f"Failed to generate candidate summary: {e}")
    
    logger.info("Learning process finished successfully")


if __name__ == "__main__":
    asyncio.run(main())
