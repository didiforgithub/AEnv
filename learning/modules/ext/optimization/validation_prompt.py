AGENT_CODE_FIX_AND_SMOKE_PROMPT = """
🎯 AGENT CODE FIX & SMOKE TEST TASK
Working Directory: {workspace}
Target Environment Folder: {env_folder_path}

Your Job: Fix candidate agent code (agent.py) in the working directory and verify it can run a short episode in the target environment (maximum {steps} steps on one validation world). If anything fails after your final attempt, stop and do not fabricate success.

Context & Constraints:
- You are inside the candidate folder: {workspace}
- The environment folder to use is: {env_folder_path}
- The agent implementation is in ./agent.py and may import from ./prompt.py
- Use Python 3.11, and repo imports are available relative to project root
- Do not modify base library files; only edit files under the working directory if needed
- Keep the agent interface compatible with BaseAgent; ensure parse_action parses <action>...</action> JSON/Literal

Required Steps (perform via bash commands):
1) SYNTAX CHECK
   - Run: python -m py_compile agent.py
   - If it fails, open and fix agent.py issues (e.g., f-string braces must be doubled {{ }}, missing imports, etc.) then re-run until it passes

2) IMPORT CHECK
   - Create a small script import_check.py that imports the solver class from agent.py similarly to how evaluation loader does:
       - Prefer classes named ReActAgent, Agent, CustomAgent, InstructionalSolverAgent, SolverAgent
       - Ensure it subclasses autoenv.agent.base.base_agent.BaseAgent or learning.solver.SolverAgent
       - Exit with non-zero code on failure
   - Run it with: python import_check.py

3) RUNTIME SMOKE TEST (MAX {steps} STEPS)
   - Write a script smoke_agent.py that:
       - from evals.benchmark import Benchmark
       - bench = Benchmark(env_folder_path="{env_folder_path}")
       - world_id = bench._list_env_worlds("{env_folder_path}", mode="val")[0]
       - env, env_info = bench._load_env_world("{env_folder_path}", world_id)
       - env_info["max_step"] = {steps}
       - Dynamically load solver class from ./agent.py (same rule as import_check.py)
       - Create LLM from optimize config if available; otherwise, construct a minimal AsyncLLM using default config
         (from autoenv.engine.async_llm import LLMsConfig, create_llm_instance; pick a small model like 4o-mini if present)
       - Run: await solver.run(env, env_info) (use asyncio.run wrapper)
       - Print SUCCESS on completion
   - Run: python smoke_agent.py

4) LOGGING & BACKUP
   - Before the first modification, save a backup: cp agent.py agent.py.bak
   - For any fix attempts, append error summaries and actions to agent_fix.log in the working directory

✅ COMPLETION
When smoke test passes (script prints SUCCESS), run:
echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'

⚠️ IMPORTANT
- Do not fabricate outputs. If after your best effort the smoke test still fails, stop and do not print completion signal.
- Keep edits minimal and focused on fixing agent code in ./agent.py and related prompt variables in ./prompt.py if necessary.
"""

