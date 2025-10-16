from __future__ import annotations

from typing import Any

from pydantic import Field, ConfigDict

from learning.modules.base.evaluation import Evaluation, EvaluationResult

import os
import time
import importlib.util
import sys
import yaml
import asyncio
import json
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Type
from base.engine.utils import  read_file_content
from base.engine.logs import logger
from base.agent.base_agent import BaseAgent
from base.env.base_env import SkinEnv


class BenchmarkEvaluation(Evaluation):
    """Evaluation adapter that runs a candidate across benchmark worlds via a runner.

    Runner contract (expected):
      - run_candidate(candidate) -> Tuple[List[Dict], Dict[EvaluationMetric, float]]
        returns (trajectories, metrics)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Any = Field(..., description="Object exposing run_candidate(candidate)")

    async def evaluate_candidate(self, candidate) -> EvaluationResult:  # noqa: ANN001
        trajectories, metrics = await self.runner.run_candidate(candidate)
        return EvaluationResult(
            candidate_round=getattr(candidate, "round", -1),
            metrics=metrics,
            trajectories=trajectories,
        )

class EnvWrapper:
    """环境包装器，确保所有环境操作都在正确的工作目录下进行"""
    
    def __init__(self, env, env_folder_path):
        self._env = env
        self._env_folder_path = env_folder_path
        self._original_cwd = os.getcwd()
    
    def __getattr__(self, name):
        """代理所有属性访问到被包装的环境对象"""
        attr = getattr(self._env, name)
        
        # 如果是方法，包装它以在正确的目录下执行
        if callable(attr):
            def wrapped_method(*args, **kwargs):
                original_cwd = os.getcwd()
                try:
                    os.chdir(self._env_folder_path)
                    return attr(*args, **kwargs)
                finally:
                    os.chdir(original_cwd)
            return wrapped_method
        else:
            return attr


class Benchmark(BaseModel):
    env_folder_path: str = Field(default="")
    result_folder_path: str = Field(default="workspace/logs/results")
    trajectory_folder_path: str = Field(default="workspace/logs/trajectories")
    llm_name: str = Field(default="")
    timestamp: str = Field(default="")
    env_name: str = Field(default="")
    results: Dict = Field(default_factory=dict)
    max_rewards: Dict = Field(default_factory=dict)
    costs: Dict[str, float] = Field(default_factory=dict)
    per_world_max_rewards: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    world_details: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    env_durations: Dict[str, float] = Field(default_factory=dict)
    event_totals: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    env_world_ids: Dict[str, List[str]] = Field(default_factory=dict)

    def _init_result_folder(self):
        # Keep a single CSV under result_folder_path; do not nest per-LLM folders
        os.makedirs(self.result_folder_path, exist_ok=True)
        # Trajectory folder is controlled by caller; ensure directory exists
        os.makedirs(self.trajectory_folder_path, exist_ok=True)

    def _list_env_worlds(self, env_folder_path: str, mode="test") -> List[str]:
        """列出指定环境文件夹下的所有关卡（世界）"""
        env_path = Path(env_folder_path)
        if mode == "val":
            levels_dir = env_path / "val_levels"
        else:
            levels_dir = env_path / "levels"
        
        if not levels_dir.exists():
            logger.warning(f"关卡目录不存在: {levels_dir}")
            return []
        
        levels = []
        for level_file in levels_dir.glob("*.yaml"):
            level_id = level_file.stem
            levels.append(level_id)
        
        return sorted(levels)

    def _load_env(self, env_folder_path: str):
        """
        动态加载环境类
        从env_folder_path中的env_main.py加载继承自SkinEnv的环境类
        """
        env_path = Path(env_folder_path)
        env_main_path = env_path / "env_main.py"
        
        if not env_main_path.exists():
            raise ValueError(f"环境主文件不存在: {env_main_path}")
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        # 动态导入模块
        spec = importlib.util.spec_from_file_location("env_main", str(env_main_path))
        env_module = importlib.util.module_from_spec(spec)
        
        # 添加环境目录到sys.path以支持相对导入
        env_dir_str = str(env_path.resolve())
        if env_dir_str not in sys.path:
            sys.path.insert(0, env_dir_str)
        
        try:
            # 临时切换到环境目录以支持相对路径
            os.chdir(env_dir_str)
            spec.loader.exec_module(env_module)
        except Exception as e:
            raise ImportError(f"无法加载环境模块 {env_folder_path}: {e}")
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
            # 清理sys.path
            if env_dir_str in sys.path:
                sys.path.remove(env_dir_str)
        
        env_class = None
        for attr_name in dir(env_module):
            attr = getattr(env_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, SkinEnv) and 
                attr != SkinEnv):
                env_class = attr
                break
        
        if env_class is None:
            raise ValueError(f"在 {env_folder_path} 中未找到有效的环境类")
        
        return env_class

    def _validate_level(self, env_folder_path: str, level_id: str) -> bool:
        """验证关卡文件是否存在且有效（支持 val_levels 与 levels）"""
        env_path = Path(env_folder_path)
        # Prefer validation in both val_levels and levels to match listing behavior
        candidates = [
            env_path / "val_levels" / f"{level_id}.yaml",
            env_path / "levels" / f"{level_id}.yaml",
        ]
        for level_path in candidates:
            if level_path.exists():
                try:
                    with open(level_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    return True
                except yaml.YAMLError:
                    return False
        return False

    def _get_env_info(self, env_folder_path: str) -> Dict[str, Any]:
        """获取环境信息"""
        env_path = Path(env_folder_path)
        
        info = {}
        
        # 读取环境描述
        agent_instruction_path = env_path / "agent_instruction.txt"
        if agent_instruction_path.exists():
            info["agent_instruction"] = read_file_content(str(agent_instruction_path))
        
        # 读取动作空间
        action_space_path = env_path / "action_space.txt"
        if action_space_path.exists():
            info["action_space"] = read_file_content(str(action_space_path))
        
        # 读取配置
        config_path = env_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                info["config"] = config
        
        return info

    def _load_env_world(self, env_folder_path: str, world_id: str):
        """加载环境实例和世界信息"""
        # 验证关卡
        if not self._validate_level(env_folder_path, world_id):
            raise ValueError(f"关卡 {world_id} 在环境 {env_folder_path} 中不存在或无效")
        
        # 为兼容某些环境仅从 ./levels 读取，必要时将 world_id 映射到 ../val_levels/
        from pathlib import Path as _Path
        _env_p = _Path(env_folder_path)
        levels_path = _env_p / "levels" / f"{world_id}.yaml"
        val_levels_path = _env_p / "val_levels" / f"{world_id}.yaml"
        effective_world_id = world_id
        if (not levels_path.exists()) and val_levels_path.exists():
            # env 内部拼接 ./levels/{world_id}.yaml，则使用相对路径引导到 val_levels
            effective_world_id = f"../val_levels/{world_id}"

        # 获取环境信息
        env_info_data = self._get_env_info(env_folder_path)
        if not env_info_data.get("agent_instruction") or not env_info_data.get("action_space"):
            raise ValueError(f"环境 {env_folder_path} 缺少必要的配置文件")
        
        # 准备环境信息
        env_info = {
            "world_id": effective_world_id,
            "agent_instruction": env_info_data["agent_instruction"],
            "action_space": env_info_data["action_space"],
        }
        
        # 从配置中获取最大步数（如果存在）
        if "config" in env_info_data:
            config = env_info_data["config"]
            if isinstance(config, dict):
                termination = config.get("termination", {})
                if isinstance(termination, dict) and "max_steps" in termination:
                    env_info["max_step"] = termination["max_steps"]
        
        # 加载环境类
        env_class = self._load_env(env_folder_path)
        
        # 创建环境实例（需要在环境目录下创建以支持相对路径）
        env_name = Path(env_folder_path).name
        original_cwd = os.getcwd()
        try:
            os.chdir(env_folder_path)
            env = env_class(env_id=f"{env_name}_benchmark")
        finally:
            os.chdir(original_cwd)
        
        # 使用包装器确保所有环境操作都在正确的目录下进行
        wrapped_env = EnvWrapper(env, env_folder_path)
        
        return wrapped_env, env_info

    def _load_max_rewards(self, env_folder_path: str) -> Dict[str, float]:
        """加载环境的最大奖励信息"""
        env_path = Path(env_folder_path)
        max_rewards_path = env_path / "level_max_rewards.json"
        
        if not max_rewards_path.exists():
            logger.warning(f"最大奖励文件不存在: {max_rewards_path}")
            return {}
        
        try:
            with open(max_rewards_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取各关卡的最大奖励
            max_rewards = {}
            levels_data = data.get("levels", {})
            for level_name, level_info in levels_data.items():
                # 去掉 .yaml 后缀作为关卡ID
                level_id = level_name.replace('.yaml', '')
                max_reward = level_info.get("max_reward", 0.0)
                max_rewards[level_id] = max_reward
            
            return max_rewards
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"读取最大奖励文件失败 {max_rewards_path}: {e}")
            return {}

    def _calculate_max_reward_total(self, env_folder_path: str, world_ids: List[str]) -> float:
        """计算指定关卡的最大奖励总分"""
        max_rewards_dict = self._load_max_rewards(env_folder_path)

        filtered_max_rewards = {world_id: max_rewards_dict.get(world_id, 0.0) for world_id in world_ids}
        self.per_world_max_rewards[env_folder_path] = filtered_max_rewards

        total_max_reward = sum(filtered_max_rewards.values())
        return total_max_reward


    def _save_env_result_to_csv(self):
        """Append a single row to timestamped result CSV safely for multi-process.

        Writes header if file does not exist. Avoids read/modify/write races.
        """
        import csv
        import shutil
        import tempfile

        timestamp_prefix = self.timestamp or time.strftime("%m%d_%H%M")
        env_name = self.env_name or "unknown_env"
        csv_filename = f"{timestamp_prefix}_{env_name}_result.csv"
        csv_path = os.path.join(self.result_folder_path, csv_filename)

        if not self.results:
            return

        env_path = list(self.results.keys())[-1]
        total_reward = float(self.results.get(env_path, 0.0) or 0.0)
        max_reward_total = float(self.max_rewards.get(env_path, 0.0) or 0.0)
        ratio = (total_reward / max_reward_total) if max_reward_total else None
        cost = self.costs.get(env_path)
        world_details = self.world_details.get(env_path, [])
        events_summary = self.event_totals.get(env_path, {}) or {}
        duration_seconds = self.env_durations.get(env_path)
        loaded_world_ids = [str(world_id) for world_id in self.env_world_ids.get(env_path, [])]
        if loaded_world_ids:
            world_ids = loaded_world_ids
        else:
            world_ids = [str(detail.get("world_id")) for detail in world_details if detail.get("world_id")]
        world_count = len(world_ids)
        total_steps = sum(int(detail.get("steps") or 0) for detail in world_details)
        avg_steps_per_world = (total_steps / world_count) if world_count else None
        avg_reward_per_world = (total_reward / world_count) if world_count else None
        ratio_values = [float(detail.get("ratio")) for detail in world_details if detail.get("ratio") is not None]
        success_worlds = sum(1 for value in ratio_values if value >= 0.999)

        row = {
            "env_folder_path": env_path,
            "llm": self.llm_name or "",
            "total_reward": total_reward,
            "max_reward_total": max_reward_total,
            "ratio": ratio,
            "cost": cost,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "world_count": world_count,
            "world_ids": "|".join(world_ids),
            "avg_reward_per_world": avg_reward_per_world,
            "total_steps": total_steps,
            "avg_steps_per_world": avg_steps_per_world,
            "success_worlds": success_worlds,
            "events_summary": json.dumps(events_summary, ensure_ascii=False) if events_summary else "",
            "duration_seconds": duration_seconds,
        }

        os.makedirs(self.result_folder_path, exist_ok=True)

        exists = os.path.exists(csv_path)
        try:
            import fcntl  # type: ignore
        except Exception:
            fcntl = None

        if exists and os.path.getsize(csv_path) > 0:
            try:
                with open(csv_path, "r", encoding="utf-8", newline="") as rf:
                    reader = csv.DictReader(rf)
                    existing_fieldnames = reader.fieldnames or []
                    old_rows = list(reader)
                if existing_fieldnames != list(row.keys()):
                    fd, tmp_path = tempfile.mkstemp(prefix="bench_migrate_", suffix=".csv")
                    os.close(fd)
                    with open(tmp_path, "w", encoding="utf-8", newline="") as wf:
                        writer = csv.DictWriter(wf, fieldnames=list(row.keys()))
                        writer.writeheader()
                        for old_row in old_rows:
                            new_row = {key: old_row.get(key, "") for key in row.keys()}
                            writer.writerow(new_row)
                    shutil.move(tmp_path, csv_path)
                    exists = os.path.exists(csv_path)
            except Exception:
                pass

        fieldnames = list(row.keys())

        with open(csv_path, "a+", newline="", encoding="utf-8") as f:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow(row)
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


    async def execute(self, solver_class: Type[BaseAgent], solver_kwargs: Optional[Dict[str, Any]] = None, 
                     world_concurrency: int = 1, max_worlds: Optional[int] = None):
        """
        执行基准测试
        Args:
            solver_class: BaseAgent 子类
            solver_kwargs: 创建 solver 实例的参数，通常包含 llm 模型配置
            world_concurrency: world 级别的并发数，默认为1（串行）
            max_worlds: 最大运行的关卡数，None表示运行所有关卡
        """
        if solver_kwargs is None:
            solver_kwargs = {}

        # 初始化结果文件夹
        self._init_result_folder()

        env_folder_path = self.env_folder_path
        llm = (solver_kwargs or {}).get("llm")
        prev_cost = None
        if llm is not None and hasattr(llm, "get_usage_summary"):
            try:
                prev_cost = float(llm.get_usage_summary().get("total_cost", 0.0))
            except Exception:
                prev_cost = None

        world_ids = self._list_env_worlds(env_folder_path)
        if max_worlds is not None:
            world_ids = world_ids[:max_worlds]
        self.env_world_ids[env_folder_path] = list(world_ids)

        max_reward_total = self._calculate_max_reward_total(env_folder_path, world_ids)
        self.max_rewards[env_folder_path] = max_reward_total

        start_time = time.perf_counter()

        async def run_world(world_id: str):
            solver = solver_class(**solver_kwargs)
            env, env_info = self._load_env_world(env_folder_path, world_id)
            result = await solver.run(env, env_info)
            return world_id, result, env_info

        semaphore = asyncio.Semaphore(world_concurrency)

        async def run_world_with_semaphore(world_id: str):
            async with semaphore:
                return await run_world(world_id)

        raw_results = await asyncio.gather(*[
            run_world_with_semaphore(world_id) for world_id in world_ids
        ])

        elapsed = time.perf_counter() - start_time
        self.env_durations[env_folder_path] = elapsed

        per_world_max = self.per_world_max_rewards.get(env_folder_path, {})
        aggregated_events: Dict[str, int] = defaultdict(int)
        world_details: List[Dict[str, Any]] = []
        world_rewards: List[float] = []

        for world_id, result, env_info in raw_results:
            reward = float(result.get("total_reward") or 0.0)
            world_rewards.append(reward)
            steps = int(result.get("step") or 0)
            events_count = result.get("events_count") or {}
            if isinstance(events_count, dict):
                for event_name, count in events_count.items():
                    try:
                        aggregated_events[event_name] += int(count)
                    except Exception:
                        continue
            max_reward_world = per_world_max.get(world_id, 0.0)
            ratio_world = (reward / max_reward_world) if max_reward_world else None
            world_details.append(
                {
                    "world_id": world_id,
                    "reward": reward,
                    "steps": steps,
                    "events_count": events_count,
                    "max_reward": max_reward_world,
                    "ratio": ratio_world,
                    "max_step": env_info.get("max_step"),
                }
            )

        self.world_details[env_folder_path] = world_details
        self.event_totals[env_folder_path] = dict(aggregated_events)

        cur_env_total_reward = sum(world_rewards)
        self.results[env_folder_path] = cur_env_total_reward
        env_cost = None
        if llm is not None and hasattr(llm, "get_usage_summary") and prev_cost is not None:
            try:
                post_cost = float(llm.get_usage_summary().get("total_cost", 0.0))
                env_cost = max(0.0, post_cost - prev_cost)
            except Exception:
                env_cost = None
        self.costs[env_folder_path] = env_cost
        self._save_env_result_to_csv()

        env_desc = f"环境 {env_folder_path}" + (f" (前{max_worlds}个关卡)" if max_worlds else "")
        logger.info(f"{env_desc}:")
        logger.info(f"  实际总奖励: {cur_env_total_reward}")
        logger.info(f"  最大总奖励: {max_reward_total}")
        logger.info(f"  达成率: {cur_env_total_reward/max_reward_total*100:.2f}%" if max_reward_total > 0 else "  达成率: N/A")

        logger.info(f"Benchmark Results Saved to {self.result_folder_path}")


    async def execute_with_five(self, solver_class: Type[BaseAgent], solver_kwargs: Optional[Dict[str, Any]] = None, world_concurrency: int = 1):
        """
        执行基准测试，最多运行5个levels（便捷方法）
        Args:
            solver_class: BaseAgent 子类
            solver_kwargs: 创建 solver 实例的参数，通常包含 llm 模型配置
            world_concurrency: world 级别的并发数，默认为1（串行）
        """
        await self.execute(solver_class, solver_kwargs, world_concurrency, max_worlds=5)


    async def __call__(self, solver_class: Type[BaseAgent], solver_kwargs):
        """方便的调用接口"""
        await self.execute(solver_class, solver_kwargs)
