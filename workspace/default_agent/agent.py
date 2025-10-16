import json
import ast
from pydantic import Field
from typing import Dict, List, Any, Tuple

from base.agent.base_agent import BaseAgent
from base.engine.utils import parse_xml_content
from base.env.base_env import SkinEnv
from base.engine.logs import logger

from .prompt import LEARNED_INSTRUCTION_PROMPT, AGENT_ACT_PROMPT

MAX_PAST_ACTIONS = None

class ReActAgent(BaseAgent):
    """
    Solver Agent is used to recive different environment desc and action space desc.
    And then solve the environment.
    """
    name: str = Field(default="sovler")
    description: str = Field(default="A solver agent for solving the environment.")
    current_environment_total_reward: int = Field(default=0)
    current_action_space: str = Field(default="")
    past_actions: List[Dict[str, Any]] = Field(default_factory=list)
    trajectory_folder_path: str = Field(default="")
    

    # ==== Basic Functions ====

    def parse_action(self, resp: str) -> Dict[str, Any]:
        """Parse LLM response to extract action data.

        Supports two formats:
        - Preferred: content inside <action>...</action> as JSON/Python-literal
        - Fallback: legacy formats handled by parse_llm_action_response
        """
        # Empty/None guard
        if not resp:
            logger.warning("Received None or empty response from LLM")
            return {"action": "no_action", "params": {}, "_parse_error": "Empty LLM response"}

        # 1) Try extracting from <action>...</action>
        try:
            action_block = parse_xml_content(resp, "action").get("action")
            action_str = action_block[-1] if isinstance(action_block, list) else action_block
            if action_str:
                # Strip fenced code if present
                if "```" in action_str:
                    start = action_str.find("```")
                    if start != -1:
                        start += 3
                        end = action_str.find("```", start)
                        if end != -1:
                            action_str = action_str[start:end].strip()
                        else:
                            action_str = action_str[start:].strip()

                # Try JSON first, then Python literal
                parsed = None
                try:
                    parsed = json.loads(action_str)
                except Exception:
                    try:
                        parsed = ast.literal_eval(action_str)
                    except Exception:
                        parsed = None

                # If list returned, take the first element
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else None

                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
                else:
                    msg = f"Invalid <action> content parsed: {parsed}"
                    logger.warning(msg)
                    return {"action": "Invalid", "params": {}, "_parse_error": msg}
            else:
                msg = "No <action> tag found"
                logger.warning(msg)
                return {"action": "Invalid", "params": {}, "_parse_error": msg}
        except Exception as e:
            msg = f"Failed to parse <action> tag: {type(e).__name__}: {e}"
            logger.warning(msg)
            return {"action": "Invalid", "params": {}, "_parse_error": msg}

    def _resolve_max_steps(self, env: SkinEnv, env_info: Dict) -> int:
        """
        Resolve step limit with precedence:
        1) If env_info provides max_step (not None), use it.
        2) Else, fallback to env.configs termination.max_steps.
        3) Else, default to 20.
        """
        # 1) Explicit override from caller
        if "max_step" in env_info and env_info.get("max_step") is not None:
            try:
                return int(env_info.get("max_step"))
            except (TypeError, ValueError):
                pass

        # 2) Config fallbacks
        configs = getattr(env, "configs", {}) if getattr(env, "configs", None) else {}
        term_steps = configs.get("termination", {}).get("max_steps")
        if isinstance(term_steps, int):
            return term_steps
        try:
            if term_steps is not None:
                return int(term_steps)
        except (TypeError, ValueError):
            pass

        # 3) Final default
        return 20


    def _get_recent_actions(self):
        """
        Get recent actions in a readable format.
        """
        lines = []

        actions_to_show = self.past_actions if MAX_PAST_ACTIONS is None else self.past_actions[-MAX_PAST_ACTIONS:]
        for i, entry in enumerate(actions_to_show):
            if isinstance(entry, dict) and "action" in entry:
                thought_part = entry.get("thought")
                action_part = entry.get("action")
                result_part = entry.get("result")
                # Compact result representation
                if isinstance(result_part, (dict, list)):
                    try:
                        result_str = json.dumps(result_part, ensure_ascii=False)
                    except Exception:
                        result_str = str(result_part)
                else:
                    result_str = str(result_part)
                lines.append(f"{i+1}. {thought_part} \n {action_part} -> {result_str}")
            else:
                lines.append(f"{i+1}. {entry}")
        return "\n".join(lines)


    # ==== Step Function, You Don't need to modify this =====
    async def step(self, agent_obs: Dict) -> Tuple[Dict, str]:
        act_prompt = AGENT_ACT_PROMPT.format(
            env_instruction=LEARNED_INSTRUCTION_PROMPT,
            action_space=self.current_action_space,
            obs=agent_obs,
            recent_actions=self._get_recent_actions(),
        )
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = None

        action = self.parse_action(resp)

        # First try to extract thinking_memory tag content
        if resp:
            thinking_memory_content = parse_xml_content(resp, "thinking_memory")
            if thinking_memory_content.get("thinking_memory"):
                thought = thinking_memory_content["thinking_memory"]
            # Fallback to original extraction logic if no thinking_memory tag found
            elif '```json' in resp:
                thought = resp.split('```json')[0].strip()
            elif '```' in resp:
                thought = resp.split('```')[0].strip()
            else:
                thought = resp.strip()
        else:
            thought = "No response from LLM"
        
        if thought:
            logger.agent_thinking(f"Agent Thought: {thought}")

        logger.agent_action(f"Agent Action: {action}")

        return action, thought


    # ===== Run Function, You Don't need to modify this =====

    async def run(self, env:SkinEnv, env_info:Dict):
        """
        env info:
        {
            "world_id": str,
            "agent_instruction": str,
            "action_space": str,
            "max_step": int,
        }
        """
        self.past_actions = []
        
        world_id = env_info["world_id"]
        env.reset(mode="load", world_id=world_id)

        self.current_action_space = env_info["action_space"]

        # Resolve step limit with override-first precedence
        max_step = self._resolve_max_steps(env, env_info)
        cur_steps = 0
        cur_reward = 0
        events_count = {}
        
        raw_obs = env.observe_semantic()
        agent_obs = env.render_skin(raw_obs)
        initial_observation = agent_obs

        # Execute in Environments
        while cur_steps < max_step and not env.done():
            logger.info(f"Environment Observation: \n{agent_obs}")
            action, thought = await self.step(agent_obs)
            _, reward, done, info = env.step(action)
            # Record action along with last action result for better context
            self.past_actions.append({
                "action": action,
                "thought": thought,
                "observation": agent_obs,
                "result": info.get("last_action_result"),
                "events": info.get("events", []),
                "reward": reward,
                "parse_error": (action or {}).get("_parse_error"),
            })
            cur_reward += reward
            agent_obs = info["skinned"]
            for e in info.get("events", []):
                events_count[e] = events_count.get(e, 0) + 1
            cur_steps += 1
            if done:
                break

        return {
            "total_reward": cur_reward,
            "events_count": events_count,
            "step": cur_steps,
            "initial_observation": initial_observation,
        }
        
