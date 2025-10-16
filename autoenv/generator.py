import os
import time
from typing import Optional
from pydantic import Field

from autoenv.prompt import (
    CRAFT_ENV_DESIGN_PROMPT,
    CRAFT_ENV_YAML_PROMPT,
    ECODE_AGENT_CODE_FIX_PROMPT,
    ECODE_AGENT_LEVEL_GENERATION_PROMPT,
    ECODE_AGENT_CALCULATE_MAX_REWARD_PROMPT,
    CRAFT_ENV_CODE_AND_INSTRUCTION_PROMPT,
    CRAFT_ENV_VALIDATOR_PROMPT,
    VALIDATOR_CHECKLIST
)
from base.agent.base_agent import BaseAgent
from autoenv.coder import ECodeAgent
from base.engine.logs import logger
from base.engine.async_llm import AsyncLLM
from base.engine.utils import read_file_content, write_file_content, parse_xml_content, archive_files

class Generator(BaseAgent):
    name: str = Field(default="Generator")
    description: str = Field(default="An agent designed for generating environments.")
    # Root directory to store generated environments
    envs_root_path: str = Field(default="workspace/envs")
    # Full path to the current environment folder (e.g., envs/<env_id>)
    env_folder_path: Optional[str] = Field(default=None)
    # Current environment id (time string)
    current_env_id: Optional[str] = Field(default=None)
    sub_code_agent: Optional[BaseAgent] = None # EcodeAgent can help refine the environments.
    re_llm: Optional[AsyncLLM] = Field(default=None)

    async def step(self):
        """
        Generator is a workflow, instead of an agent, we need to change the abstract.
        """
        pass

    def _ensure_env_folder_initialized(self, env_theme: str) -> None:
        """Ensure the environment folder path is initialized and created."""
        if not self.current_env_id:
            t = time.time()
            local_time = time.localtime(t)
            ms = int((t - int(t)) * 1_000_000)
            self.current_env_id = time.strftime("%Y%m%d_%H%M%S", local_time) + f"_env_{env_theme}"
        if not self.env_folder_path:
            self.env_folder_path = os.path.join(self.envs_root_path, self.current_env_id)
        os.makedirs(self.env_folder_path, exist_ok=True)

    async def craft_env_desc(self, requirements):
        env_desc_prompt = CRAFT_ENV_DESIGN_PROMPT.format(
            requirements=requirements,
        )
        resp = await self.re_llm(env_desc_prompt)
        env_desc_content = parse_xml_content(resp, "env_design")["env_design"]
        write_file_content(os.path.join(self.env_folder_path, "env_desc.txt"), env_desc_content)
        return env_desc_content

    async def craft_env_yaml(self, env_desc):
        env_yaml_prompt = CRAFT_ENV_YAML_PROMPT.format(
            env_desc=env_desc,
            config_yaml_example=read_file_content("autoenv/base/base_env_config.yaml"),
            environment_abstraction=read_file_content("autoenv/base/base_env.py"),
            observation_abstraction=read_file_content("autoenv/base/base_observation.py"),
            generator_abstraction=read_file_content("autoenv/base/base_generator.py")
        )
        resp = await self.re_llm(env_yaml_prompt)
        env_yaml_content = parse_xml_content(resp, "env_config")["env_config"]
        env_implement_help = parse_xml_content(resp, "env_implement_help")["env_implement_help"]
        write_file_content(os.path.join(self.env_folder_path, "config.yaml"), env_yaml_content)
        write_file_content(os.path.join(self.env_folder_path, "env_implement.txt"), env_implement_help)
        return env_yaml_content, env_implement_help
    
    async def craft_env_code_and_instruction(self, env_desc):
        env_code_and_instruction_prompt = CRAFT_ENV_CODE_AND_INSTRUCTION_PROMPT.format(
            env_desc=env_desc,
            config_yaml=read_file_content(os.path.join(self.env_folder_path, "config.yaml")),
            env_implement_help=read_file_content(os.path.join(self.env_folder_path, "env_implement.txt")),
            environment_abstraction=read_file_content("autoenv/base/base_env.py"),
            observation_abstraction=read_file_content("autoenv/base/base_observation.py"),
            generator_abstraction=read_file_content("autoenv/base/base_generator.py"),
            env_folder_path=self.env_folder_path
        )
        resp = await self.llm(env_code_and_instruction_prompt, max_tokens=32768)
        env_main_code_content = parse_xml_content(resp, "env_main_code")["env_main_code"]
        env_obs_code_content = parse_xml_content(resp, "env_obs_code")["env_obs_code"]
        env_generate_code_content = parse_xml_content(resp, "env_generate_code")["env_generate_code"]
        env_main_code_use_content = parse_xml_content(resp, "env_main_code_use")["env_main_code_use"]
        agent_instruction_content = parse_xml_content(resp, "agent_instruction")["agent_instruction"]
        action_space_content = parse_xml_content(resp, "action_space")["action_space"]
        write_file_content(os.path.join(self.env_folder_path, "env_main.py"), env_main_code_content)
        write_file_content(os.path.join(self.env_folder_path, "env_obs.py"), env_obs_code_content)
        write_file_content(os.path.join(self.env_folder_path, "env_generate.py"), env_generate_code_content)
        write_file_content(os.path.join(self.env_folder_path, "env_main_use.py"), env_main_code_use_content)
        write_file_content(os.path.join(self.env_folder_path, "agent_instruction.txt"), agent_instruction_content)
        write_file_content(os.path.join(self.env_folder_path, "action_space.txt"), action_space_content)
        return env_main_code_content, env_obs_code_content, env_generate_code_content, env_main_code_use_content, agent_instruction_content, action_space_content
    
    async def craft_env_validator(self, env_desc):
        validator_prompt = CRAFT_ENV_VALIDATOR_PROMPT.format(
            validator_checklist=VALIDATOR_CHECKLIST,
            env_desc=env_desc,
            config_yaml=read_file_content(os.path.join(self.env_folder_path, "config.yaml")),
            env_code=read_file_content(os.path.join(self.env_folder_path, "env_main.py")),
            observation_code=read_file_content(os.path.join(self.env_folder_path, "env_obs.py")),
            generator_code=read_file_content(os.path.join(self.env_folder_path, "env_generate.py"))
        )
        resp = await self.llm(validator_prompt, max_tokens=16384)
        env_validator_code_content = parse_xml_content(resp, "env_validator_code")["env_validator_code"]
        write_file_content(os.path.join(self.env_folder_path, "env_validator.py"), env_validator_code_content)
        return env_validator_code_content
    

    async def fix_env_code(self):
        """
        Use ECodeAgent to fix and validate core environment code structure.
        This is the first phase that ensures basic functionality works.
        """
        if not self.env_folder_path:
            raise ValueError("env_folder_path is not set")
        
        # Initialize ECodeAgent with LLM if not already set
        if not self.sub_code_agent:
            # Use the same LLM config as the Generator
            llm = AsyncLLM(self.llm.config)
            self.sub_code_agent = ECodeAgent(llm=llm)

        # Provide the env folder as the workspace to the ECodeAgent
        logger.info(f"Starting code fix process for environment: {self.current_env_id}")
        logger.info(f"Environment folder: {self.env_folder_path}")

        code_fix_task = ECODE_AGENT_CODE_FIX_PROMPT.format(
            env_id=self.current_env_id, 
            workspace=self.env_folder_path,
            validator_checklist=VALIDATOR_CHECKLIST,
        )
        result = await self.sub_code_agent(requirements=code_fix_task, cwds=self.env_folder_path)
        logger.info(f"Code fix completed. Result: {result}")
        return result

    async def generate_validated_levels(self):
        """
        Use ECodeAgent to generate comprehensive validated levels and perform integration testing.
        This is the second phase that builds on the fixed code to create working levels.
        """
        if not self.env_folder_path:
            raise ValueError("env_folder_path is not set")
        
        # Initialize ECodeAgent with LLM if not already set
        if not self.sub_code_agent:
            # Use the same LLM config as the Generator
            llm = AsyncLLM(self.llm.config)
            self.sub_code_agent = ECodeAgent(llm=llm)

        # Provide the env folder as the workspace to the ECodeAgent
        logger.info(f"Starting level generation process for environment: {self.current_env_id}")
        logger.info(f"Environment folder: {self.env_folder_path}")

        level_generation_task = ECODE_AGENT_LEVEL_GENERATION_PROMPT.format(
            env_id=self.current_env_id, 
            workspace=self.env_folder_path,
            validator_checklist=VALIDATOR_CHECKLIST,
        )
        result = await self.sub_code_agent(requirements=level_generation_task, cwds=self.env_folder_path)
        logger.info(f"Level generation completed. Result: {result}")
        return result
    
    async def calculate_max_rewards(self):
        """
        Use ECodeAgent to calculate the theoretical maximum reward for each generated level.
        This will analyze each level file and compute the optimal reward an agent could achieve.
        """
        if not self.env_folder_path:
            raise ValueError("env_folder_path is not set")
        
        # Initialize ECodeAgent with LLM if not already set
        if not self.sub_code_agent:
            # Use the same LLM config as the Generator
            llm = AsyncLLM(self.llm.config)
            self.sub_code_agent = ECodeAgent(llm=llm)

        # Provide the env folder as the workspace to the ECodeAgent
        logger.info(f"Starting max reward calculation for environment: {self.current_env_id}")
        logger.info(f"Environment folder: {self.env_folder_path}")

        calculation_task = ECODE_AGENT_CALCULATE_MAX_REWARD_PROMPT.format(
            env_id=self.current_env_id, 
            workspace=self.env_folder_path
        )
        result = await self.sub_code_agent(requirements=calculation_task, cwds=self.env_folder_path)
        logger.info(f"Max reward calculation completed. Result: {result}")
        return result
    
    def archive_files(self):
        """
        Clean up environment directory by archiving auxiliary files.
        Keeps only core environment files in the root directory.
        """
        if not self.env_folder_path:
            raise ValueError("env_folder_path is not set")
        
        return archive_files(self.env_folder_path, self.current_env_id)
    
    async def run(self, requirements):
        # Check if requirements is a file path and read it if so
        if isinstance(requirements, str) and requirements.endswith('.txt') and os.path.exists(requirements):
            env_theme = os.path.basename(requirements).split('.')[0]
            requirements = read_file_content(requirements)
        else:
            env_theme = "random"
        
        self._ensure_env_folder_initialized(env_theme)
        
        env_desc = await self.craft_env_desc(requirements)
        await self.craft_env_yaml(env_desc)
        await self.craft_env_code_and_instruction(env_desc)
        await self.craft_env_validator(env_desc)
        await self.fix_env_code()
        await self.generate_validated_levels()
        await self.calculate_max_rewards()
        
        # Clean up directory by archiving auxiliary files
        logger.info("Cleaning up environment directory...")
        self.archive_files()
        
        return self.env_folder_path
        