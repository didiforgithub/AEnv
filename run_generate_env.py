#!/usr/bin/env python3
"""
Environment generation script - supports single and parallel generation modes
"""
import os
import asyncio
import traceback
import yaml
from autoenv.generator import Generator
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger


def load_config():
    """Load configuration from env_gen.yaml"""
    with open("config/env_gen.yaml", "r") as f:
        return yaml.safe_load(f)


def get_requirement_files(folder_path):
    """Get all .txt files from folder"""
    return [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.endswith(".txt") and os.path.isfile(os.path.join(folder_path, fname))
    ]


async def generate_env(requirement_path, llm, envs_root_path="workspace/envs"):
    """Generate a single environment from requirement file"""
    try:
        generator = Generator(llm=llm, envs_root_path=envs_root_path)
        print(f"🚀 Starting: {requirement_path}")
        await generator.run(requirement_path)
        print(f"✅ Finished: {requirement_path}")
    except Exception as e:
        print(f"❌ Error: {requirement_path} - {e}")
        traceback.print_exc()


async def run_single_mode(config, llm, envs_root_path):
    """Run single environment generation"""
    requirement_path = config.get("requirement_path")
    if not requirement_path:
        return print("❌ Error: requirement_path not specified")
    if not os.path.exists(requirement_path):
        return print(f"❌ Error: file not found: {requirement_path}")

    print(f"📄 Processing: {requirement_path}")
    await generate_env(requirement_path, llm, envs_root_path)


async def run_parallel_mode(config, llm, envs_root_path):
    """Run parallel environment generation"""
    requirements_folder = config.get("requirements_folder")
    if not requirements_folder:
        return print("❌ Error: requirements_folder not specified")
    if not os.path.exists(requirements_folder):
        return print(f"❌ Error: folder not found: {requirements_folder}")

    requirement_files = get_requirement_files(requirements_folder)
    if not requirement_files:
        return print("❌ Error: no .txt files found in folder")

    concurrency = config.get("count", 2)
    print(f"📄 Found {len(requirement_files)} files")
    print(f"🔢 Concurrency: {concurrency}")

    sem = asyncio.Semaphore(concurrency)

    async def task(path):
        async with sem:
            await generate_env(path, llm, envs_root_path)

    await asyncio.gather(*[task(f) for f in requirement_files])


async def main():
    """Main entry point"""
    config = load_config()
    mode = config.get("mode", "single")
    llm_name = config.get("llm")
    envs_root_path = "workspace/envs"

    # Initialize
    os.makedirs(envs_root_path, exist_ok=True)
    llm = create_llm_instance(LLMsConfig.default().get(llm_name))

    print(f"🔧 Mode: {mode}")
    print(f"🚀 LLM: {llm_name}")
    print(f"📁 Output: {envs_root_path}")

    # Run based on mode
    if mode == "single":
        await run_single_mode(config, llm, envs_root_path)
    elif mode == "parallel":
        await run_parallel_mode(config, llm, envs_root_path)
    else:
        print(f"❌ Error: Unknown mode '{mode}'. Use 'single' or 'parallel'")


if __name__ == "__main__":
    asyncio.run(main())
