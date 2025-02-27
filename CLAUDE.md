# Verifiers Project Guide

## Commands
- Run a test: `python tests/test_tools.py` or `python tests/test_xmlparser.py`
- Run a specific example: `python verifiers/examples/gsm8k_simple.py`
- Training: `python train.py --config verifiers/configs/mm-qwen2.5-coder-7b.yaml`
- Accelerate launch: `accelerate launch --config_file configs/accelerate/deepspeed.yaml --num_processes [N-1] script.py`
- Torchrun: `torchrun --nproc_per_node=[N-1] script.py`

## Code Style
- Imports: Standard library → third-party → local modules
- Type annotations: Use throughout codebase (`List[Dict[str, str]]`)
- Naming: `snake_case` for variables/functions, `PascalCase` for classes
- Error handling: Use explicit try/except blocks with specific exceptions
- Environment classes: Implement required methods from base classes
- XML parsing: Use the XMLParser class for structured data extraction
- Tools: Properly document tool usage in docstrings
- Function signatures: Include return type annotations

## Project Structure
- Environments (`envs/`): Core environment implementations
- Tools (`tools/`): External tools like calculator, search
- Rubrics (`rubrics/`): Reward functions for environments
- Examples: Reference implementations for each environment