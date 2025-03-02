import logging
import sys
from typing import Optional
import datetime
import json

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
import rich
from rich.syntax import Syntax

def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.
    
    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False 


def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[dict],
    rewards: list[float],
    step: int,
    width: Optional[int] = None,
) -> None:

    console = Console(width=width) if width else Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):
        # Create a formatted Text object for completion with alternating colors based on role
        formatted_completion = Text()
        
        if isinstance(completion, dict):
            # Handle single message dict
            role = completion.get("role", "")
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            # Handle list of message dicts
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")
                
                role = message.get("role", "")
                content = message.get("content", "")
                
                # Set style based on role
                style = "bright_cyan" if role == "assistant" else "bright_magenta"
                
                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
        else:
            # Fallback for string completions
            formatted_completion = Text(str(completion))

        table.add_row(Text(prompt), formatted_completion, Text(f"{reward:.2f}"))
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
    
    
def print_thought_completion_truth_reward_sample(
    thoughts: list[str],
    completions: list[str],
    truths: list[str],
    rewards: list[float],
    step: int,
    width: Optional[int] = None,
) -> None:
    """
    Print a nicely formatted table showing thoughts, completions, truths, and rewards.
    
    Args:
        thoughts: List of thought strings
        completions: List of completion strings
        truths: List of truth strings
        rewards: List of reward values
        step: Current step number
        width: Optional width override for the console (useful in modal environments)
    """
    console = Console(width=width) if width else Console()
    table = Table(show_header=True, header_style="bold white", box=rich.box.ROUNDED, expand=True)

    # Add columns with better styling
    table.add_column("Thought", style="bright_yellow", overflow="fold")
    table.add_column("Action", style="bright_green", overflow="fold")
    table.add_column("Truth", style="bright_magenta", overflow="fold")
    table.add_column("Reward", style="bold cyan", justify="right", width=10)

    # Add a caption to explain the data
    table.caption = Text("Sample of model thoughts, actions, ground truths, and reward values", 
                         style="dim italic")

    for thought, completion, truth, reward in zip(thoughts, completions, truths, rewards, strict=True):
        # Truncate long texts for better display
        if thought is None:
            thought = ""
        if completion is None:
            completion = ""

        thought_text = Text(thought[:500] + ("..." if len(thought) > 500 else ""))
        
        # Format completion as JSON with Rich's syntax highlighting if possible
        try:
            try:
                parsed_completion = json.loads(completion)
                json_str = json.dumps(parsed_completion, indent=2)
                completion_text = Syntax(json_str, "json", theme="monokai", word_wrap=True)
            except (json.JSONDecodeError, TypeError):
                completion_text = Text(str(completion))
        except Exception:
            completion_text = Text(str(completion))

        # Format truth as JSON with Rich's syntax highlighting
        parsed_truth = json.loads(truth)
        json_str = json.dumps(parsed_truth, indent=2)
        truth_text = Syntax(json_str, "json", theme="monokai", word_wrap=True)
        
        # Format reward with color based on value
        reward_style = "green" if reward > 0 else "red"
        reward_text = Text(f"{reward:.2f}", style=reward_style)
        
        table.add_row(thought_text, completion_text, truth_text, reward_text)
        table.add_section()  # Adds a separator between rows

    panel = Panel(
        table,
        title=f"[bold]Step {step} Results[/bold]", 
        border_style="bold blue",
        subtitle=f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    console.print("\n")  # Add some spacing
    console.print(panel)
    console.print("\n")  # Add some spacing
    