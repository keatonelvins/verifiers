"""Train a language model using Verifiers for reinforcement learning.
modal run train.py --config verifiers/configs/mm-qwen2.5-coder-1.5b.yaml
"""
import os
import modal
import yaml
import sys
from pathlib import Path

# Volume names for persistent storage
DATASET_VOLUME_NAME = "verifiers-dataset"
MODEL_VOLUME_NAME = "models"

# Create a Modal image with all dependencies
verifiers_image = modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11") \
    .apt_install(["git", "wget"]) \
    .pip_install(["uv"]) \
    .run_commands([
        "uv pip install --system wheel setuptools",
        "uv pip install --system torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "uv pip install --system psutil deepspeed accelerate peft wandb duckduckgo-search",
        "uv pip install --system trl@git+https://github.com/huggingface/trl.git",
        "uv pip install --system liger-kernel>=0.5.2 vllm==0.7.3",
        "uv pip install --system flash-attn --no-build-isolation",
        "mkdir -p /root/boom/verifiers/configs/deepspeed",
        "mkdir -p /root/boom/verifiers/configs/accelerate",
        "wget -P /root/boom/verifiers/configs/deepspeed https://gist.githubusercontent.com/kalomaze/36b71be9f81151e1a6e2c6d8332c84ad/raw/e9f4b9352829929c6ac16492a4d4663a76ea8bc3/stage3.json",
        "wget -P /root/boom/verifiers/configs/accelerate https://gist.githubusercontent.com/kalomaze/2562e68017be328eefdedf930d39e8be/raw/11281bc48254206cc47d01299bbdce3d1c6d2add/deepspeed.yaml"
    ])

app = modal.App("verifiers-training")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)

# Configure resources
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
N_GPUS = int(os.environ.get("N_GPUS", 8))
N_HOURS = int(os.environ.get("N_HOURS", 10))

@app.function(
    image=verifiers_image.add_local_dir(".", remote_path="/root/boom/verifiers"),
    gpu="H100:8",
    volumes={
        "/dataset": dataset_volume,
        "/models": model_volume, 
    },
    timeout=N_HOURS * HOURS,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
)
def run_train(config_path: str, n_gpus: int):
    """Run the training with the specified config."""
    import subprocess
    import sys
    
    # Adjust the config path to look in the mounted directory
    if not os.path.isabs(config_path):
        remote_config_path = f"/root/boom/verifiers/{config_path}"
    else:
        remote_config_path = config_path
    
    # Load the config
    with open(remote_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Save config to the verifiers directory
    config_name = Path(remote_config_path).stem
    verifiers_config_path = f"/root/boom/verifiers/configs/{config_name}.yaml"
    with open(verifiers_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Set up environment variables
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    
    # Set WANDB project name if specified in config
    if "wandb" in config:
        if "project" in config["wandb"]:
            os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
    
    # Use one less GPU for training to reserve one for VLLM generations
    training_gpus = n_gpus - 1
    
    # Run the training command
    os.chdir("/root/boom/verifiers")
    script_path = config.get('script_path')
    
    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        print(f"Current directory: {os.getcwd()}")
        print(os.listdir("."))
        raise FileNotFoundError(f"Script {script_path} not found")
    
    command = f"accelerate launch --config_file configs/accelerate/deepspeed.yaml --num_processes {training_gpus} {script_path}"
    
    # Download the model to the volume
    model_name = config.get("script_args", {}).get("model_name")
    print(f"Downloading model {model_name} to {MODEL_VOLUME_NAME}")
    model_volume.reload()

    # Download the adapter/model
    from huggingface_hub import snapshot_download
    snapshot_download(
        model_name,
        local_dir=f"/{MODEL_VOLUME_NAME}/{model_name}",
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        revision=config.get("revision", "main"),
        force_download=config.get("force_download", False),
    )
    
    config["script_args"]["model_name"] = f"/{MODEL_VOLUME_NAME}/{model_name}"
    
    # Add any additional arguments from config
    if "script_args" in config:
        for key, value in config["script_args"].items():
            command += f" --{key} {value}"
    
    print(f"Running command: {command}")
    subprocess.run(
        command.split(),
        stdout=sys.stdout, stderr=sys.stderr,
        check=True,
    )
    
    # Save the trained model
    model_volume.commit()


@app.local_entrypoint()
def main(config: str):
    """Entry point for the Modal app.
    
    Args:
        config: Either the full path to a config file or just the config name 
               (without .yaml extension) to look for in verifiers/configs/
    """
    # Check if config is just a name or a full path
    if not config.endswith('.yaml'):
        config_path = f"verifiers/configs/{config}.yaml"
    else:
        config_path = config
    
    # Validate config path
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run the training
    run_train.remote(config_path=config_path, n_gpus=N_GPUS)