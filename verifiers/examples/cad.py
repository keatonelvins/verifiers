import os
import verifiers as vf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model using Verifiers for CAD tasks")
    parser.add_argument("--dataset", type=str, default="mm", help="Dataset to use for training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B", help="Model name to use for training")
    parser.add_argument("--run_name", type=str, default="mm-qwen2.5-coder-1.5b", help="Name for the training run")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_name = args.model_name
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    
    vf_env = vf.CadEnv(dataset=args.dataset)
    dataset = vf_env.get_dataset()
    rubric = vf_env.get_rubric()
    
    # Set output directory to the mounted volume location
    output_dir = f"/models/{args.run_name}"
    training_args = vf.get_default_grpo_config(
        run_name=args.run_name, 
        num_gpus=args.num_gpus,
        output_dir=output_dir
    )
    
    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rubric, 
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()