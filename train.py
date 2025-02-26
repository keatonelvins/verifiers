import verifiers as vf

model_name = "Qwen/Qwen2.5-Coder-1.5B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.CadEnv(dataset="mm")
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(run_name="mm-qwen2.5-coder-1.5b", num_gpus=1)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()