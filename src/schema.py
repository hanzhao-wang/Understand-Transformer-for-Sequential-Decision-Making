import wandb
from funcy import merge
from quinine import (
    allowed,
    default,
    nullable,
    required,
    stdict,
    tboolean,
    tdict,
    tfloat,
    tinteger,
    tlist,
    tstring,
)

wandb.login(key="<YOUR_API_KEY_HERE>")


task_schema = {
    "task_name": merge(tstring, required),
    "exp_name": merge(tstring, required),
    "data_name": merge(tstring, required),
    "gen_method": merge(tstring, required),
    "state_dim": merge(tinteger, required),
    "act_value_dim": merge(tinteger, required),
    "act_output_dim": merge(tinteger, required),
    "action_type": merge(tstring, nullable, default("continuous")),
    "params": merge(tdict, required),
    "fine_tune": merge(tboolean, nullable, default(False)),
    "FT_gen_data": merge(tinteger, nullable, default(0)),
}
eval_schema = {
    "num_eval_episodes": merge(tinteger, required),
    "train_horizon": merge(tinteger, required),
    "test_horizon": merge(tinteger, required),
    "temp": merge(tfloat, nullable, default(1.0)),
    "need_env_info": merge(tboolean, nullable, default(False)),
    "for_demand_pred": merge(tboolean, nullable, default(False)),
    "output_attention": merge(tboolean, nullable, default(False)),
    "output_hidden_states": merge(tboolean, nullable, default(False)),
    "output_probs": merge(tboolean, nullable, default(False)),
    "warm_up_steps": merge(tlist, nullable, default([0])),
    "warm_up_pure_random": merge(tboolean, nullable, default(True)),
    "scaling_reward": merge(tboolean, nullable, default(False)),
}


model_schema = {
    "model_type": merge(tstring, allowed(["DT", "ICL", "DPT"])),
    "hidden_size": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "dropout": merge(tfloat, nullable, default(0.05)),
    "window_len": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "need_time_embed": merge(tboolean, nullable, default(False)),
}

train_schema = {
    "batch_size": merge(tinteger, default(64)),
    "num_train_iters": merge(tinteger, default(100)),
    "num_steps_per_iter": merge(tinteger, default(1000)),
    "train_data_horizon": merge(tinteger, required),
    "loss_fn": merge(tstring, required),
    "warmup_steps": merge(tinteger, default(3000)),
    "lr": merge(tfloat, default(1e-4)),
    "weight_decay": merge(tfloat, default(1e-4)),
    "seed": merge(tinteger, default(1122)),
    "eval_last_regs_len": merge(tinteger, default(30)),
    "curr": merge(tdict, nullable, default(None)),
    "FT_time": merge(tinteger, nullable, default(0)),
}


wandb_schema = {
    "project": merge(tstring, default("<PROJECT_NAME_HERE>")),
    "entity": merge(tstring, default("<ENTITY_NAME_HERE>")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
}

multigpu_schema = {
    "dist_on_itp": merge(tboolean, nullable, default(False)),
    "dist_url": merge(tstring, nullable, default("env://")),
    "dist_backend": merge(tstring, nullable, default("nccl")),
}

schema = {
    "model": stdict(model_schema),
    "task": stdict(task_schema),
    "train": stdict(train_schema),
    "eval": stdict(eval_schema),
    "wandb": stdict(wandb_schema),
    "multigpu": stdict(multigpu_schema),
}
