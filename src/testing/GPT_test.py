import contextlib
import copy
from collections import OrderedDict

import numpy as np
import torch

from evaluation.evaluate_episodes import evaluate_episode
from model.DPT_Transformer import DecisionPretrainedTransformer
from model.ICL_Transformer import GPT2_ICL
from model.OM_Transformer import GPT2_OM


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def test_model(
    args,
    test_env_cov=None,
    history_data=None,
    verbose=True,
    batch_size=100,
    model_name=None,
):
    warmup_steps_list = args.eval.warm_up_steps
    max_ep_len = args.train.train_data_horizon

    target_len = args.model.window_len
    model_type = args.model.model_type
    hidden_size = args.model.hidden_size
    nn_layer = args.model.n_layer
    dropout_rate = args.model.dropout
    need_time_embed = args.model.need_time_embed
    # args.train.model.n_head

    obs_dim = args.task.state_dim
    act_value_dim = args.task.act_value_dim
    act_output_dim = args.task.act_output_dim
    action_type = args.task.action_type

    if model_type == "DT":
        model = GPT2_OM(
            obs_dim,
            act_value_dim,
            act_output_dim,
            hidden_size,
            max_ep_len,
            target_len,
            need_time_embed=need_time_embed,
            n_layer=nn_layer,
            n_head=args.model.n_head,
            n_inner=4 * hidden_size,
            resid_pdrop=dropout_rate,
            attn_pdrop=dropout_rate,
        )
    elif model_type == "ICL":
        model = GPT2_ICL(
            obs_dim,
            act_value_dim,
            act_output_dim,
            action_type,
            hidden_size,
            max_ep_len,
            target_len,
            need_time_embed=need_time_embed,
            n_layer=nn_layer,
            n_head=args.model.n_head,
            n_inner=4 * hidden_size,
            resid_pdrop=dropout_rate,
            attn_pdrop=dropout_rate,
        )
    elif model_type == "DPT":
        model = DecisionPretrainedTransformer(
            obs_dim,
            act_value_dim,
            act_output_dim,
            action_type,
            hidden_size,
            nn_layer,
            args.model.n_head,
            4 * hidden_size,
            dropout_rate,
            dropout_rate,
            max_ep_len,
            target_len,
            need_time_embed,
            mask_act_idx=True,
        )

    if not args.multigpu.distributed:
        model.cuda()
        if model_name is None:
            model_name = "best_model_" + model_type + ".pth"
        state_dict = torch.load(
            "../model_logs/"
            + args.task.task_name
            + "/"
            + args.task.exp_name
            + "/"
            + args.task.gen_method
            + "/"
            + model_name
        )
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print("not support multi_GPU")

    eval_args = args.eval
    task_args = args.task
    eval_args_copy = copy.deepcopy(args.eval)
    eval_args_copy.warm_up_steps = warmup_steps_list[0]
    with torch.no_grad():
        (
            extra,
            state,
            act_ids,
            act_values,
            reward,
            opt_act_id,
            cum_reg,
            opt_action,
            actions_list,
        ) = evaluate_episode(
            task_args,
            eval_args_copy,
            model,
            testing_mode=True,
            test_env_cov=test_env_cov,
            history_data=history_data,
            batch_size=batch_size,
        )
        extras = extra
        opt_act = opt_action
        regs = cum_reg[:, -1]
        acts = actions_list

        cum_regs = cum_reg
        states = state
        rewards = reward

    return extras, states, rewards, acts, opt_act, regs, cum_regs
