import argparse
import copy
import random
from collections import OrderedDict

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from quinine import Quinfig

from evaluation.evaluate_episodes import evaluate_episode
from model.DPT_Transformer import DecisionPretrainedTransformer
from model.ICL_Transformer import GPT2_ICL
from model.OM_Transformer import GPT2_OM
from schema import schema
from training.seq_trainer import SequenceTrainer
from utils.loss_fun import (
    CE_loss,
    CE_loss_MLP,
    CE_pure_loss,
    Huber_loss,
    Huber_loss_pure,
    L1_loss,
    L1_loss_pure,
    L2_loss,
    L2_pure_loss,
)
from utils.multigpu_utils import init_distributed_mode

torch.backends.cudnn.benchmark = True


def train(args):
    warmup_steps_list = args.eval.warm_up_steps

    batch_size = args.train.batch_size
    num_train_iters = args.train.num_train_iters
    num_steps_per_iter = args.train.num_steps_per_iter
    max_ep_len = args.train.train_data_horizon
    loss_dict = {
        "L2": L2_loss,
        "L2_pure": L2_pure_loss,
        "CE": CE_loss,
        "CE_pure": CE_pure_loss,
        "CE_MLP": CE_loss_MLP,
        "L1": L1_loss,
        "L1_pure": L1_loss_pure,
        "Huber": Huber_loss,
        "Huber_pure": Huber_loss_pure,
    }
    loss_fn = loss_dict[args.train.loss_fn]
    warmup_steps = args.train.warmup_steps
    learning_rate = args.train.lr
    weight_decay = args.train.weight_decay
    seed = args.train.seed
    eval_last_regs_len = args.train.eval_last_regs_len
    FT_time = args.train.FT_time

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
    # set random seed
    seed = (
        seed
        if not args.multigpu.distributed
        else seed + args.multigpu.local_rank
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    else:
        raise NotImplementedError

    if not args.multigpu.distributed:
        model.cuda()
    else:
        model.to(device=torch.device("cuda", args.multigpu.local_rank))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.multigpu.local_rank],
            output_device=args.multigpu.local_rank,
            find_unused_parameters=True,
        )

    if args.task.fine_tune:
        if not args.multigpu.distributed:
            state_dict = torch.load(
                "../../model_logs/"
                + args.task.task_name
                + "/"
                + args.task.exp_name
                + "/"
                + args.task.gen_method
                + "/best_model.pth"
            )
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(
                torch.load(
                    "../../model_logs/"
                    + args.task.task_name
                    + "/"
                    + args.task.exp_name
                    + "/"
                    + args.task.gen_method
                    + "/best_model.pth"
                )
            )
        print("load the model for fine-tune")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    def eval_episodes(args, warm_up_k):
        eval_args_copy = copy.deepcopy(args.eval)
        eval_args_copy.warm_up_steps = warm_up_k

        def fn(model):
            logs = dict()
            num_eval_episodes = args.eval.num_eval_episodes

            regs, cum_regs, acts, opt_acts = [], [], [], []
            (
                states_list,
                act_ids_list,
                act_values_list,
                rewards_list,
                opt_acts_list,
            ) = ([], [], [], [], [])
            gen_data = {}
            for i in range(num_eval_episodes):
                with torch.no_grad():
                    (
                        _,
                        states,
                        act_ids,
                        act_values,
                        rewards,
                        opt_act_id,
                        cum_reg,
                        opt_action,
                        actions_list,
                    ) = evaluate_episode(
                        args.task,
                        eval_args_copy,
                        model,
                        history_data=None,
                        test_env_cov=None,
                    )
                cum_reg = np.mean(cum_reg, axis=0)
                regs.append(cum_reg[-1])
                cum_regs.append(cum_reg)
                acts.append(actions_list)
                opt_acts.append(opt_action)
                if FT_time > 0:
                    states_list.append(states.detach().cpu().numpy())
                    act_ids_list.append(act_ids.detach().cpu().numpy())
                    act_values_list.append(act_values.detach().cpu().numpy())
                    rewards_list.append(rewards.detach().cpu().numpy())
                    opt_acts_list.append(opt_act_id)
            if warm_up_k == 0:
                plt.figure()
                plt.plot(acts[0][0], label="act")
                plt.plot(np.array(opt_acts[0][0]), label="opt")
                plt.legend()
                plt.show()
                plt.savefig("eval_action_plot" + args.task.exp_name + ".png")
                plt.close()
                plt.figure()
                plt.plot(np.mean(np.array(cum_regs), axis=0), label="regret")
                plt.legend()
                plt.show()
                plt.savefig("eval_regret_plot" + args.task.exp_name + ".png")
                plt.close()
            task_log = {
                f"regret_mean_" + str(warm_up_k): np.mean(regs),
                f"regret_std_" + str(warm_up_k): np.std(regs),
                f"last"
                + str(eval_last_regs_len)
                + "_regret_mean_"
                + str(warm_up_k): np.mean(
                    np.mean(np.array(cum_regs), axis=0)[-eval_last_regs_len:]
                    - np.mean(np.array(cum_regs), axis=0)[
                        -eval_last_regs_len - 1 : -1
                    ]
                ),
                f"last"
                + str(eval_last_regs_len)
                + "_action_error_mean_"
                + str(warm_up_k): np.mean(
                    np.mean(
                        np.abs(np.array(acts[0] - np.array(opt_acts)[0])),
                        axis=0,
                    )[-eval_last_regs_len:]
                ),
            }
            gen_data = {
                "states_list": np.concatenate(states_list, axis=0),
                "act_ids_list": np.concatenate(act_ids_list, axis=0),
                "act_values_list": np.concatenate(act_values_list, axis=0),
                "rewards_list": np.concatenate(rewards_list, axis=0),
                "opt_acts_list": np.concatenate(opt_acts_list, axis=0),
            }
            logs.update(task_log)

            return logs, gen_data

        return fn

    trainer = SequenceTrainer(
        args.task,
        args.multigpu,
        model,
        target_len,
        optimizer,
        batch_size,
        loss_fn,
        scheduler,
        eval_fns=None,
        curr=args.train.curr,
        FT_time=FT_time,
        model_type=model_type,
    )

    trainer.eval_fns = [eval_episodes(args, k) for k in warmup_steps_list]
    for iter in range(num_train_iters):
        outputs = trainer.train_iteration(
            num_steps=num_steps_per_iter, iter_num=iter + 1, print_logs=True
        )
        if iter == FT_time - 3:
            num_steps_per_iter = 50
            args.eval.num_eval_episodes = 10
            # args.eval.num_eval_episodes=num_steps_per_iter*2

        if (not args.multigpu.distributed) or (args.multigpu.local_rank == 0):

            wandb.log(
                outputs,
                step=iter,
            )


def main(args):
    if (not args.multigpu.distributed) or (args.multigpu.local_rank == 0):

        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=False,
        )

    train(args)

    if (not args.multigpu.distributed) or (args.multigpu.local_rank == 0):

        wandb.finish()


def light_parser():

    parser = argparse.ArgumentParser(
        description="a light parser that leverages quinine parse"
    )
    parser.add_argument(
        "--quinine_config_path",
        type=str,
    )
    parser.add_argument("--local_rank", default=0, type=int)

    return parser


if __name__ == "__main__":

    raw_args = light_parser().parse_args()

    args = Quinfig(config_path=raw_args.quinine_config_path, schema=schema)

    init_distributed_mode(args.multigpu)

    print(f"Running with: {args}")

    main(args)
