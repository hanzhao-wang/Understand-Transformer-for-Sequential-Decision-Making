import time

import matplotlib.pyplot as plt
import numpy as np

from envs.Bandits_gen import LinB_sample_multi, MAB_sample_multi
from envs.DP_gen_cts import DP_cts_sample_multi
from envs.NV_gen_cts import NV_ctx_sample_multi_cts

DP_continuous_dict = {
    "inf_4d_123": {
        "exp_name": "inf_4d_123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 0,
        "seed": 123,
        "need_square": False,
    },
    "16env_4d_sq123": {
        "exp_name": "16env_4d_sq123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 16,
        "seed": 123,
        "need_square": True,
    },
    "inf_4d_sq123": {
        "exp_name": "inf_4d_sq123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 0,
        "seed": 123,
        "need_square": True,
    },
    "inf_2d_sq123": {
        "exp_name": "inf_2d_sq123",
        "gen_method": "Perturb",
        "dim_num": 2,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 0,
        "seed": 123,
        "need_square": True,
    },
    "100env_4d_123": {
        "exp_name": "100env_4d_123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 100,
        "seed": 123,
        "need_square": False,
    },
    "4env_4d_123": {
        "exp_name": "4env_4d_123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 4,
        "seed": 123,
        "need_square": False,
    },
    "4env_4d_132": {
        "exp_name": "4env_4d_132",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 4,
        "seed": 132,
        "need_square": False,
    },
    "4env_4d_231": {
        "exp_name": "4env_4d_231",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 10,
        "num_samp": 1000000,
        "horizon": 100,
        "err_std": 0.2,
        "finite_envs_num": 4,
        "seed": 231,
        "need_square": False,
    },
}


NV_continuous_dict = {
    "4env_4d_123": {
        "exp_name": "4env_4d_123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 30,
        "num_samp": 1000000,
        "horizon": 100,
        "perishable": True,
        "censor": False,
        "finite_envs_num": 4,
        "seed": 123,
    },
    "inf_4d_123": {
        "exp_name": "inf_4d_123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 30,
        "num_samp": 1000000,
        "horizon": 100,
        "perishable": True,
        "censor": False,
        "finite_envs_num": 0,
        "seed": 123,
    },
    "inf_2d_123": {
        "exp_name": "inf_2d_123",
        "gen_method": "Perturb",
        "dim_num": 2,
        "action_ub": 30,
        "num_samp": 1000000,
        "horizon": 100,
        "perishable": True,
        "censor": False,
        "finite_envs_num": 0,
        "seed": 123,
    },
    "100env_4d_123": {
        "exp_name": "100env_4d_123",
        "gen_method": "Perturb",
        "dim_num": 4,
        "action_ub": 30,
        "num_samp": 1000000,
        "horizon": 100,
        "perishable": True,
        "censor": False,
        "finite_envs_num": 100,
        "seed": 123,
    },
    "16env_2d_sq": {
        "exp_name": "16env_2d_sq",
        "gen_method": "Perturb",
        "dim_num": 2,
        "action_ub": 100,
        "num_samp": 1000000,
        "horizon": 100,
        "perishable": True,
        "censor": False,
        "finite_envs_num": 16,
        "seed": 123,
        "need_square": True,
    },
    "inf_2d_sq": {
        "exp_name": "inf_2d_sq",
        "gen_method": "Perturb",
        "dim_num": 2,
        "action_ub": 100,
        "num_samp": 1000000,
        "horizon": 100,
        "perishable": True,
        "censor": False,
        "finite_envs_num": 0,
        "seed": 123,
        "need_square": True,
    },
}

MAB_dict = {
    "4env_20d": {
        "exp_name": "4env_20d",
        "gen_method": "Perturb",
        "dim_num": 20,
        "num_samp": 1000000,
        "horizon": 100,
        "finite_envs_num": 4,
        "err_std": 0.2,
        "finenvs_seed": 123,
    },
    "100env_20d": {
        "exp_name": "100env_20d",
        "gen_method": "Perturb",
        "dim_num": 20,
        "num_samp": 1000000,
        "horizon": 100,
        "finite_envs_num": 100,
        "err_std": 0.2,
        "finenvs_seed": 123,
    },
    "inf_20d": {
        "exp_name": "inf_20d",
        "gen_method": "Perturb",
        "dim_num": 20,
        "num_samp": 1000000,
        "horizon": 100,
        "finite_envs_num": 0,
        "err_std": 0.2,
        "finenvs_seed": 123,
    },
}

LinB_dict = {
    "4env_2d": {
        "exp_name": "4env_2d",
        "gen_method": "Perturb",
        "dim_num": 2,
        "num_samp": 1000000,
        "horizon": 100,
        "finite_envs_num": 4,
        "err_std": 0.2,
        "finenvs_seed": 123,
    },
    "100env_2d": {
        "exp_name": "100env_2d",
        "gen_method": "Perturb",
        "dim_num": 2,
        "num_samp": 1000000,
        "horizon": 100,
        "finite_envs_num": 100,
        "err_std": 0.2,
        "finenvs_seed": 123,
    },
    "inf_2d": {
        "exp_name": "inf_2d",
        "gen_method": "Perturb",
        "dim_num": 2,
        "num_samp": 1000000,
        "horizon": 100,
        "finite_envs_num": 0,
        "err_std": 0.2,
        "finenvs_seed": 123,
    },
}


samp_function_pool = {
    "DP_continuous": DP_cts_sample_multi,
    "NV_continuous": NV_ctx_sample_multi_cts,
    "MAB": MAB_sample_multi,
    "LinB": LinB_sample_multi,
}
# the data list to be generated

task_pool = {
    "DP_continuous": [
        DP_continuous_dict["16env_4d_sq123"],
        DP_continuous_dict["100env_4d_123"],
        DP_continuous_dict["inf_4d_123"],
    ],
}
task_pool = {
    "LinB": [LinB_dict["4env_2d"], LinB_dict["100env_2d"], LinB_dict["inf_2d"]]
}
task_pool = {
    "MAB": [MAB_dict["4env_20d"], MAB_dict["100env_20d"], MAB_dict["inf_20d"]]
}
task_pool = {
    "NV_continuous": [
        NV_continuous_dict["4env_4d_123"],
        NV_continuous_dict["inf_4d_123"],
        NV_continuous_dict["100env_4d_123"],
    ],
}
task_pool = {
    "DP_continuous": [DP_continuous_dict["inf_4d_sq123"]],
}
task_pool = {
    "NV_continuous": [
        NV_continuous_dict["16env_2d_sq"],
        NV_continuous_dict["inf_2d_sq"],
    ],
}
task_pool = {
    "DP_continuous": [DP_continuous_dict["inf_2d_sq123"]],
}
task_pool = {"NV_continuous": [NV_continuous_dict["inf_2d_123"]]}


def main():
    for task_type, task_args in task_pool.items():
        samp_function = samp_function_pool[task_type]

        for task in task_args:
            print("####################################")
            print("task_type:", task_type)
            print("task_info:", task)
            tik = time.time()
            regs, act_values, best_actions = samp_function(**task)
            print("##############finish#################")
            print("generation time:", (time.time() - tik) / 60)
            if task_type not in ["MAB", "LinB"]:
                action_error = np.abs(act_values - best_actions.squeeze())
            elif task_type == "MAB":
                # switch the one-hot encoding to the original action
                act_values = np.argmax(act_values, axis=2)
                action_error = np.abs(act_values - best_actions.squeeze())
            elif task_type == "LinB":
                # compute the L2 distance between the action and the optimal action
                action_error = np.linalg.norm(act_values - best_actions, axis=2)
            print("action errors:", action_error[:5, :])

            plt.plot(np.mean(action_error, axis=0), label="mean_act_error")
            plt.legend()
            plt.savefig(task_type + "_action_error.png")
            plt.show()
            plt.close()


if __name__ == "__main__":
    main()
