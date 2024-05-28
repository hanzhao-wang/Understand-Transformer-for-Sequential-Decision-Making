import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from quinine import Quinfig

from schema import schema
from testing.Bandits_test import LinB_test, MAB_test
from testing.DP_test import DP_test
from testing.NV_test import NV_test
from utils.multigpu_utils import init_distributed_mode


def Bayes_match_plot(Bayes_acts, GPT_acts, exp_name, trun_len=30):
    Bayes_acts = np.array(Bayes_acts)
    GPT_acts = np.array(GPT_acts)
    for i in range(5):
        plt.figure(figsize=(10, 10))
        plt.scatter(
            np.arange(trun_len),
            Bayes_acts[i, :trun_len],
            label=r"$f^*(h)$ (Oracle Posterior)",
            marker="o",
            color="blue",
            s=150,
        )
        plt.scatter(
            np.arange(trun_len),
            GPT_acts[i, :trun_len],
            label=r"$f_{\hat{\theta}}(h)$ (Transformer)",
            marker="x",
            color="red",
            s=150,
        )
        plt.xlabel("Time Step")
        plt.ylabel("Action Value")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("../figs/" + exp_name + "_Act_match_" + str(i) + ".png")
        plt.close()


def regret_plot(results, method_list, exp_name):
    plt.figure(figsize=(10, 10))
    colors_b = plt.cm.Blues(np.linspace(0.5, 0.9, len(method_list)))
    colors_g = plt.cm.Greens(np.linspace(0.5, 0.9, len(method_list)))
    for j, method in enumerate(method_list):
        reg_ = np.mean(np.array(results[method]["cum_regs"]).squeeze(), axis=0)
        reg_5per = np.percentile(
            np.array(results[method]["cum_regs"]).squeeze(), 5, axis=0
        )
        reg_95per = np.percentile(
            np.array(results[method]["cum_regs"]).squeeze(), 95, axis=0
        )
        if method == "GPT":
            color = "red"
            plt.plot(
                reg_,
                label="Transformer: " + str(np.round(reg_[-1], 2)),
                linestyle="-",
                color=color,
                linewidth=5,
            )
        elif method[:5] == "OrcTS":
            color = colors_b[j]
            if method == "OrcTS":
                label = "Oracle Posterior Sampling: " + str(
                    np.round(reg_[-1], 2)
                )
            else:
                label = r"Oracle Posterior $f^*(h)$: " + str(
                    np.round(reg_[-1], 2)
                )
            plt.plot(
                reg_,
                label=label,
                linestyle="--",
                color=colors_g[j],
                linewidth=5,
            )
        else:
            color = colors_b[j]
            plt.plot(
                reg_,
                label=method + ": " + str(np.round(reg_[-1], 2)),
                linestyle="-.",
                color=colors_b[j],
                linewidth=5,
            )
        plt.fill_between(
            np.arange(100), reg_5per, reg_95per, alpha=0.2, color=color
        )
    # log-scale x-axis
    plt.xscale("log")

    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("../figs/" + exp_name + "_Reg.png")
    plt.close()


def subact_plot(results, method_list, exp_name, norm="L1"):
    plt.figure(figsize=(10, 10))
    colors_b = plt.cm.Blues(np.linspace(0.5, 0.9, len(method_list)))
    colors_g = plt.cm.Greens(np.linspace(0.5, 0.9, len(method_list)))
    for j, method in enumerate(method_list):
        acts = np.array(results[method]["acts"])
        opt_acts = np.array(results[method]["opt_acts"]).squeeze()
        if norm == "L1":
            acts_opt = np.mean(
                np.abs(np.array(acts) - np.array(opt_acts)), axis=0
            )
            act_per5 = np.percentile(
                np.abs(np.array(acts) - np.array(opt_acts)), 5, axis=0
            )
            act_per95 = np.percentile(
                np.abs(np.array(acts) - np.array(opt_acts)), 95, axis=0
            )
        elif norm == "L2":
            acts_opt = np.mean(
                np.sqrt(
                    np.sum((np.array(acts) - np.array(opt_acts)) ** 2, axis=2)
                ),
                axis=0,
            )
            act_per5 = np.percentile(
                np.sqrt(
                    np.sum((np.array(acts) - np.array(opt_acts)) ** 2, axis=2)
                ),
                5,
                axis=0,
            )
            act_per95 = np.percentile(
                np.sqrt(
                    np.sum((np.array(acts) - np.array(opt_acts)) ** 2, axis=2)
                ),
                95,
                axis=0,
            )
        if method == "GPT":
            color = "red"
            plt.plot(
                acts_opt,
                label="Transformer",
                linestyle="-",
                color=color,
                linewidth=5,
            )
        elif method[:5] == "OrcTS":
            color = colors_b[j]
            if method == "OrcTS":
                label = "Oracle Posterior Sampling"
            else:
                label = r"Oracle Posterior $f^*(h)$"
            plt.plot(
                acts_opt,
                label=label,
                linestyle="--",
                color=colors_g[j],
                linewidth=5,
            )
        else:
            color = colors_b[j]
            plt.plot(
                acts_opt,
                label=method,
                linestyle="-.",
                color=colors_b[j],
                linewidth=5,
            )
        plt.fill_between(
            np.arange(100), act_per5, act_per95, alpha=0.2, color=color
        )
    plt.xlabel("Time Step")
    plt.ylabel("Action Suboptimality")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("../figs/" + exp_name + "_Act.png")
    plt.close()


def act_dif_plot(Bayes_acts, GPT_acts, exp_name, type="Hist", truncate_len=30):
    plt.figure(figsize=(10, 10))
    diff = (np.array(Bayes_acts) - np.array(GPT_acts)).squeeze()
    if type == "Hist":
        diff = diff[:, :truncate_len].flatten()
        plt.hist(diff, bins=20)
        # plot mean
        plt.axvline(
            np.mean(diff),
            color="k",
            linestyle="dashed",
            linewidth=1,
            label="Mean",
        )
        plt.xlabel(r"Diff. of $f_{\hat{\theta}}(h)$ and $f^*(h)$")
        plt.ylabel("Frequency")

    elif type == "Scatter":
        act_dim = diff.shape[-1]
        diff = diff[:, :truncate_len, :].reshape(-1, act_dim)
        diff_1 = diff[:, 0]
        diff_2 = diff[:, 1]
        plt.scatter(diff_1, diff_2, s=15)
        # plot mean
        plt.axhline(
            np.mean(diff_2),
            color="k",
            linestyle="dashed",
            linewidth=1,
            label="Mean",
        )
        plt.axvline(np.mean(diff_1), color="k", linestyle="dashed", linewidth=1)
        plt.xlabel(r"Diff. of $f_{\hat{\theta}}(h)$ and $f^*(h)$ (dim 1)")
        plt.ylabel(r"Diff. of $f_{\hat{\theta}}(h)$ and $f^*(h)$ (dim 2)")

    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("../figs/" + exp_name + "_Act_diff.png")
    plt.close()


def linear_bandit_plot(
    exp_names=["4env_2d", "100env_2d", "inf_2d"], tune=False
):

    ###############LinearBandits##########################

    for exp_name in exp_names:
        args_path = "../experiment/LinB/" + exp_name + ".yaml"
        if exp_name != "inf_2d":
            method_list = ["TS", "UCB", "OrcTS", "OrcTS_mean", "GPT"]
        else:
            method_list = ["TS", "UCB", "GPT"]
        args = Quinfig(config_path=args_path, schema=schema)
        args.eval.warm_up_steps = [0]
        args.eval.num_eval_episodes = 1
        init_distributed_mode(args.multigpu)

        test = LinB_test(args)
        if exp_name != "inf_2d":
            Bayes_acts, GPT_acts, Opt_acts = test.Bayes_check(100)
            act_dif_plot(
                Bayes_acts, GPT_acts, "LinB/_" + exp_name, type="Scatter"
            )
        results = test.compare(
            args, 100, paras_tuning=tune, method_names=method_list, seed=33
        )
        regret_plot(
            results, method_list, "LinB/_" + exp_name + "T_" + str(tune)
        )
        subact_plot(
            results,
            method_list,
            "LinB/_" + exp_name + "T_" + str(tune),
            norm="L2",
        )
        if exp_name in ["4env_2d", "100env_2d"]:
            # only check ['OrcTS','OrcTS_mean','GPT']
            regret_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "LinB/_" + exp_name + "_slim",
            )
            subact_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "LinB/_" + exp_name + "_slim",
                norm="L2",
            )


def MAB_plot(exp_names=["4env_20d", "100env_20d", "inf_20d"], tune=False):
    ###############MultiArmBandits##########################

    for exp_name in exp_names:
        args_path = "../experiment/MAB/" + exp_name + ".yaml"
        if exp_name != "inf_20d":
            method_list = ["TS", "UCB", "OrcTS", "OrcTS_argmax", "GPT"]
        else:
            method_list = ["TS", "UCB", "GPT"]
        args = Quinfig(config_path=args_path, schema=schema)
        args.eval.warm_up_steps = [0]
        args.eval.num_eval_episodes = 1
        init_distributed_mode(args.multigpu)

        test = MAB_test(args)
        if exp_name != "inf_20d":
            Bayes_acts, GPT_acts, Opt_acts = test.Bayes_check(100)
            act_dif_plot(Bayes_acts, GPT_acts, "MAB/_" + exp_name, type="Hist")
            Bayes_match_plot(Bayes_acts, GPT_acts, "MAB/_" + exp_name)
        results = test.compare(
            args, 100, paras_tuning=tune, method_names=method_list, seed=33
        )
        regret_plot(results, method_list, "MAB/_" + exp_name + "T_" + str(tune))
        subact_plot(
            results,
            method_list,
            "MAB/_" + exp_name + "T_" + str(tune),
            norm="L1",
        )
        if exp_name in ["4env_20d", "100env_20d"]:
            regret_plot(
                results,
                ["OrcTS", "OrcTS_argmax", "GPT"],
                "MAB/_" + exp_name + "_slim",
            )
            subact_plot(
                results,
                ["OrcTS", "OrcTS_argmax", "GPT"],
                "MAB/_" + exp_name + "_slim",
                norm="L1",
            )


def DP_plot(
    exp_names=["4env_4d_132", "100env_4d_123", "inf_4d_123"], tune=False
):
    #############DynamicPricing_PureLinearDemand##########################

    for exp_name in exp_names:
        args_path = "../experiment/DP_cts/" + exp_name + ".yaml"
        if exp_name == "inf_4d_123":
            method_list = ["ILSE", "CILS", "TS", "GPT"]
        else:
            method_list = ["ILSE", "CILS", "TS", "OrcTS", "OrcTS_mean", "GPT"]
        gen_method = None

        args = Quinfig(config_path=args_path, schema=schema)

        args.eval.warm_up_steps = [0]
        args.eval.num_eval_episodes = 1
        init_distributed_mode(args.multigpu)

        test = DP_test(args)
        if exp_name != "inf_4d_123":
            Bayes_acts, GPT_acts, Opt_acts = test.Bayes_check(100)
            act_dif_plot(Bayes_acts, GPT_acts, "DP/_" + exp_name, type="Hist")
            Bayes_match_plot(Bayes_acts, GPT_acts, "DP/_" + exp_name)

        results = test.compare(
            args,
            100,
            paras_tuning=tune,
            method_names=method_list,
            gen_method=gen_method,
        )
        regret_plot(results, method_list, "DP/_" + exp_name + "T_" + str(tune))
        subact_plot(
            results,
            method_list,
            "DP/_" + exp_name + "T_" + str(tune),
            norm="L1",
        )
        if exp_name in ["4env_4d_132", "100env_4d_123"]:
            # only check ['OrcTS','OrcTS_mean','GPT']
            regret_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "DP/_" + exp_name + "_slim",
            )
            subact_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "DP/_" + exp_name + "_slim",
                norm="L1",
            )


def DP_2demands_plot(exp_names=["16env_4d_sq123", "inf_2d_sq"], tune=False):
    #############DynamicPricing_TwoDemand##########################

    for exp_name in exp_names:
        args_path = "../experiment/DP_cts/" + exp_name + ".yaml"
        if exp_name == "inf_2d_sq":
            method_list = ["ILSE", "CILS", "TS", "GPT"]
            gen_list = [None, "Square"]
        else:
            method_list = ["ILSE", "CILS", "TS", "OrcTS", "OrcTS_mean", "GPT"]
            gen_list = [None]

        args = Quinfig(config_path=args_path, schema=schema)

        args.eval.warm_up_steps = [0]
        args.eval.num_eval_episodes = 1
        init_distributed_mode(args.multigpu)

        test = DP_test(args)
        if exp_name != "inf_2d_sq":
            Bayes_acts, GPT_acts, Opt_acts = test.Bayes_check(100)
            act_dif_plot(
                Bayes_acts, GPT_acts, "DP_2demands/_" + exp_name, type="Hist"
            )
            Bayes_match_plot(Bayes_acts, GPT_acts, "DP_2demands/_" + exp_name)
            test.traj_compare(
                args,
                val_num=30,
                truncate_len=20,
                train_steps=[40, 50, 70, 80, 110, 120],
                seed=1,
            )
        for gen_method in gen_list:
            results = test.compare(
                args,
                100,
                paras_tuning=tune,
                method_names=method_list,
                gen_method=gen_method,
                seed=1,
            )
            regret_plot(
                results,
                method_list,
                "DP_2demands/_"
                + exp_name
                + "_gen_"
                + str(gen_method)
                + "T_"
                + str(tune),
            )
            subact_plot(
                results,
                method_list,
                "DP_2demands/_"
                + exp_name
                + "_gen_"
                + str(gen_method)
                + "T_"
                + str(tune),
                norm="L1",
            )
        if exp_name in ["16env_4d_sq123"]:
            # only check ['OrcTS','OrcTS_mean','GPT']
            regret_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "DP_2demands/_" + exp_name + "_slim",
            )
            subact_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "DP_2demands/_" + exp_name + "_slim",
                norm="L1",
            )


def NV_plot(exp_names=["4env_4d", "100env_4d", "inf_2d"], tune=False):
    #############Newsvendor##########################

    for exp_name in exp_names:
        args_path = "../experiment/NV_cts/" + exp_name + ".yaml"
        if exp_name != "inf_2d":
            method_list = ["ERM", "OGD", "OrcTS", "OrcTS_mean", "GPT"]
        else:
            method_list = ["ERM", "OGD", "GPT"]
        args = Quinfig(config_path=args_path, schema=schema)
        args.eval.warm_up_steps = [0]
        args.eval.num_eval_episodes = 1
        init_distributed_mode(args.multigpu)

        test = NV_test(args)
        if exp_name != "inf_2d":
            for median in [True, False]:
                Bayes_acts, GPT_acts, Opt_acts = test.Bayes_check(100, median)
                name = (
                    "NV/_" + exp_name + "Median"
                    if median
                    else "NV/_" + exp_name + "Mean"
                )
                act_dif_plot(Bayes_acts, GPT_acts, name, type="Hist")
                Bayes_match_plot(Bayes_acts, GPT_acts, name)
        results = test.compare(
            args, 100, paras_tuning=tune, method_names=method_list, seed=33
        )
        regret_plot(results, method_list, "NV/_" + exp_name + "T_" + str(tune))
        subact_plot(
            results,
            method_list,
            "NV/_" + exp_name + "T_" + str(tune),
            norm="L1",
        )
        if exp_name in ["4env_4d", "100env_4d"]:
            # only check ['OrcTS','OrcTS_mean','GPT']
            regret_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "NV/_" + exp_name + "_slim",
            )
            subact_plot(
                results,
                ["OrcTS", "OrcTS_mean", "GPT"],
                "NV/_" + exp_name + "_slim",
                norm="L1",
            )


def NV_2demands_plot(exp_names=["16env_2d_sq", "inf_2d_sq"], tune=False):

    #############DynamicPricing_TwoDemand##########################

    for exp_name in exp_names:
        args_path = "../experiment/NV_cts/" + exp_name + ".yaml"
        if exp_name == "inf_2d_sq":
            method_list = ["ERM", "OGD", "GPT"]
            gen_list = [None, "Square"]
        else:
            method_list = ["ERM", "OGD", "OrcTS", "OrcTS_mean", "GPT"]
            gen_list = [None]

        args = Quinfig(config_path=args_path, schema=schema)

        args.eval.warm_up_steps = [0]
        args.eval.num_eval_episodes = 1
        init_distributed_mode(args.multigpu)

        test = NV_test(args)
        if exp_name != "inf_2d_sq":
            Bayes_acts, GPT_acts, Opt_acts = test.Bayes_check(100)
            act_dif_plot(
                Bayes_acts, GPT_acts, "NV_2demands/_" + exp_name, type="Hist"
            )
            Bayes_match_plot(Bayes_acts, GPT_acts, "NV_2demands/_" + exp_name)
        for gen_method in gen_list:
            results = test.compare(
                args,
                100,
                paras_tuning=tune,
                method_names=method_list,
                gen_method=gen_method,
                seed=1,
            )
            regret_plot(
                results,
                method_list,
                "NV_2demands/_"
                + exp_name
                + "_gen_"
                + str(gen_method)
                + "T_"
                + str(tune),
            )
            subact_plot(
                results,
                method_list,
                "NV_2demands/_"
                + exp_name
                + "_gen_"
                + str(gen_method)
                + "T_"
                + str(tune),
                norm="L1",
            )


if __name__ == "__main__":

    # set the font size
    plt.rcParams.update({"font.size": 20})
    # check the path
    tasks = ["LinB", "MAB", "DP", "DP_2demands", "NV", "NV_2demands"]
    for task in tasks:
        if not os.path.exists("../figs/" + task):
            os.makedirs("../figs/" + task)

    linear_bandit_plot()
    print("LinearBandit Done")
    DP_plot()
    print("DP Done")
    MAB_plot()
    print("MAB Done")
    print("DP_2demands Done")
    DP_2demands_plot()
    NV_plot()
    print("NV Done")
    NV_2demands_plot()
