import matplotlib.pyplot as plt
import numpy as np


def plot_bar(method_list, x_axis_list_len, data, x_label, y_label):
    num_methods = len(method_list)
    num_categories = len(x_axis_list_len)

    # Creating a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    # font size
    plt.rcParams.update({"font.size": 18})
    # The x position of the groups
    group_pos = np.arange(1, num_categories + 1)

    # Boxplot width
    box_width = 0.15
    bsr_list = []
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, num_methods))
    # Plotting each method's data as a separate boxplot
    for i, method in enumerate(method_list):
        pos = group_pos - (num_methods / 2.0 - i) * box_width
        for j, x_axis in enumerate(x_axis_list_len):
            color = colors[i]
            label = method
            if i == 0:
                color = "r"
                label = "OMGPT"
            bp = ax.bar(
                pos[j],
                data[method][str(x_axis)],
                width=box_width,
                color=color,
                align="center",
                label=label,
            )
        bsr_list.append(bp)

        # Set properties for each boxplot

    # Setting the x-ticks to be in the middle of the groups
    ax.set_xticks(group_pos)
    ax.set_xticklabels(x_axis_list_len)

    # Adding title and labels

    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Adding a legend with the corresponding group colors

    ax.legend(bsr_list, method_list, fontsize=18)
    return ax


def plot_boxplot(method_list, x_axis_list_len, data, y_label=None, labels=None):

    num_methods = len(method_list)
    num_categories = len(x_axis_list_len)

    # Creating a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    # font size
    plt.rcParams.update({"font.size": 14})
    # The x position of the groups
    group_pos = np.arange(1, num_categories + 1)

    # Boxplot width
    box_width = 0.2
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, num_methods))
    bp_list = []
    # Plotting each method's data as a separate boxplot
    for i, method in enumerate(method_list):
        pos = group_pos - (num_methods / 2.0 - i) * box_width + box_width / 2
        data_plot = []

        for j, config in enumerate(x_axis_list_len):
            if len(method_list) == 1:
                data_plot.append(data[str(config)])
            else:
                data_plot.append(data[str(method)][str(config)])
        bp = ax.boxplot(
            np.array(data_plot).T,
            positions=pos,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            labels=labels,
        )
        bp_list.append(bp)

        # Set properties for each boxplot

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i])
            # set the transparency of the box
            patch.set_alpha(1)
        for element in ["medians"]:
            plt.setp(bp[element], color="black")
        # set the line width of the median
        plt.setp(bp["medians"], linewidth=1)
        # set the box's outline to black
        plt.setp(bp["boxes"], linewidth=1, edgecolor="black")
        # add legend

    # Setting the x-ticks to be in the middle of the groups
    ax.set_xticks(group_pos)
    ax.set_xticklabels(x_axis_list_len)
    ax.set_xlim(0.5, num_categories + 0.8)
    # add y label
    if y_label != None:
        ax.set_ylabel(y_label)
    # Adding title and labels

    ax.grid(True)
    # add the legend
    method_name_list = method_list
    if labels != None:

        ax.legend(
            [bp_list[i]["boxes"][0] for i in range(len(bp_list))],
            method_name_list,
            loc="upper left",
            fontsize=18,
        )

    # Show the plot
    plt.show()
    return fig, ax
