import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import RNA
import pandas as pd
import re
import copy

pgf_with_custom_preamble = {
    "font.family": "serif",  # use serif/main font for text elements
    "font.size": 8,
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "pgf.preamble": [
        "\\usepackage{units}",  # load additional packages
        "\\usepackage{metalogo}",
        "\\usepackage{unicode-math}",  # unicode math setup
        r"\setmathfont{xits-math.otf}",
        r"\setmainfont{DejaVu Serif}",  # serif font via preamble
    ],
}
cm = 1 / 2.54

plt.rcParams.update(pgf_with_custom_preamble)

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

markers = ["o", "^", "X", "D", "P", "s"]


def plot_unpaired(
    dataset=None,
    figure_path=None,
    plot_interval_start=None,
    plot_interval_end=None,
    grey=False,
):

    rnafold = dataset.global_folding.unpaired_P
    stochastic = dataset.local_sampling.unpaired_P
    mea_struct_stochastic = dataset.local_sampling.mea_DB
    rnafold_mfe = dataset.local_sampling.mfe_DB
    seq = dataset.sequence

    plot_start = 0
    plot_end = len(seq)

    if plot_interval_start != None:
        plot_start = plot_interval_start
    if plot_interval_end != None:
        plot_end = plot_interval_end

    plot_len = plot_end - plot_start
    plot_seq = seq[plot_start:plot_end]

    plt.rcParams.update(pgf_with_custom_preamble)

    fig, axs = plt.subplots(figsize=(20 * cm, 4 * cm), nrows=1, ncols=1, layout="tight")

    plt.rcParams.update(pgf_with_custom_preamble)

    if grey:
        axs.plot(rnafold[1:], c="#000000", alpha=1, label="$p_k^{\circ}$", lw=0.7)
        axs.plot(stochastic[1:], c="#999999", alpha=1, label="$q_k$", lw=0.7)
    else:
        axs.plot(rnafold[1:], c="tab:blue", alpha=1, label="$p_k^{\circ}$", lw=0.7)
        axs.plot(stochastic[1:], c="tab:orange", alpha=1, label="$q_k$", lw=0.7)

    axs.legend(loc="right")
    axs.set_xlabel("sequence position $k$")
    axs.set_xlim(plot_start, plot_end)

    if plot_len <= 200:
        axs.set_xticks(list(range(plot_start, plot_end)))
        plot_mea = mea_struct_stochastic[plot_start:plot_end]
        ss = rnafold_mfe[plot_start:plot_end]
        labels = []
        for i, l in enumerate(ss):
            labels.append("\n".join([str(i + plot_start), plot_seq[i], l, plot_mea[i]]))
        axs.set_xticklabels(labels)

    if figure_path is not None:
        fig.savefig(figure_path)


def heatmap(
    dataset,
    start=1,
    end=None,
    figure_path=None,
    threshold=None,  # 0.10,
    grey=False,
    what="joint_paired",
):
    """
    Plot the conditional & unpaired probability in a heatmap

    param dataset: extract conditional probability, unpaired_P, sequence
    param figure_path: output path and name
    param threshold: any probability below this threshold is set to -1
    param annot: draw a plfold heatmap with the numeric values in each cell
    param grey: plot black-white
    param what: joint_paired, joint_unpaired, cond_paired, cond_unpaired, rnafold_bpp
    """
    # i = start
    if end == None:
        end = len(dataset.sequence)
    # j = end

    if what == "joint_paired":
        cond_unpaired_P = copy.deepcopy(
            dataset.local_sampling.joint_paired_P[start:end, start:end]
        )
        unpaired_P = np.array(copy.deepcopy(dataset.local_sampling.paired_P[start:end]))
    elif what == "joint_unpaired":
        cond_unpaired_P = copy.deepcopy(
            dataset.local_sampling.joint_unpaired_P[start:end, start:end]
        )
        unpaired_P = np.array(
            copy.deepcopy(dataset.local_sampling.unpaired_P[start:end])
        )
    elif what == "cond_paired":
        cond_unpaired_P = copy.deepcopy(
            dataset.local_sampling.cond_paired_P[start:end, start:end]
        )
        unpaired_P = np.array(copy.deepcopy(dataset.local_sampling.paired_P[start:end]))
    elif what == "cond_unpaired":
        cond_unpaired_P = copy.deepcopy(
            dataset.local_sampling.cond_unpaired_P[start:end, start:end]
        )
        unpaired_P = np.array(
            copy.deepcopy(dataset.local_sampling.unpaired_P[start:end])
        )
    elif what == "rnafold_bpp":
        cond_unpaired_P = copy.deepcopy(
            dataset.global_folding.bp_P[start:end, start:end]
        )
        unpaired_P = np.array(copy.deepcopy(dataset.local_sampling.paired_P[start:end]))

    sequence = dataset.sequence[start:end]

    unpaired_P2 = np.array(unpaired_P)
    unpaired_P2 = unpaired_P2.reshape(unpaired_P2.shape[0], 1)

    # threshold preperation
    if threshold is not None:
        pos = np.argwhere(unpaired_P2 < threshold)[:, 0]

        unpaired_P[unpaired_P < threshold] = -1
        unpaired_P2[unpaired_P2 < threshold] = -1
        if pos.size:
            for n in pos:
                cond_unpaired_P[:, n] = -1
                cond_unpaired_P[n] = -1
                vmin = -0.5
                center = 0
        else:
            vmin = 0
            center = 0

    else:
        vmin = 0
        center = 0

    # get labels
    if sequence is not None:
        labels = [str(e + 1) + "-" + s for e, s in enumerate(sequence)]
    else:
        labels = list(range(1, mx.shape[0]))

    # generate plotting specifications
    f = plt.figure(figsize=(30, 30))

    gs = GridSpec(
        2,
        2,
        left=0,
        right=1,
        wspace=-0.3,
        hspace=0.08,
        width_ratios=[1, len(unpaired_P)],
        height_ratios=[len(unpaired_P), 1],
    )
    ax1 = f.add_subplot(gs[0, 1])
    ax2 = f.add_subplot(gs[0, 0])
    ax3 = f.add_subplot(gs[1, 1])

    if grey == False:
        cmap = "RdGy_r"
    else:
        cmap = "Greys"
        ax1.patch.set(hatch="xx", edgecolor="black")
        ax2.patch.set(hatch="xx", edgecolor="black")
        ax3.patch.set(hatch="xx", edgecolor="black")

    # conditional probability
    sns.heatmap(
        cond_unpaired_P,
        ax=ax1,
        # linewidths=0,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
        cbar_kws={"label": "unpaired probability", "shrink": 0.1},
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        center=center,
        cbar=False,
        linewidths=0.005,
    )

    ax1.set_xticklabels(labels, rotation=90, fontsize=16)
    ax1.set_yticklabels(labels, rotation=1, fontsize=16)

    sns.heatmap(
        unpaired_P2,
        ax=ax2,
        # linewidths=0,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        center=center,
        cbar=False,
        linewidths=0.005,
    )

    sns.heatmap(
        [unpaired_P],
        ax=ax3,
        # linewidths=0,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
        annot=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        center=center,
        cbar=False,
        linewidth=0.005,
    )

    if figure_path is not None:
        f.savefig(figure_path, bbox_inches="tight")


def bp_list_from_db(db):
    """finde pairing positions in stockholm bracket notation string"""
    bra = []  # opening positions
    pairs = []  # positions that form a basepair
    uppercase_regex = re.compile("[A-Z]")  # letters for pseudo knots
    lowercase_regex = re.compile("[a-z]")
    knote_bp = {}
    for i in range(0, len(db)):
        if db[i] in ["<", "(", "[", "{"]:  # unknoted base apirs
            bra.append(i)
        elif db[i] in [">", ")", "]", "}"]:
            pairs.append([bra.pop(), i])
        elif uppercase_regex.match(db[i]):
            if db[i] in knote_bp.keys():
                test = knote_bp[db[i]]
                test.append(i)
                knote_bp[db[i]] = test
            else:
                knote_bp[db[i]] = [i]
        elif lowercase_regex.match(db[i]):
            pairs.append([knote_bp[db[i].upper()].pop(), i])
    return pairs


def db_from_bp_list(bp_list, seq_len):
    """Get dotbraket string from base pair list."""
    db = ["."] * seq_len
    for pair in bp_list:
        db[pair[0]] = "("
        db[pair[1]] = ")"
    return "".join(db)


def struct_ps(dataset, start, end, annotations=None, figure_path=None, context=False):
    """
    Plot in PS format for a substructure.

    param dataset: extract sequence, mea_DB
    param start, end: index from 1 to len(sequence)
    """
    seq = dataset.sequence
    struct = dataset.local_sampling.mea_DB

    if start == 0:
        raise Exception("Sorry, indexing start at 1")
    if start > end:
        raise Exception("start should be smaller than end")

    if context:
        ind_struct = RNA.ptable(struct)
        new_end = end
        new_start = start
        changed = False
        first = True
        smaller_start = []
        bigger_end = []

        for s in range(start, end):
            if ind_struct[s] > 0 and ind_struct[s] < new_start:
                smaller_start.append((ind_struct[s], s))
                new_start = ind_struct[s]
                changed = True
            elif ind_struct[s] > new_end:
                bigger_end.append((s, ind_struct[s]))
                new_end = ind_struct[s]
                changed = True
        if changed == True:
            print(f"end={end} has been changed to {new_end}")
            print(f"i={start} has been changed to {new_start}")

        filename = f"struct_{dataset.seq_id}_{start}_{end}_context.eps"

        new_seq = seq[new_start - 1 : new_end]
        new_struct = struct[new_start - 1 : new_end]

    else:
        new_start = start
        new_seq = seq[start - 1 : end]
        bp_list = bp_list_from_db(struct)
        # remove any absepairs that are
        bp_list = [
            pair for pair in bp_list if pair[0] >= start - 1 and pair[1] <= end - 1
        ]
        new_struct = db_from_bp_list(bp_list, len(seq))[start - 1 : end]
        filename = f"struct_{dataset.seq_id}_{start}_{end}.eps"

    if figure_path is not None:
        filename = figure_path

    if annotations:
        annotation_string = ""
        for sub_s in annotations:
            annotation_string += (
                f"{sub_s[0]+1-new_start} {sub_s[1]+1-new_start} 12 0.8 0.8 0.8 omark "
            )
        RNA.cvar.rna_plot_type = 4
        RNA.file_PS_rnaplot_a(new_seq, new_struct, filename, annotation_string, "")
    else:
        RNA.cvar.rna_plot_type = 4
        RNA.file_PS_rnaplot(new_seq, new_struct, filename)


def diff_sum(a, b):
    diff = np.abs(a - b).sum()
    return diff


def diff_per_nt(a, b):
    diff = np.abs(a - b).sum() / (len(a) - 1)
    return diff


def diff_norm(a, b):
    diff = np.abs(a - b).sum() / b.sum()
    return diff


def diff_rel(a, b):
    diff = np.nan_to_num(2 * np.abs(a - b) / (a + b), nan=0).sum() / len(a)
    return diff


def difference(dataset, grey=False):
    """
    Calculate the differences between different sample and windowsizes
    param id: sequence id
    param data: list to data collection
    param difftrue: plot data true/false
    param grey: plot in black/white
    """
    data = []
    for d in dataset:
        data.append(
            {
                "win_len": d.window_size,
                "samplesize": d.num_samples,
                "seq": d.sequence,
                "up_sample": np.asarray(d.local_sampling.unpaired_P),
                "up_plfold": np.asarray(d.local_exact.unpaired_P),
                "id": d.seq_id,
                "bpspan": d.max_bp_span,
            }
        )

    df = pd.DataFrame.from_records(data)

    df["diff_sum"] = df.apply(
        lambda row: diff_sum(row["up_sample"], row["up_plfold"]), axis=1
    )
    df["diff_norm"] = df.apply(
        lambda row: diff_norm(row["up_sample"], row["up_plfold"]), axis=1
    )
    df["diff_rel"] = df.apply(
        lambda row: diff_rel(row["up_sample"], row["up_plfold"]), axis=1
    )
    df["diff_per_nt"] = df.apply(
        lambda row: diff_per_nt(row["up_sample"], row["up_plfold"]), axis=1
    )

    return df


def error_vs_sample_coverage(data, figure_path=None):

    plot_df = pd.DataFrame(
        data, columns=["sample size", "window size", "diff", "coverage"]
    )

    fig, axs = plt.subplots(figsize=(8 * cm, 5 * cm), nrows=1, ncols=1)

    i = 0
    for length, gdf in plot_df.groupby("window size"):
        gdf.plot.scatter(
            x="coverage",
            y="diff",
            ax=axs,
            label=str(length),
            s=8,
            linewidths=0.5,
            c=colors[i],
            marker=markers[i],
        )
        i += 1
    axs.set_xlabel("$W*s$")
    axs.set_ylabel("mean $\\mid q_k^{exact}-q_k^{sample}\\mid$")
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)

    plt.legend(
        title="$W$",
        ncol=3,
        loc="upper right",
        labelspacing=0.3,
        handletextpad=0.5,
        columnspacing=1,
        fontsize="small",
    )

    # axs.set_xlim(0,2800)
    # axs.set_ylim(0,0.03)
    # axs.set_xscale('log')

    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches="tight")


def plot_diff_per_samplesize(id, df, diff, figure_path=None, grey=False):
    """
    Plot the difference between pflfold and sampling - PER SAMPLESIZE

    param df: dataframe with the win_len, samplessize, seq, ...
    param diff: respective difference to be plotted
    param figure_path=None: figure path & name
    param grey: plot black-white
    """

    def forward(x):
        return np.log2(x)

    f, ax = plt.subplots(figsize=(7, 4))
    groups = df.groupby(by="win_len")

    if grey == False:
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purpel",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        for i, g in enumerate(groups):
            g[1].plot.scatter(
                x="samplesize",
                y=diff,
                ax=ax,
                legend=True,
                label=g[0],
                color=[colors[i]] * len(g[1]),
                # colormap = 'Blues',
                alpha=0.4,
            )
    if grey == True:
        markers = ["o", "^", "s", "D", "p", "P", "X", "*", "p", "d"]
        for i, g in enumerate(groups):
            g[1].plot.scatter(
                x="samplesize",
                y=diff,
                ax=ax,
                legend=True,
                label=g[0],
                marker=markers[i],
                color="black",
                # markersize=12,
                alpha=0.4,
            )

    ax.set_xlim(df["samplesize"].min() / 2, df["samplesize"].max() * 2)
    ax.set_xscale("function", functions=(forward, forward))
    ax.set_xticks(list(set(df["samplesize"].to_list())))
    ax.legend(title="win_len")
    ax.set_ylim(bottom=0)
    if figure_path is not None:
        f.savefig(figure_path)


def plot_diff_per_samplesize_window(id, df, diff, figure_path=None, grey=False):
    """
    Plot the difference between pflfold and sampling - PER SAMPLESIZE & WINDOW

    param df: dataframe with the win_len, samplessize, seq, ...
    param diff: respective difference to be plotted
    param figure_path=None: figure path & name
    """
    f, ax = plt.subplots(figsize=(13, 6))
    if grey == False:
        c = "#1f77b4"
    if grey == True:
        c = "black"

    df.boxplot(column=[diff], by=["samplesize", "win_len"], ax=ax, color=c)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if figure_path is not None:
        f.savefig(figure_path)
    else:
        f.savefig(diff + "id_" + id + "_per_sample_window_size_box.pdf")


def plot_unpaired_hist(datasets, figure_path=None, tsv_file=None):

    windowsizes = list()
    data = dict()

    for i, dataset in enumerate(datasets):
        windowsize = dataset.window_size
        windowsizes.append(windowsize)

        data[str(windowsize).zfill(4) + " RNAfold"] = dataset.global_folding.unpaired_P[
            1:
        ]  # [0]
        data[
            str(windowsize).zfill(4) + " RNAPLfold"
        ] = dataset.local_sampling.unpaired_P

    df = pd.DataFrame.from_dict(data)
    df = df.reindex(sorted(df.columns), axis=1)
    fig, axs = plt.subplots(
        figsize=(6.4 * cm, 8 * cm),
        nrows=len(windowsizes) + 1,
        ncols=1,
        gridspec_kw={"hspace": 0.1},
    )
    plt.rcParams.update(pgf_with_custom_preamble)

    plot_columns = [str(c) for c in df.columns if str("RNAPLfold") in str(c)]
    plot_columns = plot_columns + [
        [str(c) for c in df.columns if str("RNAfold") in str(c)][0]
    ]
    for i, c in enumerate(plot_columns):
        bins = [b / 20 for b in list(range(0, 21))]
        df.hist(
            c, ax=axs[i], bins=bins, grid=False, color="lightgrey"
        )  # legend = str(c))
        axs[i].set_xlim(-0.1, 1.1)
        axs[i].set_ylim(0, axs[i].get_ylim()[1] * 1.4)
        axs[i].set_yticks([])
        axs[i].set_xlabel(None)
        axs[i].set_ylabel("$q_k$; $W=" + c.split(" ")[0].lstrip("0") + "$", rotation=0)
        axs[i].set_title(None)
        axs[i].spines.right.set_visible(False)
        axs[i].spines.left.set_visible(False)
        axs[i].spines.top.set_visible(False)
        if "RNAfold" in c:
            axs[i].set_ylabel("$p_k^{\circ}$")
        axs[i].yaxis.set_label_coords(0.5, 0.2)

    if figure_path != None:
        fig.savefig(figure_path)


def plot_error_vs_windowsize_boxplot(datasets, figure_path=None, tsv_file=None):

    windowsizes = list()
    data = {}

    for dataset in datasets:
        windowsize = dataset.window_size
        rnafold_unpaired = dataset.global_folding.unpaired_P
        unpaired_prob = dataset.local_sampling.unpaired_P

        windowsizes.append(windowsize)

        data[str(windowsize).zfill(4) + " RNAfold"] = rnafold_unpaired[0]
        data[str(windowsize).zfill(4) + " RNAPLfold"] = unpaired_prob
        data[str(windowsize).zfill(4) + " diff"] = unpaired_prob - rnafold_unpaired[0]
        data[str(windowsize).zfill(4) + " absdiff"] = np.absolute(
            data[str(windowsize).zfill(4) + " diff"]
        )

    df = pd.DataFrame.from_dict(data)
    df = df.reindex(sorted(df.columns), axis=1)

    fig, axs = plt.subplots(figsize=(6 * cm, 8 * cm), nrows=1, ncols=1, layout="tight")

    plt.rcParams.update(pgf_with_custom_preamble)
    plot_columns = [str(c) for c in df.columns if str("absdiff") in str(c)]
    whiskerprops = dict(color="black")
    boxprops = dict(color="black")
    medianprops = dict(color="darkgrey")
    df.boxplot(
        plot_columns,
        ax=axs,
        whis=(5, 95),
        showfliers=False,
        grid=False,
        whiskerprops=whiskerprops,
        boxprops=boxprops,
        medianprops=medianprops,
    )
    axs.set_ylim(-0.1, 1.1)
    axs.set_xticklabels(
        [l.get_text().split(" ")[0].lstrip("0") for l in axs.get_xticklabels()],
        rotation=90,
    )
    axs.set_ylabel("$\mid q_k - p_k^{\circ} \mid$")
    axs.set_xlabel("$W$")

    if figure_path != None:
        fig.savefig(
            figure_path,
        )


def unpairedP_correlation_local_sampling_vs_RNAfold(
    dataset, figure_path=None, tsv_file=None
):

    data = {}
    bins = 20
    data[
        str(dataset.window_size).zfill(4) + " RNAfold"
    ] = dataset.global_folding.unpaired_P[1:]
    data[
        str(dataset.window_size).zfill(4) + " local sampling"
    ] = dataset.local_sampling.unpaired_P

    windowsize = dataset.window_size

    df = pd.DataFrame.from_dict(data)
    df = df.reindex(sorted(df.columns), axis=1)
    fig = plt.figure(figsize=(9.6 * cm, 9.6 * cm))
    plt.rcParams.update(pgf_with_custom_preamble)

    widths = [4, 1]
    heights = [1, 4]

    ### gridspec preparation
    spec = fig.add_gridspec(
        ncols=2,
        nrows=2,
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.0001,
        hspace=0.0001,
    )  # setting spaces

    ### setting axes
    axs = {}
    for i in range(len(heights) * len(widths)):
        axs[i] = fig.add_subplot(spec[i // len(widths), i % len(widths)])

    ###  q_k (local unpaiired probability) vs p_k (RNA fold with max bp span)

    df.plot.scatter(
        x=str(windowsize).zfill(4) + " local sampling",
        y=str(windowsize).zfill(4) + " RNAfold",
        alpha=0.4,
        ax=axs[2],
        s=0.5,
        c="#333333",
    )
    axs[2].set_ylim(0, 1)
    axs[2].set_xlim(0, 1)

    # histogram q_k
    df.hist(
        str(windowsize).zfill(4) + " local sampling",
        ax=axs[0],
        bins=bins,
        grid=False,
        color="grey",
    )

    axs[0].set_xlim(0, 1)
    axs[0].set_xlabel("")
    axs[0].set_xticklabels([])
    axs[0].spines["left"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    # histogram p_k
    df.hist(
        str(windowsize).zfill(4) + " RNAfold",
        ax=axs[3],
        bins=bins,
        grid=False,
        color="grey",
        orientation="horizontal",
    )
    axs[3].set_ylim(0, 1)
    axs[3].set_ylabel("")
    axs[3].set_yticklabels([])
    axs[3].spines["bottom"].set_visible(False)
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)

    # not using axis element 1
    axs[1].axis("off")

    # remove redundent labels and titles
    axs[0].set_title("")
    axs[3].set_title("")

    axs[2].set_yticks([0, 0.5, 1])
    axs[2].set_xticks([0, 0.5])

    axs[3].set_yticks([0, 0.5, 1])
    axs[3].set_xticks([])
    axs[0].set_yticks([])

    axs[0].set_ylabel("")
    axs[3].set_xlabel("")
    axs[0].set_yticklabels([])
    axs[3].set_xticklabels([])

    axs[2].set_ylabel("$p_k^{\circ}$")
    axs[2].set_xlabel("$q_k$; $W=" + str(windowsize) + "$")

    fig.suptitle(
        f"{dataset.seq_id}\n max. base pair span: {dataset.max_bp_span}, sample size: {dataset.num_samples}"
    )

    if figure_path != None:
        fig.savefig(figure_path, bbox_inches="tight")


def get_dist_df(mx):
    dists = []
    for i in range(0, len(mx)):
        for j in range(i + 1, len(mx)):
            dists.append([j - i, mx[i, j]])

    df = pd.DataFrame(dists, columns=["dist", "value"])
    return df.groupby("dist").sum()


def bp_span_hist(dataset, figure_path=None):

    fig, axs = plt.subplots(figsize=(20 * cm, 15 * cm), nrows=2, ncols=1)

    df = pd.DataFrame()

    for d in dataset:
        df1 = get_dist_df(d.global_folding.bp_P)
        df1.rename(
            columns={"value": f"RNAfold with max bp span {d.max_bp_span}"},
            inplace=True,
        )
        df2 = get_dist_df(d.local_sampling.bp_P)
        df2.rename(
            columns={
                "value": f"local sampling: $W$ = {d.window_size}, max_bp_span = {d.max_bp_span}"
            },
            inplace=True,
        )

        df = df.merge(df1, how="outer", left_index=True, right_index=True)
        df = df.merge(df2, how="outer", left_index=True, right_index=True)

    max_count = df.max().max() * 1.1
    rnafold_cols = [c for c in df.columns if "RNAfold" in c]
    stochastic_cols = [c for c in df.columns if "sampling" in c]
    df.plot(y=rnafold_cols, ax=axs[0])
    df.plot(y=stochastic_cols, ax=axs[1])

    axs[0].set_xlim(0, max([dataset[i].max_bp_span for i in range(0, len(dataset))]))
    axs[1].set_xlim(0, max([dataset[i].max_bp_span for i in range(0, len(dataset))]))
    axs[0].set_ylim(0, max_count)
    axs[1].set_ylim(0, max_count)
    axs[0].set_xlabel("base pair span")
    axs[1].set_xlabel("base pair span")
    axs[0].set_ylabel("probability weighted base pair count")
    axs[1].set_ylabel("probability weighted base pair count")
    fig.suptitle(
        "frequency of base pair spans\nsequence length: {seq_len}; sample size: {sample_size}".format(
            seq_len=len(dataset[0].sequence), sample_size=dataset[0].num_samples
        )
    )
    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches="tight")
