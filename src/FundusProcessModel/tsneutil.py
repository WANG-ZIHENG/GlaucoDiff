from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp
import random

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")


def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


# MOUSE_10X_COLORS = {}
# for i in range(200):
#     MOUSE_10X_COLORS[i] = generate_random_color()


MOUSE_10X_COLORS = {0: '#FFFF00', 1: '#1CE6FF', 2: '#FF34FF', 3: '#FF4A46', 4: '#008941', 5: '#006FA6', 6: '#A30059',
                    7: '#FFDBE5', 8: '#7A4900', 9: '#0000A6', 10: '#63FFAC', 11: '#B79762', 12: '#004D43',
                    13: '#8FB0FF', 14: '#997D87', 15: '#5A0007', 16: '#809693', 17: '#FEFFE6', 18: '#1B4400',
                    19: '#4FC601', 20: '#3B5DFF', 21: '#4A3B53', 22: '#FF2F80', 23: '#61615A', 24: '#BA0900',
                    25: '#6B7900', 26: '#00C2A0', 27: '#FFAA92', 28: '#FF90C9', 29: '#B903AA', 30: '#D16100',
                    31: '#DDEFFF', 32: '#000035', 33: '#7B4F4B', 34: '#A1C299', 35: '#300018', 36: '#0AA6D8',
                    37: '#013349', 38: '#00846F', 39: '#DEB887', 40: '#5F9EA0', 41: '#7FFF00', 42: '#D2691E',
                    43: '#6495ED', 44: '#DC143C', 45: '#B8860B', 46: '#696969', 47: '#2F4F4F', 48: '#FFD700',
                    49: '#808080', 50: '#008000', 51: '#F0E68C', 52: '#556B2F', 53: '#48D1CC', 54: '#B0C4DE',
                    55: '#4682B4', 56: '#FF69B4', 57: '#D2B48C', 58: '#BC8F8F', 59: '#20B2AA', 60: '#778899',
                    61: '#ADFF2F', 62: '#F08080', 63: '#E0FFFF', 64: '#D8BFD8', 65: '#FF6347', 66: '#40E0D0',
                    67: '#EE82EE', 68: '#F5DEB3', 69: '#F5F5DC', 70: '#F0FFF0', 71: '#E6E6FA', 72: '#FFF0F5',
                    73: '#7CFC00', 74: '#FFFACD', 75: '#ADD8E6', 76: '#FAFAD2', 77: '#FFDAB9', 78: '#8A2BE2',
                    79: '#00FFFF', 80: '#7FFFD4', 81: '#FFE4E1', 82: '#FA8072', 83: '#7B68EE', 84: '#BA55D3',
                    85: '#F4A460', 86: '#00FF7F', 87: '#87CEEB', 88: '#B0E0E6', 89: '#32CD32', 90: '#8B008B',
                    91: '#FF8C00', 92: '#4169E1', 93: '#8B4513', 94: '#2E8B57', 95: '#9932CC', 96: '#8FBC8F',
                    97: '#483D8B', 98: '#00CED1', 99: '#9400D3', 100: '#B22222', 101: '#FF1493'}


def plot(
        x,
        y,
        ax=None,
        title=None,
        draw_legend=True,
        draw_centers=False,
        draw_cluster_labels=False,
        colors=None,
        legend_kwargs=None,
        label_order=None,
        save_path="tsne.png",
        **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
        # matplotlib.pyplot.show()
        matplotlib.pyplot.savefig(save_path)
