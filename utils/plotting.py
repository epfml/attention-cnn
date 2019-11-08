import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Rectangle
import itertools


def plot_grid_query_pix(width, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.set_xticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.set_aspect(1)
    ax.set_yticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, alpha=0.5)

    # query pixel
    querry_pix = Rectangle(xy=(-0.5,-0.5),
                          width=1,
                          height=1,
                          edgecolor="black",
                          fc='None',
                          lw=2)

    ax.add_patch(querry_pix);

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)
    ax.set_aspect("equal")

def plot_attention_layer(model, layer_idx, width, ax=None):
    """Plot the 2D attention probabilities of all heads on an image
    of layer layer_idx
    """
    if ax is None:
        fig, ax = plt.subplots()

    attention = model.encoder.layer[layer_idx].attention.self
    attention_probs = attention.get_attention_probs(width + 2, width + 2)

    contours = np.array([0.9, 0.5])
    linestyles = [":", "-"]
    flat_colors = ["#3498db", "#f1c40f", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#34495e", "#1abc9c", "#95a5a6"]

    if ax is None:
        fig, ax = plt.subplots()

    shape = attention_probs.shape
    # remove batch size if present
    if len(shape) == 6:
        shape = shape[1:]
    height, width, num_heads, _, _ = shape

    attention_at_center = attention_probs[width // 2, height // 2]
    attention_at_center = attention_at_center.detach().cpu().numpy()

#     compute integral of distribution for thresholding
    n = 1000
    t = np.linspace(0, attention_at_center.max(), n)
    integral = ((attention_at_center >= t[:, None, None, None]) * attention_at_center).sum(
        axis=(-1, -2)
    )

    plot_grid_query_pix(width - 2, ax)

    for h, color in zip(range(num_heads), itertools.cycle(flat_colors)):
        f = interpolate.interp1d(integral[:, h], t, fill_value=(1, 0), bounds_error=False)
        t_contours = f(contours)

        # remove duplicate contours if any
        keep_contour = np.concatenate([np.array([True]), np.diff(t_contours) > 0])
        t_contours = t_contours[keep_contour]

        for t_contour, linestyle in zip(t_contours, linestyles):
            ax.contour(
                np.arange(-width // 2, width // 2) + 1,
                np.arange(-height // 2, height // 2) + 1,
                attention_at_center[h],
                [t_contour],
                extent=[- width // 2, width // 2 + 1, - height // 2, height // 2 + 1],
                colors=color,
                linestyles=linestyle
            )

    return ax


def plot_attention_positions_all_layers(model, width, tensorboard_writer=None, global_step=None):

    for layer_idx in range(len(model.encoder.layer)):
        fig, ax = plt.subplots()
        plot_attention_layer(model, layer_idx, width, ax=ax)

        ax.set_title(f"Layer {layer_idx + 1}")
        if tensorboard_writer:
            tensorboard_writer.add_figure(f"attention/layer{layer_idx}", fig, global_step=global_step)
        plt.close(fig)
