import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib.colors as mcolors


def plot_attention_contours(attention_probs, ax=None):
    """Plot the 2D attention probabilities of all heads on an image

    Args:
        attention_probs (tensor):
        ax (plt.Axes):
    """
    if ax is None:
        fig, ax = plt.subplots()

    shape = attention_probs.shape
    # remove batch size if present
    if len(shape) == 6:
        shape = shape[1:]
    height, width, num_heads, _, _ = shape

    attention_at_center = attention_probs[width // 2, height // 2]
    attention_at_center = attention_at_center.detach().cpu().numpy()

    contours = np.array([0.9, 0.5, 0.1])

    # compute transparency levels for the different contours
    min_contour = contours.min()
    max_contour = contours.max()
    min_alpha = 0.3
    alphas = min_alpha + (1 - min_alpha) * (1 - (contours - min_contour) / max_contour)

    # compute integral of distribution for thresholding
    n = 1000
    t = np.linspace(0, attention_at_center.max(), n)
    integral = ((attention_at_center >= t[:, None, None, None]) * attention_at_center).sum(
        axis=(-1, -2)
    )

    for h in range(num_heads):
        f = interpolate.interp1d(integral[:, h], t, fill_value=(1, 0), bounds_error=False)
        t_contours = f(contours)
        colors = [mcolors.to_rgba(f"C{h}", alpha) for alpha in alphas]

        # remove duplicate contours if any
        keep_contour = np.concatenate([np.array([True]), np.diff(t_contours) > 0])
        t_contours = t_contours[keep_contour]
        colors = [c for keep, c in zip(keep_contour, colors) if keep]

        ax.contour(attention_at_center[h], t_contours, extent=[0, width, 0, height], colors=colors)

    # draw the grid
    ax.set_xticks(np.arange(width))  # , minor=True)
    ax.set_aspect(1)
    ax.set_yticks(np.arange(height))  # , minor=True)
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    plt.grid(True, alpha=0.5)

    # draw red point at centered pixel
    ax.scatter([width // 2], [width // 2], c="r", zorder=2)

    return ax


def plot_attention_positions_all_layers(model, image_shape, tensorboard_writer, global_step):
    width, height = image_shape

    for i, layer in enumerate(model.encoder.layer):
        fig, ax = plt.subplots()
        attention = layer.attention.self
        attention_probs = attention.get_attention_probs(width, height)
        plot_attention_contours(attention_probs, ax=ax)
        ax.set_title(f"Layer {i}")
        tensorboard_writer.add_figure(f"attention/layer{i}", fig, global_step=global_step)
        plt.close(fig)
