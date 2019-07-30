import matplotlib.pyplot as plt


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
    width, height, num_heads, _, _ = shape

    attention_at_center = attention_probs[width // 2, height // 2]
    attention_at_center = attention_at_center.detach().cpu()

    for h in range(num_heads):
        cs = ax.contour(attention_at_center[h], levels=[0.1, 0.4], colors=f"C{h}")

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_aspect(aspect=1)
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
