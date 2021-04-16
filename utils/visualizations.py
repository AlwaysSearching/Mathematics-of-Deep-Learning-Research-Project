import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib

import numpy as np
import pickle as pkl


def load_results(path):
    """
    Helper method to read in and format the results from training convnets.
    returns dictionary of form:
    {
        'widths': np.array,
        'loss': np.array,
        'accuracy': np.array,
        'val_loss': np.array,
        'val_accuracy': np.array
    }
    """

    metrics = pkl.load(open(path, "rb"))

    # extract train/test loss and accuracy  for each model.
    widths = []
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []

    for model_id, history in metrics.items():
        # all models are named in the form 'conv_net_depth_{depth}_width_{init_channels}'
        # or ResNet18_width_{width}_UniformHe_init.
        width = list(filter(lambda x: x.isnumeric(), model_id.split("_")))[-1]
        widths.append(int(width))

        train_losses.append(history.get("loss"))
        train_accuracy.append(history.get("accuracy"))
        test_losses.append(history.get("val_loss"))
        test_accuracy.append(history.get("val_accuracy"))

    train_losses = np.array(train_losses)
    train_accuracy = np.array(train_accuracy)
    test_losses = np.array(test_losses)
    test_accuracy = np.array(test_accuracy)

    return {
        "widths": widths,
        "loss": train_losses,
        "accuracy": train_accuracy,
        "val_loss": test_losses,
        "val_accuracy": test_accuracy,
    }


def plot_loss_from_file(path):
    """
    Function to plot the results from previous runs stored in the experimental_results folder.

    The path should point to a pickled dictionary with the following structure:
    {
        'model_id'{
            'loss': list,
            'accuracy': list,
            'val_loss': list,
            'val_accuracy': list,
            ...
        }
    }

    This is a dictionary where the key is the model id generated by the _get() function for the desired model, and
    the items being the history returned by calling model.fit().
    """

    results = load_results(path)

    widths = results.get("widths")
    train_losses = results.get("loss")
    train_accuracy = results.get("accuracy")
    test_losses = results.get("val_loss")
    test_accuracy = results.get("val_accuracy")

    # optimal early stopping values
    optimal_test_idx = test_accuracy.argmax(axis=1)
    optimal_early_train_losses = np.array(
        [train_losses[i, idx] for i, idx in enumerate(optimal_test_idx)]
    )
    optimal_early_train_accuracy = np.array(
        [train_accuracy[i, idx] for i, idx in enumerate(optimal_test_idx)]
    )
    optimal_early_test_losses = np.array(
        [test_losses[i, idx] for i, idx in enumerate(optimal_test_idx)]
    )
    optimal_early_test_accuracy = np.array(
        [test_accuracy[i, idx] for i, idx in enumerate(optimal_test_idx)]
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

    train_loss_plt = axes[0][0]
    test_loss_plt = axes[0][1]
    train_accy_plt = axes[1][0]
    test_accy_plt = axes[1][1]

    mrkr_size = 2

    # plot final and optimal early stopping train loss
    train_loss_plt.plot(
        widths,
        train_losses[:, -1],
        marker="o",
        markersize=mrkr_size,
        label="Final Train Loss",
    )
    train_loss_plt.plot(
        widths,
        optimal_early_train_losses,
        marker="o",
        markersize=mrkr_size,
        label="Optimal Early Stopping Train Loss",
    )
    train_loss_plt.set_ylabel("Train Loss", fontsize=16)

    # plot final and optimal early stopping test loss
    test_loss_plt.plot(
        widths,
        test_losses[:, -1],
        marker="o",
        markersize=mrkr_size,
        label="Final Test Loss",
    )
    test_loss_plt.plot(
        widths,
        optimal_early_test_losses,
        marker="o",
        markersize=mrkr_size,
        label="Optimal Early Stopping Test Loss",
    )
    test_loss_plt.set_ylabel("Test Loss", fontsize=16)

    # plot final and optimal early stopping train error
    train_accy_plt.plot(
        widths,
        100 * (1 - train_accuracy[:, -1]),
        marker="o",
        markersize=mrkr_size,
        label="Final Train Error",
    )
    train_accy_plt.plot(
        widths,
        100 * (1 - optimal_early_train_accuracy),
        marker="o",
        markersize=mrkr_size,
        label="Optimal Early Stopping Train Error",
    )
    train_accy_plt.set_ylabel("Train Error", fontsize=16)

    # plot final and optimal early stopping test error
    test_accy_plt.plot(
        widths,
        100 * (1 - test_accuracy[:, -1]),
        marker="o",
        markersize=mrkr_size,
        label="Final Test Error",
    )
    test_accy_plt.plot(
        widths,
        100 * (1 - optimal_early_test_accuracy),
        marker="o",
        markersize=mrkr_size,
        label="Optimal Early Stopping Test Error",
    )
    test_accy_plt.set_ylabel("Test Error", fontsize=16)

    for ax in axes.flatten():
        ax.set_xlabel("Layer Width", fontsize=16)
        ax.legend(fontsize=14)
        ax.grid(alpha=0.5)

    fig.tight_layout(pad=1.15, h_pad=2)
    plt.show()


def plot_loss_vs_epoch_from_file(path, x_idx, contour_levels=[0.1], save_fig=None):
    """
    Plots the Seaplots as seen on page 2 of Deep Double Descent. Plots them in the form of 1 1x2 matplotlib figure.

    Note: may need to add more adjustments to allow this to work for both convnets and resnets. possible pass in the needed
    data rather than passing the path.

    Parameters
    ----------

    path: str
        path to pickled file. see load load_results function
    x_idx: list
        list of tick marks for the x axis. Should correspond to the model width parameter.
    contour_levels: list[float]
        a list of the contour line values to add to the training loss plot.
    save_fig: str
        A file name to save the resulting image to.
    """

    results = load_results(path)

    widths = results.get("widths")
    train_losses = results.get("loss")
    train_accuracy = results.get("accuracy")
    test_losses = results.get("val_loss")
    test_accuracy = results.get("val_accuracy")

    fig, axes = plt.subplots(figsize=(15, 12), ncols=1, nrows=2)
    train_plot = axes[0]
    test_plot = axes[1]

    # 1e-15 is there since imshow sometimes raises errors for non-positive input.
    train_error = 1 - train_accuracy + 1e-15
    test_error = 1 - test_accuracy + 1e-15

    ax_label_fs = 14
    ax_label_pad = 15
    plt_title_fs = 18
    plt_title_pad = 15

    # normalize the color range relative to the input values
    vmin = np.min(train_error)
    vmax = np.max(train_error)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    # seaplot train data
    train_im = train_plot.imshow(
        train_error.T, aspect="auto", origin="lower", norm=norm, interpolation="nearest"
    )

    # set axis labels and title
    train_plot.set_xlabel(f"Width", fontsize=ax_label_fs, labelpad=ax_label_pad)
    train_plot.set_ylabel("Epoch", fontsize=ax_label_fs, labelpad=ax_label_pad)
    train_plot.set_title("Train", fontsize=plt_title_fs, pad=plt_title_pad)

    # set x axis ticks. (note that there is an offset of 1 since the array is 0 indexed)
    x_vals = [idx - 1 for idx in x_idx]
    x_labels = [f"{tick}" for tick in x_idx]
    train_plot.set_xticks(x_vals)
    train_plot.set_xticklabels(x_labels, fontsize=16)

    # set y ticks and adjust scale
    train_plot.set_yscale("symlog", linthresh=10)
    train_plot.set_yticks([i - 1 for i in [1, 10, 100, 1000]])
    train_plot.set_yticklabels([1, 10, 100, 1000], fontsize=12)

    # Colorbar
    ticks = [0.0, 0.1, 0.3, 0.5, 0.7]
    cbar = fig.colorbar(train_im, ax=train_plot, pad=0.025, ticks=ticks, fraction=0.1)
    cbar.ax.set_xlabel("% Error", labelpad=15)

    # add interpolation point contour and label
    if contour_levels is not None:
        train_plot.contour(
            train_error.T,
            levels=contour_levels,
            colors="white",
            linestyles="dashed",
            linewidths=1.5,
            alpha=0.5,
        )

    # normalize the color range relative to the input values
    vmin = round(np.min(test_error), 1)
    vmax = np.max(test_error)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    # Plot the test error sea plot
    test_im = test_plot.imshow(test_error.T, aspect="auto", norm=norm, origin="lower")

    # set axis labels and title
    test_plot.set_xlabel(f"Width", fontsize=ax_label_fs, labelpad=ax_label_pad)
    test_plot.set_ylabel("Epoch", fontsize=ax_label_fs, labelpad=ax_label_pad)
    test_plot.set_title("Test", fontsize=plt_title_fs, pad=plt_title_pad)

    # set x axis ticks (note that there is an offset of 1 since the array is 0 indexed)
    x_vals = [idx - 1 for idx in x_idx]
    x_labels = [f"{tick}" for tick in x_idx]
    test_plot.set_xticks(x_idx)
    test_plot.set_xticklabels(x_labels, fontsize=14)

    # set y ticks and adjust scale
    test_plot.set_yscale("symlog", linthresh=10)
    test_plot.set_yticks([i - 1 for i in [1, 10, 100, 1000]])
    test_plot.set_yticklabels([1, 10, 100, 1000], fontsize=12)

    # Colorbar
    ticks = [0.0, round(vmin, 1), 0.3, 0.5, 0.7]
    cbar = fig.colorbar(test_im, ax=test_plot, pad=0.025, ticks=ticks, fraction=0.1)
    cbar.ax.set_xlabel("% Error", labelpad=15)

    # display the image.
    plt.tight_layout(h_pad=2)
    plt.show()

    # if a path is provided, save image.
    if isinstance(save_fig, str):
        # check that path ends in png.
        save_fig = (
            save_fig
            if save_fig.endswith(".png") or save_fig.endswith(".jpg")
            else save_fig + ".png"
        )
        fig.savefig(save_fig, dpi=300)
