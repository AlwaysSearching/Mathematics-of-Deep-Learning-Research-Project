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
    
    weight_change_l2 = []
    weight_change_inf = []
    
    for model_id, history in metrics.items():
        # all models are named in the form 'conv_net_depth_{depth}_width_{init_channels}'
        # or ResNet18_width_{width}_UniformHe_init.
        width = list(filter(lambda x: x.isnumeric(), model_id.split("_")))[-1]
        widths.append(int(width))

        train_losses.append(history.get("loss"))
        train_accuracy.append(history.get("accuracy"))
        test_losses.append(history.get("val_loss"))
        test_accuracy.append(history.get("val_accuracy"))
        weight_change_l2.append(history.get("weight_change_l2"))
        weight_change_inf.append(history.get("weight_change_inf"))

    train_losses = np.array(train_losses)
    train_accuracy = np.array(train_accuracy)
    test_losses = np.array(test_losses)
    test_accuracy = np.array(test_accuracy)
    weight_change_l2 = np.array(weight_change_l2)
    weight_change_inf = np.array(weight_change_inf)

    return {
        "widths": widths,
        "loss": train_losses,
        "accuracy": train_accuracy,
        "val_loss": test_losses,
        "val_accuracy": test_accuracy,
        "weight_change_l2": weight_change_l2,
        "weight_change_inf": weight_change_inf
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
    test_loss_plt.plot(
        widths,
        train_losses[:, -1],
        marker="o",
        linestyle="dotted",
        markersize=mrkr_size,
        label="Final Training Loss",
        color="green"
    )
    test_loss_plt.set_ylabel("Loss", fontsize=16)

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
    line1 = test_accy_plt.plot(
        widths,
        100 * (1 - test_accuracy[:, -1]),
        marker="o",
        markersize=mrkr_size,
        label="Final Test Error",
    )
    line2 = test_accy_plt.plot(
        widths,
        100 * (1 - optimal_early_test_accuracy),
        marker="o",
        markersize=mrkr_size,
        label="Optimal Early Stopping Test Error",
    ) 
    ax2 = test_accy_plt.twinx()
    line3 = ax2.plot(
        widths,
        train_losses[:, -1],
        marker="o",
        linestyle="dotted",
        markersize=mrkr_size,
        label="Final Training Loss",
        color="green"
    )
    ax2.set_ylabel('Final Training Loss', fontsize=16)

    lines = line1+line2+line3
    labs = [l.get_label() for l in lines]
    test_accy_plt.set_ylabel("Test Error", fontsize=16)
    
    for ax in axes.flatten():
        ax.set_xlabel("Layer Width", fontsize=16)
        ax.legend(lines, labs, fontsize=12) if ax == test_accy_plt else ax.legend(fontsize=12) 
        ax.grid(alpha=0.5)

    fig.tight_layout(pad=1.15, h_pad=2)
    plt.show()

def plot_loss_from_file_overlay(paths, alphas):
    """
    Function to plot the results from previous runs stored in the experimental_results folder.
    
    paths should be a list of paths
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
    
    alphas should be a list of alpha values corresponding to the paths

    This is a dictionary where the key is the model id generated by the _get() function for the desired model, and
    the items being the history returned by calling model.fit().
    """
    data_arr = []
    for path in paths:
        results = load_results(path)
        data = {}
        data['widths'] = results.get("widths")
        data['train_losses'] = results.get("loss")
        data['train_accuracy'] = results.get("accuracy")
        data['test_losses'] = results.get("val_loss")
        data['test_accuracy'] = results.get("val_accuracy")

        # optimal early stopping values
        data['optimal_test_idx'] = data['test_accuracy'].argmax(axis=1)
        data['optimal_early_train_losses'] = np.array(
            [data['train_losses'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data['optimal_early_train_accuracy'] = np.array(
            [data['train_accuracy'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data['optimal_early_test_losses'] = np.array(
            [data['test_losses'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data['optimal_early_test_accuracy'] = np.array(
            [data['test_accuracy'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data_arr.append(data)
        
    fig, axes = plt.subplots(nrows=2, ncols=len(data_arr), figsize=(8*len(data_arr), 12))
    
    for data_idx in range(len(data_arr)):
        if len(data_arr)==1:
            loss_plt = axes[0]
            accy_plt = axes[1]
        else:
            loss_plt = axes[0][data_idx]
            accy_plt = axes[1][data_idx]
        
        widths = data_arr[data_idx]['widths']
        train_losses = data_arr[data_idx]['train_losses']
        train_accuracy = data_arr[data_idx]['train_accuracy']
        test_losses = data_arr[data_idx]['test_losses']
        test_accuracy = data_arr[data_idx]['test_accuracy']
        optimal_test_idx = data_arr[data_idx]['optimal_test_idx']
        optimal_early_train_losses = data_arr[data_idx]['optimal_early_train_losses']
        optimal_early_train_accuracy = data_arr[data_idx]['optimal_early_train_accuracy']
        optimal_early_test_losses = data_arr[data_idx]['optimal_early_test_losses']
        optimal_early_test_accuracy = data_arr[data_idx]['optimal_early_test_accuracy']
        alpha = alphas[data_idx]
        
        mrkr_size = 2
        
        loss_plt.title.set_text('alpha=%s' %str(alpha))
        loss_plt.title.set_size(16)
        
        # plot loss
        loss_plt.plot(
            widths,
            test_losses[:, -1],
            marker="o",
            markersize=mrkr_size,
            label="Final Test Loss",
        )
        loss_plt.plot(
            widths,
            optimal_early_test_losses,
            marker="o",
            markersize=mrkr_size,
            label="Optimal Early Stopping Test Loss",
        )
        loss_plt.plot(
            widths,
            train_losses[:, -1],
            marker="o",
            linestyle="dotted",
            markersize=mrkr_size,
            label="Final Training Loss",
            color="green"
        )
        loss_plt.set_ylabel("Loss", fontsize=12)
        loss_plt.set_xlabel("Layer Width", fontsize=12)
        loss_plt.legend(fontsize=12) 
        loss_plt.grid(alpha=0.5)
        

        # plot error
        line1 = accy_plt.plot(
            widths,
            100 * (1 - test_accuracy[:, -1]),
            marker="o",
            markersize=mrkr_size,
            label="Final Test Error",
        )
        line2 = accy_plt.plot(
            widths,
            100 * (1 - optimal_early_test_accuracy),
            marker="o",
            markersize=mrkr_size,
            label="Optimal Early Stopping Test Error",
        ) 
        ax2 = accy_plt.twinx()
        line3 = ax2.plot(
            widths,
            train_losses[:, -1],
            marker="o",
            linestyle="dotted",
            markersize=mrkr_size,
            label="Final Training Loss",
            color="green"
        )
        ax2.set_ylabel('Final Training Loss', fontsize=12)
        accy_plt.set_ylim([0, 100])
    
        lines = line1+line2+line3
        labs = [l.get_label() for l in lines]
        accy_plt.set_ylabel("Test Error", fontsize=12)
        accy_plt.set_xlabel("Layer Width", fontsize=12)
        accy_plt.legend(lines, labs, fontsize=12)
        accy_plt.grid(alpha=0.5)

    fig.tight_layout(pad=1.15, h_pad=2)
    plt.show()
    
def plot_weight_from_file_overlay(paths, alphas, layer_reduction='mean', index=None, scale=(120,20), include='loss'):
    """
    Function to plot the results from previous runs stored in the experimental_results folder.
    
    paths should be a list of paths
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
    
    alphas should be a list of alpha values corresponding to the paths
    
    layer_reduction chooses how the weight changes per layer would be aggregated (mean, sum, index)
    
    scale is a tuple containing the maximum y-axis limit for l2 and inf norms
    
    include specify whether to show loss or error along with the weight changes

    This is a dictionary where the key is the model id generated by the _get() function for the desired model, and
    the items being the history returned by calling model.fit().
    """
    data_arr = []
    for path in paths:
        results = load_results(path)
        data = {}
        data['widths'] = results.get("widths")
        data['train_losses'] = results.get("loss")
        data['train_accuracy'] = results.get("accuracy")
        data['test_losses'] = results.get("val_loss")
        data['test_accuracy'] = results.get("val_accuracy")
        data['weight_change_l2'] = results.get("weight_change_l2")
        data['weight_change_inf'] = results.get("weight_change_inf")
        
        if layer_reduction == 'mean':
            data['weight_change_l2'] = np.mean(data['weight_change_l2'], axis=2)
            data['weight_change_inf'] = np.mean(data['weight_change_inf'], axis=2)
        if layer_reduction == 'sum':
            data['weight_change_l2'] = np.sum(data['weight_change_l2'], axis=2)
            data['weight_change_inf'] = np.sum(data['weight_change_inf'], axis=2)
        if layer_reduction == 'index':
            if index==None: raise Exception('index must be specified when reducing weight change of layers by index')
            data['weight_change_l2'] = data['weight_change_l2'][:,:,index]
            data['weight_change_inf'] = data['weight_change_inf'][:,:,index]

        # optimal early stopping values
        data['optimal_test_idx'] = data['test_accuracy'].argmax(axis=1)
        data['optimal_early_train_losses'] = np.array(
            [data['train_losses'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data['optimal_early_train_accuracy'] = np.array(
            [data['train_accuracy'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data['optimal_early_test_losses'] = np.array(
            [data['test_losses'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        data['optimal_early_test_accuracy'] = np.array(
            [data['test_accuracy'][i, idx] for i, idx in enumerate(data['optimal_test_idx'])]
        )
        
        data_arr.append(data)
        
    fig, axes = plt.subplots(nrows=2, ncols=len(data_arr), figsize=(8*len(data_arr), 12))
    
    for data_idx in range(len(data_arr)):
        if len(data_arr)==1:
            l2_plt = axes[0]
            inf_plt = axes[1]
        else:
            l2_plt = axes[0][data_idx]
            inf_plt = axes[1][data_idx]
        
        widths = data_arr[data_idx]['widths']
        train_losses = data_arr[data_idx]['train_losses']
        train_accuracy = data_arr[data_idx]['train_accuracy']
        test_losses = data_arr[data_idx]['test_losses']
        test_accuracy = data_arr[data_idx]['test_accuracy']
        optimal_test_idx = data_arr[data_idx]['optimal_test_idx']
        optimal_early_train_losses = data_arr[data_idx]['optimal_early_train_losses']
        optimal_early_train_accuracy = data_arr[data_idx]['optimal_early_train_accuracy']
        optimal_early_test_losses = data_arr[data_idx]['optimal_early_test_losses']
        optimal_early_test_accuracy = data_arr[data_idx]['optimal_early_test_accuracy']
        
        weight_change_l2 = data_arr[data_idx]['weight_change_l2']
        weight_change_inf = data_arr[data_idx]['weight_change_inf']
        
#         print('w', weight_change_l2.shape)
#         print('w', weight_change_inf.shape)
#         print('loss', train_losses.shape)
#         print('acc', train_accuracy.shape)

        if np.any(weight_change_l2==None):
            raise Exception('l2_weights_change data missing from %s' %paths[data_idx])
        if np.any(weight_change_inf==None):
            raise Exception('inf_weights_change data missing from %s' %paths[data_idx])
        
        alpha = alphas[data_idx]
        
        mrkr_size = 2
        
        l2_plt.title.set_text('alpha=%s' %str(alpha))
        l2_plt.title.set_size(16)
        
        # plot l2 weight movements
        line1 = l2_plt.plot(
            widths,
            test_losses[:, -1] if include=='loss' else 100 * (1 - test_accuracy[:, -1]),
            marker="o",
            markersize=mrkr_size,
            label="Final Test Loss" if include=='loss' else "Final Test Error"
        )
        line2 = l2_plt.plot(
            widths,
            train_losses[:, -1] if include=='loss' else 100 * (1 - train_accuracy[:, -1]),
            marker="o",
            markersize=mrkr_size,
            label="Final Training Loss" if include=='loss' else "Final Training Error"
        )
        if include=='error': l2_plt.set_ylim([0,100])
        
        ax2 = l2_plt.twinx()
        line3 = ax2.plot(
            widths,
            weight_change_l2[:, -1],
            marker="o",
            linestyle="dotted",
            markersize=mrkr_size,
            color="green",
            label="l2-normed Weight Movements From Initialization"
        )
        ax2.set_ylabel('norm of Weight Movements', fontsize=12)
        ax2.set_ylim([0,scale[0]])
    
        lines = line1+line2+line3
        labs = [l.get_label() for l in lines]
        l2_plt.set_ylabel("Loss" if include=='loss' else "Error", fontsize=12)
        l2_plt.set_xlabel("Layer Width", fontsize=12)
        l2_plt.legend(lines, labs, fontsize=12)
        l2_plt.grid(alpha=0.5)
        

        # plot inf weight movements
        line1 = inf_plt.plot(
            widths,
            test_losses[:, -1] if include=='loss' else 100 * (1 - test_accuracy[:, -1]),
            marker="o",
            markersize=mrkr_size,
            label="Final Test Loss" if include=='loss' else "Final Test Error"
        )
        line2 = inf_plt.plot(
            widths,
            train_losses[:, -1] if include=='loss' else 100 * (1 - train_accuracy[:, -1]),
            marker="o",
            markersize=mrkr_size,
            label="Final Training Loss" if include=='loss' else "Final Training Error"
        )
        if include=='error': inf_plt.set_ylim([0,100])
        
        ax2 = inf_plt.twinx()
        line3 = ax2.plot(
            widths,
            weight_change_inf[:, -1],
            marker="o",
            linestyle="dotted",
            markersize=mrkr_size,
            color="green",
            label="infinity-normed Weight Movements From Initialization"
        )
        ax2.set_ylabel('norm of Weight Movements', fontsize=12)
        ax2.set_ylim([0,scale[1]])
        
        lines = line1+line2+line3
        labs = [l.get_label() for l in lines]
        inf_plt.set_ylabel("Loss" if include=='loss' else "Error", fontsize=12)
        inf_plt.set_xlabel("Layer Width", fontsize=12)
        inf_plt.legend(lines, labs, fontsize=12)
        inf_plt.grid(alpha=0.5)

    fig.tight_layout(pad=1.15, h_pad=2)
    plt.show()

# plot ONLY the weight changes for the different alpha values in the same graph
def plot_weights(paths, alphas, layer_reduction='mean', index=None, scale=None):
    """
    Function to plot the results from previous runs stored in the experimental_results folder.
    
    paths should be a list of paths
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
    
    alphas should be a list of alpha values corresponding to the paths
    
    layer_reduction chooses how the weight changes per layer would be aggregated (mean, sum, index)
    
    scale is a tuple containing the maximum y-axis limit for l2 and inf norms

    This is a dictionary where the key is the model id generated by the _get() function for the desired model, and
    the items being the history returned by calling model.fit().
    """
    data_arr = []
    for path in paths:
        results = load_results(path)
        data = {}
        data['widths'] = results.get("widths")
        data['weight_change_l2'] = results.get("weight_change_l2")
        data['weight_change_inf'] = results.get("weight_change_inf")
        
        if layer_reduction == 'mean':
            data['weight_change_l2'] = np.mean(data['weight_change_l2'], axis=2)
            data['weight_change_inf'] = np.mean(data['weight_change_inf'], axis=2)
        if layer_reduction == 'sum':
            data['weight_change_l2'] = np.sum(data['weight_change_l2'], axis=2)
            data['weight_change_inf'] = np.sum(data['weight_change_inf'], axis=2)
        if layer_reduction == 'index':
            if index==None: raise Exception('index must be specified when reducing weight change of layers by index')
            data['weight_change_l2'] = data['weight_change_l2'][:,:,index]
            data['weight_change_inf'] = data['weight_change_inf'][:,:,index]
        
        data_arr.append(data)
        
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    
    l2_plt = axes[0]
    inf_plt = axes[1]
    
    for data_idx in range(len(data_arr)):
        widths = data_arr[data_idx]['widths']
        weight_change_l2 = data_arr[data_idx]['weight_change_l2']
        weight_change_inf = data_arr[data_idx]['weight_change_inf']

        if np.any(weight_change_l2==None):
            raise Exception('l2_weights_change data missing from %s' %paths[data_idx])
        if np.any(weight_change_inf==None):
            raise Exception('inf_weights_change data missing from %s' %paths[data_idx])
        
        alpha = alphas[data_idx]
        
        mrkr_size = 2
        
        # plot l2 weight movements
        l2_plt.plot(
            widths,
            weight_change_l2[:, -1],
            marker="o",
            markersize=mrkr_size,
            label="alpha=%.2f" %alpha
        )
        
        # plot inf weight movements
        inf_plt.plot(
            widths,
            weight_change_inf[:, -1],
            marker="o",
            markersize=mrkr_size,
            label="alpha=%.2f" %alpha
        )
    if scale: l2_plt.set_ylim([0,scale[0]])
    l2_plt.set_ylabel("l2-norm of Weight Movements", fontsize=12)
    l2_plt.set_xlabel("Layer Width", fontsize=12)
    l2_plt.legend(fontsize=12) 
    l2_plt.grid(alpha=0.5)
    
    if scale: inf_plt.set_ylim([0,scale[1]])
    inf_plt.set_ylabel("inf-norm of Weight Movements", fontsize=12)
    inf_plt.set_xlabel("Layer Width", fontsize=12)
    inf_plt.legend(fontsize=12) 
    inf_plt.grid(alpha=0.5)
        
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
