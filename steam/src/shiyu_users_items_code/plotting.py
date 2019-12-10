import numpy as np
import matplotlib.pyplot as plt


def histogram_plot(sorted_data_sequence, title=None):
    fig, ax = plt.subplots(1, 1)
    ax.hist(sorted_data_sequence)
    plt.show()


def bar_plot(sorted_data_sequence, title=None, x_lim=None, y_lim=None):
    fig, ax = plt.subplots(1, 1)
    total_count = len(sorted_data_sequence)
    x_loc = np.arange(total_count)
    ax.bar(x_loc, sorted_data_sequence, width=1)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if title:
        ax.set_title(title)


def heatmap_plot(data_array, title=None, min_value=None, max_value=None):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data_array, cmap='Blues', vmin=min_value, vmax=max_value)
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)


def violin_plot(data_array, y_lim=None, label=None):
    fig, ax = plt.subplots(1, 1)
    ax.violinplot([data_array], widths=0.8, showextrema=False, points=2000)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xticks([1])
    if label:
        ax.set_xticklabels([label])
    else:
        ax.set_xticklabels([''])


def box_plot(data_array, y_lim=None):
    fig, ax = plt.subplots(1, 1)
    ax.boxplot(data_array)
    if y_lim:
        ax.set_ylim(y_lim)

