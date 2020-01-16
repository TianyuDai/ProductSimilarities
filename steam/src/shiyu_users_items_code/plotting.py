import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def histogram_plot(sorted_data_sequence, title=None, bins=None, y_lim=None):
    fig, ax = plt.subplots(1, 1)
    ax.hist(sorted_data_sequence, bins=bins)
    if y_lim:
        ax.set_ylim(y_lim)
    if title:
        ax.set_title(title)


def bar_histogram_plot(sorted_data_sequence, title=None, x_lim=None, y_lim=None):
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


def bar_plot(data_dict, title=None, x_lim=None, y_lim=None):
    fig, ax = plt.subplots(1, 1)
    x_loc = np.arange(len(data_dict))
    x_label = list(data_dict.keys())
    x_value = list(data_dict.values())
    ax.bar(x_loc, x_value, tick_label=x_label)
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


def scatter2d_plot(data_array, marker_size=None, color=None, x_lim=None, y_lim=None, title=None):
    fig, ax = plt.subplots(1, 1)
    # color_seq = color.reshape([-1, 1])
    ax.scatter(data_array[:, 0], data_array[:, 1], s=marker_size, facecolor=color, edgecolor=color)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if title:
        ax.set_title(title)


def scatter3d_plot(data_array, marker_size=None, color=None, x_lim=None, y_lim=None, z_lim=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], s=marker_size, facecolor=color, edgecolor=color)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if z_lim:
        ax.set_zlim(z_lim)
    if title:
        ax.set_title(title)


def line_plot(data_array_dict, y_lim=None, title=None):
    fig, ax = plt.subplots(1, 1)
    for label, data_array in data_array_dict.items():
        ax.plot(data_array[0, :], data_array[1, :], label=label)
    ax.legend()
    if y_lim:
        ax.set_ylim(y_lim)
    if title:
        ax.set_title(title)
