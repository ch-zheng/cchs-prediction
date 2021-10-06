import time
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np


def plot_face(sample):
    _, ax = plt.subplots()
    ax.scatter(sample[3::2], sample[4::2])
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    plt.show()

def plot_measure(X):
    data = []
    for row in X:
        a = row[3+2*39+1]
        b = row[3+2*42+1]
        c = row[4+2*8] - stat.mean((a, b))
        data.append(c)

    _, axis = plt.subplots()
    axis.hist(data, bins=20)
    plt.show()

def test():
    X = np.load('data/individualized/samples.npy')

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots()

    for row in X:
        ax.clear()
        ax.scatter(row[3::2], row[4::2])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)

test()
