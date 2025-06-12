#!/usr/bin/env python3
"""
$ python -m environment.viewer
Opens a Matplotlib window with the 3-D scene – useful for a quick check.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from environment import Environment


def main() -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    Environment().draw_environment(ax, view="3d")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
