import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

def plot_robot(ax, q, link_lengths=(2, 2)):
    l1, l2 = link_lengths
    theta1, theta2 = q
    j1 = np.array([l1 * math.cos(theta1), l1 * math.sin(theta1)])
    j2 = j1 + np.array([l2 * math.cos(theta1 + theta2), l2 * math.sin(theta1 + theta2)])
    ax.plot([0, j1[0], j2[0]], [0, j1[1], j2[1]], 'o-', linewidth=3)

def animate_simulation_combined(states, collision_point):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame):
        ax1.cla()
        ax1.set_xlim([-5, 5])
        ax1.set_ylim([-5, 5])
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Workspace")
        
        # workspace中的障碍物
        circle = plt.Circle((2.5, 2.5), 0.5, color='blue', fill=True, alpha=0.5)
        ax1.add_artist(circle)

        plot_robot(ax1, states[frame])

        ax2.cla()
        ax2.set_xlim(-np.pi, np.pi)
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_xlabel("q1")
        ax2.set_ylabel("q2")
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title("C-space")

        # 统一颜色为蓝色半透明
        ax2.scatter(collision_point[:, 0], collision_point[:, 1], color='blue', s=1, alpha=0.5)
        ax2.plot(states[:frame+1, 0], states[:frame+1, 1], color='#1f77b4', linewidth=2)

    anim = FuncAnimation(fig, update, frames=len(states), interval=100)
    anim.save('robot_animation.gif', writer='pillow')
    plt.tight_layout()
    
    plt.show()

# === Load data and run animation ===
states = np.load('states.npy')              # shape: (N, 2)
collision_point = np.load('collision_point.npy')  # shape: (M, 2)

animate_simulation_combined(states, collision_point)
