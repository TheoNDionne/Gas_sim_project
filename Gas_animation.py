import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.load("Python/Gas_project/gas_simulation_result.npy")

fig, ax = plt.subplots()

# initialize proper stuff
scat = ax.scatter(data[:,0,0], data[:,1,0], s=10)

ax.set_aspect("equal")
ax.set_xlim((0,1))
ax.set_ylim((0,1))

# define the animation function
def func(frame):
    ax.clear()

    ax.set_aspect("equal")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))

    ax.scatter(data[:,0,frame], data[:,1,frame], s=10)

anim = FuncAnimation(fig, func, frames=100, interval=100)

plt.show()