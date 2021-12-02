import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = []
for i in range(108):
    frame = np.loadtxt(f"data_{i}.txt")
    data.append(frame)



fig, ax = plt.subplots()
xdata = np.arange(data[0].shape[0])
ydata = data[0]
ln, = plt.plot(xdata, ydata)


def init():
    plt.grid()
    plt.xlabel("X [-]")
    plt.ylabel("Temperature [-]")
    return ln,

def update(frame):
    # ydata = data[frame]
    print(frame)
    # ydata = np.ones_like(xdata)
    ln.set_data(xdata, data[frame])
    return ln,

ani = FuncAnimation(fig, update, frames=range(data[0].shape[0]), init_func=init, blit=True)
ani.save('myAnimation.gif', writer='imagemagick', fps=10)
plt.show()

plt.plot(xdata, data[0])
plt.show()