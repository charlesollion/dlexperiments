import matplotlib.pyplot as plt
import jax.numpy as np

def display_burgers_grid(batched_u, resolution):
    """
    Display prediction grid u(t, x) for t in [0,1] and x in [-1,1]
    """
    nt, nx = resolution,resolution
    t = np.linspace(0, 1, nt)
    x = np.linspace(-1, 1, nx)
    tv, xv = np.meshgrid(t, x)
    tv = tv.reshape((nt * nx, 1))
    xv = xv.reshape((nt * nx, 1))

    values = batched_u(tv, xv)
    grid = values.reshape(nt, nx)
    plt.imshow(grid)
    plt.xticks([x*resolution/5 for x in range(5)], [round(i/(1.0*5), 2) for i in range(5)])
    plt.yticks([x*resolution/5 for x in range(5)], [round(1.0-2*i/(5), 2) for i in range(5)])
    plt.colorbar()


def display_burgers_slice(batched_u, resolution=30, slices=[0.25, 0.5, 0.75]):
    num = len(slices)
    plt.figure(figsize=(10,6))
    for i in range(num):
        plt.subplot(1, num,i+1)
        t = np.ones((resolution, 1)) * slices[i]
        x = np.expand_dims(np.linspace(-1,1, resolution),-1)
        values = batched_u(t, x)
        plt.plot(values)
        plt.ylim((-1, 1))


def display_KPP_at_times(batched_u, resolution=30, times=[0.0,]):
    num = len(times)
    plt.figure(figsize=(10,6))
    for i in range(num):
        plt.subplot(1, num,i+1)
        tt = np.ones((resolution, 1)) * times[i]
        xx = np.expand_dims(np.linspace(0,1, resolution),-1)
        yy = np.expand_dims(np.linspace(0,1, resolution),-1)
        map_out=[]

        for ix in range(resolution):
            xrow = np.expand_dims(np.repeat(xx[ix], resolution),axis=-1)
            row = np.concatenate((xrow,yy), axis=-1)
            outp = batched_u(tt, row)
            #outp = u0(row)
            map_out.append(np.expand_dims(outp, axis=-1))
        map_out = np.hstack(map_out)
        plt.imshow(map_out, vmin=-0.05, vmax=1.0)
        plt.xticks([x*resolution/5 for x in range(5)], [round(i/5.0, 2) for i in range(5)])
        plt.yticks([x*resolution/5 for x in range(5)], [round(i/5.0, 2) for i in range(5)])

    plt.colorbar()
