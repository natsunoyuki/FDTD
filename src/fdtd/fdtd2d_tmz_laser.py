# Copyright 2025 Natsunoyuki.
#
# FDTD is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
#
# FDTD is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# FDTD. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm

# fdtd = fdtd2d_tmz_laser()
# fdtd.run()
# fdtd.plot()

# FDTD 2D, TMZ polarization laser simulation.
# The electric field oscillates only in the z direction, while the magnetic field
# oscillates only in the x and y directions. A staggered grid in space is used to 
# simulate E_z, H_x and H_y:
#     H_x(m, n+1/2) 
#     E_z(m, n)     H_y(m+1/2, n)

class fdtd2d_tmz_laser:
    def __init__(self, Nx = 501, Ny = 501, c = 1, dx = 1, dy = 1, drill_holes = False):
        # Grid attributes
        self.Nx = Nx # Number of x grid cells.
        self.Ny = Ny # Number of y grid cells.
        self.c = c # Speed of light, normalized to 1.
        self.dx = dx # x step size.
        self.dy = dy # y step size.
        self.dt = min(dx, dy) / np.sqrt(2) / c # Time step size.

        self.x = np.arange(Nx)
        self.y = np.arange(Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Source location
        self.source_x = int(Nx / 2)
        self.source_y = int(Ny / 2)

        # Maxwell-Bloch equation parameters
        self.gperp = 0.01
        self.gpara = self.gperp / 100.0
        self.ka = 0.1
        
        # Magnetic fields H_x and H_y
        self.H_x = np.zeros([Nx, Ny - 1])
        self.H_y = np.zeros([Nx - 1, Ny])
        
        # Electric field E_z
        self.E_z = np.zeros([Nx, Ny])
        
        # Dielectric slab attributes
        self.radius = 200
        self.mask = np.zeros([Nx, Ny])
        for i in range(Nx):
            for j in range(Ny):
                if np.sqrt((i - self.source_x)**2 + (j - self.source_y)**2) < self.radius:
                    self.mask[i, j] = 1
                    
        self.drill_holes = drill_holes
        if self.drill_holes == True:
            self.sub_radius = 30
            self.centres = [[143, 137], 
                            [107, 254],
                            [415, 257],
                            [379, 165],
                            [165, 303],
                            [307, 122],
                            [225, 222],
                            [363, 328],
                            [322, 219],
                            [266, 389],
                            [231,  93]]
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                for i in range(-c_x, c_x + self.sub_radius):
                    for j in range(-c_y, c_y + self.sub_radius):
                        if np.sqrt((i - c_x)**2 + (j - c_y)**2) < self.sub_radius:
                            self.mask[i, j] = 0
        
        # Permittivity epsilon (E) and permeability mu (M)
        self.n1 = 1.0
        self.n2 = 3.0
        self.E = np.logical_not(self.mask) * self.n1**2 + self.mask * self.n2**2 
        # we assume that the dielectric is a perfect magnetic material with
        # mu = 1 everywhere. So we do not need to explicitly have a vector 
        # for mu. If this condition is not satisfied, then mu must be taken
        # into account as well in the simulation.
        
        # Polarization field P (strictly speaking P_z)
        self.P = np.zeros([Nx, Ny])
        self.Place = np.zeros([Nx, Ny]) # polarization one time step before
        self.Pold = np.zeros([Nx, Ny]) # polarization two time steps before
        
        # Population inversion D
        self.D0 = 10.0 # pump strength
        self.D = self.mask * self.D0
        
        # Time dependent field for plotting and animation
        self.E_z_t = []
        self.H_x_t = []
        self.H_y_t = []
        self.E_z_timeseries = []
        
        # Mur absorbing boundaries
        self.E_z_n = self.E_z.copy()     # data for t = n
        self.E_z_n_1 = self.E_z_n.copy() # data for t = n-1
        self.H_x_n = self.H_x.copy()     # data for t = n
        self.H_x_n_1 = self.H_x_n.copy() # data for t = n-1
        self.H_y_n = self.H_x.copy()     # data for t = n
        self.H_y_n_1 = self.H_y_n.copy() # data for t = n-1
        
    def run(self, n_iter = 10000, initiate_pulse = False, verbose = False):
        # MB equation constants
        c1 = 1.0 / self.dt ** 2 + self.gperp / self.dt / 2.0
        c2 = 2.0 / self.dt ** 2 - self.ka ** 2 - self.gperp ** 2
        c3 = 1.0 / self.dt ** 2 - self.gperp / self.dt / 2.0
        c4 = self.ka * self.gperp / 2.0 / np.pi
        c5 = 1.0 / self.dt + self.gpara / 2.0
        c6 = 1.0 / self.dt - self.gpara / 2.0
        c7 = 2.0 * np.pi * self.gpara / self.ka
        c8 = 1.0 / self.dt + self.gperp / 2.0
        c9 = -1.0 / self.dt + self.gperp / 2.0
        
        # Mur absorbing boundary constants
        dtdx = np.sqrt(self.dt / self.dx * self.dt / self.dy)
        dtdx_2 = 1 / dtdx + 2 + dtdx
        c_0 = -(1 / dtdx - 2 + dtdx) / dtdx_2
        c_1 = -2 * (dtdx - 1 / dtdx) / dtdx_2
        c_2 = 4 * (dtdx + 1 / dtdx) / dtdx_2
        
        # FDTD Loop
        for n in tqdm.trange(n_iter, disable = np.logical_not(verbose)):
            # Update magnetic fields H_x, H_y
            self.H_x = self.H_x - self.dt / self.dy * (self.E_z[:, 1:] - self.E_z[:, :-1])
            self.H_y = self.H_y + self.dt / self.dx * (self.E_z[1:, :] - self.E_z[:-1, :])

            # Update polarization field P
            self.P = self.mask / c1*(c2*self.P - c3*self.Pold - c4*self.E_z*self.D)
            self.Pold = self.Place.copy() 
            self.Place = self.P.copy() # carry the current value of P for two time steps
            
            # Update electric field E_z
            diff_H_x = self.dt / self.dy / self.E[1:-1, 1:-1] * (self.H_x[1:-1, 1:] - self.H_x[1:-1, :-1])
            diff_H_y = self.dt / self.dx / self.E[1:-1, 1:-1] * (self.H_y[1:, 1:-1] - self.H_y[:-1, 1:-1])
            diff_P = -4 * np.pi / self.E[1:-1, 1:-1] * (self.P[1:-1, 1:-1] - self.Pold[1:-1, 1:-1])
            self.E_z[1:-1, 1:-1] = self.E_z[1:-1, 1:-1] + diff_P + (diff_H_y - diff_H_x)
            
            # Update population inversion D
            self.D = self.mask / c5*(c6*self.D + self.gpara*self.D0 + c7*self.E_z*(c8*self.P + c9*self.Pold))
            
            # Pulse at time step 
            if initiate_pulse == True:
                tp = 30
                if n * self.dt <= tp:
                    C = (7 / 3) ** 3 * (7 / 4) ** 4
                    pulse = C * (n * self.dt / tp) ** 3 * (1 - n * self.dt / tp) ** 4
                else:
                    pulse = 0
                self.E_z[self.source_x, self.source_y] = self.E_z[self.source_x, self.source_y] + pulse

            # Mur ABC for top boundary
            self.E_z[0, :] = c_0 * (self.E_z[2, :] + self.E_z_n_1[0, :]) +    \
                             c_1 * (self.E_z_n[0, :] + self.E_z_n[2, :] -    \
                                    self.E_z[1, :] - self.E_z_n_1[1, :]) +    \
                             c_2 * self.E_z_n[1, :] - self.E_z_n_1[2, :]

            # Mur ABC for bottom boundary
            self.E_z[-1, :] = c_0 * (self.E_z[-3, :] + self.E_z_n_1[-1, :]) +    \
                              c_1 * (self.E_z_n[-1, :] + self.E_z_n[-3, :] -    \
                                     self.E_z[-2, :] - self.E_z_n_1[-2, :]) +    \
                              c_2 * self.E_z_n[-2, :] - self.E_z_n_1[-3, :]

            # Mur ABC for left boundary
            self.E_z[:, 0] = c_0 * (self.E_z[:, 2] + self.E_z_n_1[:, 0]) +    \
                             c_1 * (self.E_z_n[:, 0] + self.E_z_n[:, 2] -    \
                                    self.E_z[:, 1] - self.E_z_n_1[:, 1]) +    \
                             c_2 * self.E_z_n[:, 1] - self.E_z_n_1[:, 2]

            # Mur ABC for right boundary
            self.E_z[:, -1] = c_0 * (self.E_z[:, -3] + self.E_z_n_1[:, -1]) +    \
                              c_1 * (self.E_z_n[:, -1] + self.E_z_n[:, -3] -    \
                                     self.E_z[:, -2] - self.E_z_n_1[:, -2]) +    \
                              c_2 * self.E_z_n[:, -2] - self.E_z_n_1[:, -3]

            # Store magnetic and electric fields for ABC at time step n
            self.H_x_n_1 = self.H_x_n.copy() # data for t = n-1
            self.H_x_n = self.H_x.copy()     # data for t = n

            self.H_y_n_1 = self.H_y_n.copy() # data for t = n-1
            self.H_y_n = self.H_y.copy()     # data for t = n

            self.E_z_n_1 = self.E_z_n.copy() # data for t = n-1
            self.E_z_n = self.E_z.copy()     # data for t = n

            #self.E_t.append(self.E_z[self.source_x, self.source_y])
            self.E_z_t.append(self.E_z.copy())
            self.H_x_t.append(self.H_x.copy())
            self.H_y_t.append(self.H_y.copy())
            if len(self.E_z_t) > 500:
                del self.E_z_t[0]
                del self.H_x_t[0]
                del self.H_y_t[0]
                
            self.E_z_timeseries.append(self.E_z[450, 450])

    def plot_timeseries(self):
        t = np.arange(0, self.dt * len(self.E_z_timeseries), self.dt)
        plt.plot(t, self.E_z_timeseries)
        plt.xlabel("Time")
        plt.ylabel("E")
        plt.grid(True)
        plt.show()
        
    def plot_E(self, i = 70):
        if i >= len(self.E_z_t):
            i = len(self.E_z_t) - 1
            
        plt.figure(figsize = (5, 5))
        plt.pcolormesh(self.X, self.Y, self.E_z_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                plt.gca().add_patch(circle)
        
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
        
    def plot_H(self, i = 70):
        if i >= len(self.H_x_t):
            i = len(self.H_x_t) - 1
            
        plt.figure(figsize = (10, 5))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.X[1:, :], self.Y[1:, :], self.H_x_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                plt.gca().add_patch(circle)
        
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.pcolormesh(self.X[:, 1:], self.Y[:, 1:], self.H_y_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                plt.gca().add_patch(circle)
        
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
        
    def animate_E(self, file_dir = "fdtd_2d_E_animation.gif", N = 500):
        # animate self.E_z_t as a .gif file.
        # N: number of total steps to save as .gif animation.
        E_z_t = self.E_z_t[-N:]

        fig, ax = plt.subplots(figsize = (5, 5))
        cax = ax.pcolormesh(self.X, self.Y, E_z_t[0].T, 
                            vmin = np.min(E_z_t), vmax = np.max(E_z_t), 
                            shading = "auto", cmap = "gray")
        plt.axis("equal")
        plt.grid(True)
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)
        
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                plt.gca().add_patch(circle)

        def animate(i):
            cax.set_array(E_z_t[i].T.flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(E_z_t) - 1)
        anim.save(file_dir, writer = "pillow")
        plt.show()
        
    def animate_H(self, file_dir = "fdtd_2d_H_animation.gif", N = 500):
        # animate self.H_x,y_t as a .gif file.
        # N: number of total steps to save as .gif animation.
        H_x_t = self.H_x_t[-N:]
        H_y_t = self.H_y_t[-N:]
        
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
        cax1 = ax1.pcolormesh(self.X[1:, :], self.Y[1:, :], H_x_t[0].T, 
                              vmin = np.min(H_x_t), vmax = 0.1 * np.max(H_x_t), 
                              shading = "auto", cmap = "gray")
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        ax1.add_patch(circle)
        
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                ax1.add_patch(circle)
        
        ax1.axis("equal")
        ax1.grid(True)
        
        cax2 = ax2.pcolormesh(self.X[:, 1:], self.Y[:, 1:], H_y_t[0].T, 
                              vmin = np.min(H_y_t), vmax = 0.1 * np.max(H_y_t), 
                              shading = "auto", cmap = "gray")
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        ax2.add_patch(circle)
        
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                ax2.add_patch(circle)
        
        ax2.axis("equal")
        ax2.grid(True)

        def animate(i):
            cax1.set_array(H_x_t[i].T.flatten())
            cax2.set_array(H_y_t[i].T.flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(H_x_t) - 1)
        anim.save(file_dir, writer = "pillow")
        plt.show()
        
    def plot_device(self):
        plt.pcolormesh(self.X, self.Y, self.mask.T, shading = "auto", cmap = "binary")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
