import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm

# fdtd = fdtd2d_tez_laser()
# fdtd.run()
# fdtd.plot()

# FDTD 2D, TEZ polarization laser simulation.
# The magnetic field oscillates only in the z direction, while the electric field
# oscillates only in the x and y directions. A staggered grid in space is used to 
# simulate H_z, E_x and E_y:
#     E_y(m, n+1/2) H_z(m+1/2, n+1/2) 
#                   E_x(m+1/2, n) 

class fdtd2d_tez_laser:
    def __init__(self, Nx = 501, Ny = 501, c = 1, dx = 1, dy = 1, drill_holes = False):
        # Grid attributes
        self.Nx = Nx
        self.Ny = Ny
        self.c = c
        self.dx = dx
        self.dy = dy
        self.dt = min(dx, dy) / np.sqrt(2) / c

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
        
        # Magnetic and electric fields in TEZ polarization
        self.H_z = np.zeros([Nx - 1, Ny - 1])
        self.E_x = np.zeros([Nx - 1, Ny])
        self.E_y = np.zeros([Nx, Ny - 1])

        # Dielectric slab attributes
        self.radius = 200
        self.mask_x = np.zeros([Nx - 1, Ny])
        for i in range(Nx - 1):
            for j in range(Ny):
                if np.sqrt((i - self.source_x)**2 + (j - self.source_y)**2) < self.radius:
                    self.mask_x[i, j] = 1
        self.mask_y = np.zeros([Nx, Ny - 1])
        for i in range(Nx):
            for j in range(Ny - 1):
                if np.sqrt((i - self.source_x)**2 + (j - self.source_y)**2) < self.radius:
                    self.mask_y[i, j] = 1     

        self.drill_holes = drill_holes
        if self.drill_holes == True:
            self.sub_radius = 30
            self.centres = [[150, 170], 
                            [165, 300],
                            [300, 120],
                            [225, 220],
                            [360, 330],
                            [320, 220],
                            [230, 360]]
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                for i in range(-c_x, c_x + self.sub_radius):
                    for j in range(-c_y, c_y + self.sub_radius):
                        if np.sqrt((i - c_x)**2 + (j - c_y)**2) < self.sub_radius:
                            self.mask_x[i, j] = 0
                            self.mask_y[i, j] = 0

        # Permittivity and permeability
        self.n1 = 1.0
        self.n2 = 3.0
        self.E_x = np.logical_not(self.mask_x) * self.n1**2 + self.mask_x * self.n2**2 
        self.E_y = np.logical_not(self.mask_y) * self.n1**2 + self.mask_y * self.n2**2 
        # we assume that the dielectric is a perfect magnetic material with
        # mu = 1 everywhere. So we do not need to explicitly have a vector 
        # for mu. If this condition is not satisfied, then mu must be taken
        # into account as well in the simulation.
        
        # Polarization field P
        self.P_x = np.zeros([Nx - 1, Ny])
        self.Place_x = np.zeros([Nx - 1, Ny]) # polarization one time step before
        self.Pold_x = np.zeros([Nx - 1, Ny]) # polarization two time steps before
        self.P_y = np.zeros([Nx, Ny - 1])
        self.Place_y = np.zeros([Nx, Ny - 1]) # polarization one time step before
        self.Pold_y = np.zeros([Nx, Ny - 1]) # polarization two time steps before
        
        # Population inversion D
        self.D0 = 10.0 # pump strength
        self.D_x = self.mask_x * self.D0
        self.D_y = self.mask_y * self.D0
        
        # Time dependent field for plotting and animation
        self.H_t = []
        self.H_timeseries = []

        # Fields at time n, n-1 for Mur ABC
        self.H_z_n = self.H_z.copy()
        self.H_z_n_1 = self.H_z_n.copy()
        self.E_x_n = self.E_x.copy()
        self.E_x_n_1 = self.E_x_n.copy()
        self.E_y_n = self.E_y.copy()
        self.E_y_n_1 = self.E_y_n.copy()
        
    def run(self, n_iter = 10000):
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
        for n in tqdm.trange(n_iter):
            # Update magnetic field at time step n+1/2
            diff_E_x = self.dt / self.dy * (self.E_x[:, 1:] - self.E_x[:, :-1])
            diff_E_y = self.dt / self.dx * (self.E_y[1:, :] - self.E_y[:-1, :])
            self.H_z = self.H_z - (diff_E_y - diff_E_x)
            
            # Update polarization fields at time step n+1/2
            self.P_x = self.mask_x / c1*(c2*self.P_x - c3*self.Pold_x - c4*self.E_x*self.D_x)
            self.Pold_x = self.Place_x.copy() 
            self.Place_x = self.P_x.copy() # carry the current value of P for two time steps
            self.P_y = self.mask_y / c1*(c2*self.P_y - c3*self.Pold_y - c4*self.E_y*self.D_y)
            self.Pold_y = self.Place_y.copy() 
            self.Place_y = self.P_y.copy() # carry the current value of P for two time steps
            
            # Update electric fields at time step n+1
            diff_P_x = -4 * np.pi / self.E_x[:, 1:-1] * (self.P_x[:, 1:-1] - self.Pold_x[:, 1:-1])
            diff_P_y = -4 * np.pi / self.E_y[1:-1, :] * (self.P_y[1:-1, :] - self.Pold_y[1:-1, :])
            
            self.E_x[:, 1:-1] = self.E_x[:, 1:-1] + diff_P_x + self.dt / self.dy * (self.H_z[:, 1:] - self.H_z[:, :-1])
            self.E_y[1:-1, :] = self.E_y[1:-1, :] + diff_P_y - self.dt / self.dx * (self.H_z[1:, :] - self.H_z[:-1, :])
            
            # Update population inversion D
            self.D_x = self.mask_x / c5*(c6*self.D_x + self.gpara*self.D0 + c7*self.E_x*(c8*self.P_x + c9*self.Pold_x))
            self.D_y = self.mask_y / c5*(c6*self.D_y + self.gpara*self.D0 + c7*self.E_y*(c8*self.P_y + c9*self.Pold_y))
            
            # Pulse at time step 
            tp = 30
            if n * self.dt <= tp:
                C = (7 / 3) ** 3 * (7 / 4) ** 4
                pulse = C * (n * self.dt / tp) ** 3 * (1 - n * self.dt / tp) ** 4
            else:
                pulse = 0
                
            self.H_z[self.source_x, self.source_y] = self.H_z[self.source_x, self.source_y] + pulse
 
            # Mur ABC for left boundary
            self.E_y[0, :] = c_0 * (self.E_y[2, :] + self.E_y_n_1[0, :]) +    \
                             c_1 * (self.E_y_n[0, :] + self.E_y_n[2, :] -    \
                                    self.E_y[1, :] - self.E_y_n_1[1, :]) +    \
                             c_2 * self.E_y_n[1, :] - self.E_y_n_1[2, :]

            # Mur ABC for right boundary
            self.E_y[-1, :] = c_0 * (self.E_y[-3, :] + self.E_y_n_1[-1, :]) +    \
                              c_1 * (self.E_y_n[-1, :] + self.E_y_n[-3, :] -    \
                                     self.E_y[-2, :] - self.E_y_n_1[-2, :]) +    \
                              c_2 * self.E_y_n[-2, :] - self.E_y_n_1[-3, :]

            # Mur ABC for bottom boundary
            self.E_x[:, 0] = c_0 * (self.E_x[:, 2] + self.E_x_n_1[:, 0]) +    \
                             c_1 * (self.E_x_n[:, 0] + self.E_x_n[:, 2] -    \
                                    self.E_x[:, 1] - self.E_x_n_1[:, 1]) +    \
                             c_2 * self.E_x_n[:, 1] - self.E_x_n_1[:, 2]

            # Mur ABC for right boundary
            self.E_x[:, -1] = c_0 * (self.E_x[:, -3] + self.E_x_n_1[:, -1]) +    \
                              c_1 * (self.E_x_n[:, -1] + self.E_x_n[:, -3] -    \
                                     self.E_x[:, -2] - self.E_x_n_1[:, -2]) +    \
                              c_2 * self.E_x_n[:, -2] - self.E_x_n_1[:, -3]

            # Store magnetic and electric fields for ABC at time step n
            self.E_x_n_1 = self.E_x_n.copy() # data for t = n-1
            self.E_x_n = self.E_x.copy()     # data for t = n

            self.E_y_n_1 = self.E_y_n.copy() # data for t = n-1
            self.E_y_n = self.E_y.copy()     # data for t = n

            self.H_z_n_1 = self.H_z_n.copy() # data for t = n-1
            self.H_z_n = self.H_z.copy()     # data for t = n

            self.H_t.append(self.H_z.copy())
            if len(self.H_t) > 500:
                del self.H_t[0]
            
    def plot(self, i = -1):
        plt.figure(figsize = (5, 5))
        #plt.pcolormesh(self.x, self.y, self.E_z, shading = "auto", cmap = "gray")
        plt.pcolormesh(self.x, self.y, self.H_t[i].T, 
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
        plt.axis("equal")
        #plt.colorbar()
        plt.show()
            
    def animate(self, file_dir = "fdtd_2d_animation.gif", N = 500):
        # animate self.H_t as a .gif file.
        # N: number of total steps to save as .gif animation.
        H_t = self.H_t[-N:]

        fig, ax = plt.subplots(figsize = (5, 5))
        cax = ax.pcolormesh(self.x, self.y, H_t[0].T, 
                            vmin = np.min(H_t), vmax = np.max(H_t), 
                            shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)
        
        if self.drill_holes == True:
            for c in range(len(self.centres)):
                c_x = self.centres[c][0]
                c_y = self.centres[c][1]
                circle = plt.Circle((c_x, c_y), self.sub_radius, color = "k", fill = False)
                plt.gca().add_patch(circle)

        def animate(i):
            cax.set_array(H_t[i].T.flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(E_t) - 1)

        #plt.show()

        anim.save(file_dir, writer = "pillow")