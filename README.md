# FDTD 
This is a set of Python codes used to perform 1D and 2D finite difference time domain simulations.

<img src="https://github.com/natsunoyuki/FDTD/blob/main/images/tmz_2d.png?raw=True" alt="drawing" width=300/>

# Installation
```
pip install git+https://github.com/natsunoyuki/FDTD
```

# Usage
```
import fdtd

fdtd2d_tmz = fdtd.fdtd2d_tmz()
fdtd2d_tmz.run(n_iter = 50000, initiate_pulse = True)
fdtd2d_tmz.plot_E()
```
## Available FDTD Simulators
### One Dimensional FDTD
* fdtd1d: 1D FDTD.
* fdtd1d_laser: 1D laser FDTD.
### Two Dimensional FDTD
* fdtd2d_tez: Transverse electric field 2D FDTD.
* fdtd2d_tez_laser: Transverse electric field laser 2D FDTD.
* fdtd2d_tmz: Transverse magnetic field 2D FDTD.
* fdtd2d_tmz_laser: Transverse magnetic field laser 2D FDTD.
