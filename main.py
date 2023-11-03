import numpy as np

class config:
    def __init__(self, spins, J, H):
        self.spins = spins
        self.N = len(spins)
        self.J = J
        self.H = H
        self.M = np.sum
        self.energy = -self.J*np.vdot(spins, np.roll(spins, 1)) - self.H
        self.bond_energy =
        self.ext_energy = np.sum(-1*spins)