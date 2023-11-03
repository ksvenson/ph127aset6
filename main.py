import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


class config:
    def __init__(self, spins, J, h, M=None, energy=None):
        self.spins = spins
        self.N = len(spins)
        self.J = J
        self.h = h
        if M is not None:
            self.M = M
        else:
            self.M = np.sum(spins)
        if energy is not None:
            self.energy = energy
        else:
            self.energy = -self.J*np.vdot(spins, np.roll(spins, 1)) - self.h*self.M

    def flip_rand_spin(self):
        i0 = rng.integers(0, self.N)
        new_spins = np.copy(self.spins)
        new_spins[i0] = -self.spins[i0]
        new_M = self.M + 2*new_spins[i0]
        new_energy = self.energy - 2*self.J*self.spins[i0]*(self.spins[i0-1] + self.spins[(i0+1) % self.N])
        new_energy -= 2*self.h*self.spins[i0]
        return config(new_spins, self.J, self.h, new_M, new_energy)


def generate_samples(initial, J, h, T, size):
    output = [initial]
    alpha = config(initial, J, h)

    for i in range(size):
        beta = alpha.flip_rand_spin()
        if beta.energy <= alpha.energy:
            output += beta
        else:
            prob = np.exp((alpha.energy - beta.energy)/T)
            if rng.random() < prob:
                output += beta
            else:
                output += alpha
    return output


if __name__ == '__main__':
    N = 20
    J = 1
    h = J * np.arange(-2, 2, 0.02)
    size = 10**7
    initial = np.full(N, 1)
    initial[rng.choice(N, size=N//2, replace=False)] = -1
    for T in (J/2, J, 2*J):
        samples = generate_samples(initial, J, h, T, size)
        M = [s.M for s in samples]

        plt.figure()
        plt.plot(h, M)
    plt.show()


