import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


class config:
    """
    An object representing a 1D lattice of Ising spins
    """
    def __init__(self, spins, J, h, M=None, energy=None):
        self.spins = spins
        self.N = len(spins)
        self.J = J
        self.h = h
        if M is not None:
            self.M = M
        else:
            self.M = self.compute_M()
        if energy is not None:
            self.energy = energy
        else:
            self.energy = self.compute_energy()

    def compute_energy(self):
        return -self.J*np.vdot(self.spins, np.roll(self.spins, 1)) - self.h*self.M

    def compute_M(self):
        return np.sum(self.spins)

    def flip_rand_spin(self):
        i0 = rng.integers(0, self.N)
        new_spins = np.copy(self.spins)
        new_spins[i0] = -self.spins[i0]
        new_M = self.M + 2*new_spins[i0]
        new_energy = self.energy - 2*self.J*new_spins[i0]*(new_spins[i0-1] + new_spins[(i0+1) % self.N])
        new_energy -= 2*self.h*new_spins[i0]
        return config(new_spins, self.J, self.h, new_M, new_energy)


def generate_samples(alpha, J, h, T, size):
    output = [alpha]
    for i in range(size):
        beta = alpha.flip_rand_spin()
        if beta.energy <= alpha.energy:
            output += [beta]
            alpha = beta
        else:
            prob = np.exp((alpha.energy - beta.energy)/T)
            if rng.random() < prob:
                output += [beta]
                alpha = beta
            else:
                output += [alpha]
    return output


def exact_limit_m(J, h, T):
    """
    Returns exact magnetization per spin in the thermodynamic limit N-->infinity
    """
    beta = 1/T
    sinh = np.sinh(beta*h)
    return sinh/np.sqrt(sinh**2 + np.exp(-4*beta*J))


def exact_m(J, h, T, N):
    """
    Returns exact magnetization per spin where there are `N` spins.
    """
    beta = 1/T
    cosh = np.cosh(beta*h)
    sinh = np.sinh(beta*h)
    expBJ = np.exp(beta*J)
    sqrt = np.sqrt(sinh**2 + expBJ**(-4))
    lm = expBJ*(cosh-sqrt)
    lp = expBJ*(cosh+sqrt)
    Z = lm**N + lp**N

    der = cosh/sqrt  # helper variable, short for "derivative"
    m = expBJ*sinh*(lm**(N-1)*(1-der) + lp**(N-1)*(1+der))/Z
    return m


if __name__ == '__main__':
    N = 100  # number of spins
    J = 1  # coupling constant
    size = 10**6  # number of Monte Carlo steps taken for each pair (T, h)
    idx_eq = int(size * 0.05)  # index after which observable quantities equilibrate

    # Create initial state of 50 spins up and 50 spins down randomly distributed.
    initial_spins = np.full(N, 1)
    initial_spins[rng.choice(N, size=N//2, replace=False)] = -1

    # T in units of energy
    for T in (J/2, J, 2*J):
        avgM = []
        avgeng = []
        hspace = J*np.arange(-2, 2, 0.02)  # applied magnetic field strength
        for h in hspace:
            print(h)
            initial = config(initial_spins, J, h, M=0)
            samples = generate_samples(initial, J, h, T, size)
            # Truncate early samples
            samples = samples[idx_eq:]
            avgM.append(np.mean([s.M for s in samples]))
            avgeng.append(np.mean([e.energy for e in samples]))
        plt.figure()
        avgM = np.array(avgM)
        exact = N*exact_m(J, hspace, T, N)
        plt.plot(hspace, avgM, label='Monte Carlo')
        plt.plot(hspace, exact, label=rf'Exact Solution for $N={N}$')
        plt.legend(bbox_to_anchor=(0.5,-0.35), loc='lower center')
        plt.title(rf'Magnetization at $T = {T/J} \cdot J/k_B$')
        plt.xlabel(rf'$h/J$')
        plt.ylabel('Average Magnetization')
        plt.savefig(f'temp_{T}.png', bbox_inches='tight')

        plt.figure()
        plt.plot(hspace, avgM-exact)
        plt.title(rf'Error at $T = {T/J} \cdot J/k_B$')
        plt.xlabel(rf'$h/J$')
        plt.ylabel('Error')
        plt.savefig(f'temp_{T}_error.png', bbox_inches='tight')
        plt.yscale('log')
        plt.savefig(f'temp_{T}_log_error.png', bbox_inches='tight')
    plt.show()


