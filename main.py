import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


class config_array:
    """
    An object representing many instances of a 1D lattice of Ising spins. Each instance has a different value for the
    parameter `h`.
    """
    def __init__(self, spins, J, h, M=None, energy=None):
        """
        :param spins: A 2D array of spins (-1 or 1). The first dimension should equal the length of `h`.
                      The second dimension is the number of spins in the model.
        :param J: Coupling strength (scalar)
        :param h: External field, 1 dimensional array.
        :param M: Magnetization. Passing as a parameter saves computation time.
        :param energy: Energy. Passing as a parameter saves computation time.
        """
        self.spins = spins
        self.N = spins.shape[-1]
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
        return -self.J*np.sum(self.spins * np.roll(self.spins, 1, axis=-1), axis=-1) - self.h*np.sum(self.spins, axis=-1)

    def compute_M(self):
        return np.sum(self.spins, axis=-1)

    def flip_rand_spin(self):
        i0 = rng.integers(0, self.N, len(self.h))
        idx = np.arange(len(self.h))  # helper array for operations below.
        new_spins = np.copy(self.spins)
        new_spins[idx, i0] = -self.spins[idx, i0]
        new_M = self.M + 2*new_spins[idx, i0]
        new_energy = np.copy(self.energy)
        new_energy -= 2*self.J*new_spins[idx, i0]*(new_spins[idx, (i0-1) % self.N] + new_spins[idx, (i0+1) % self.N])
        new_energy -= 2*self.h*new_spins[idx, i0]
        return config_array(new_spins, self.J, self.h, M=new_M, energy=new_energy)


def generate_samples(alpha, J, h, T, size):
    output_M = [alpha.M]
    output_energy = [alpha.energy]
    for i in range(size):
        print(f'i: {i}')
        beta = alpha.flip_rand_spin()
        prob = np.exp((alpha.energy - beta.energy)/T)
        rand = rng.random(size=len(h))
        accept = (beta.energy <= alpha.energy) | (rand < prob)
        new_M = np.where(accept, beta.M, alpha.M)
        new_energy = np.where(accept, beta.energy, alpha.energy)
        accept = np.array([[x]*alpha.N for x in accept])
        new_spins = np.where(accept, beta.spins, alpha.spins)
        alpha = config_array(new_spins, J, h, M=new_M, energy=new_energy)
        output_M.append(alpha.M)
        output_energy.append(alpha.energy)
    return np.array(output_M), np.array(output_energy)


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
    size = 10**4  # number of Monte Carlo steps taken for each pair (T, h)
    idx_eq = int(size * 0.05)  # index after which observable quantities equilibrate
    hspace = J*np.arange(-2, 2, 0.02)
    # Create initial state of 50 spins up and 50 spins down randomly distributed.
    initial_spins = np.full((len(hspace), N), 1)
    for i in range(len(hspace)):
        initial_spins[i, rng.choice(N, size=N//2, replace=False)] = -1
    # T in units of energy
    for T in (J/2, J, 2*J):
        initial = config_array(initial_spins, J, hspace, M=np.zeros(len(hspace)))
        sample_M, sample_energy = generate_samples(initial, J, hspace, T, size)
        avgM = np.mean(sample_M[idx_eq:], axis=0)
        # plt.figure()
        # plt.plot(np.arange(size+1), sample_M[:,100])
        # plt.show()
        # quit()

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


