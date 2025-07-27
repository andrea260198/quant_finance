import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from support.quant_dataclass import ImmutableDataclass


class AmericanOptionLeastSquareMonteCarloPricer(ImmutableDataclass):
    S_0: float = 100
    r: float = 0.02
    sigma: float = 0.20
    K: float = 100
    M: int = 10_000
    T: float = 1
    dt: float = 0.001
    polinomial_fitting_order: int = 4

    @property
    def _timesteps(self) -> int:
        return int(self.T // self.dt)

    @property
    def _SS(self) -> np.array:
        SS = np.zeros((self.M, self._timesteps))
        SS[:, 0] = self.S_0
        for j in range(self._timesteps - 1):
            SS[:, j+1] = SS[:, j] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.M))
        return SS

    def run(self):
        r = self.r
        dt = self.dt
        K = self.K
        SS = self._SS
        timesteps = self._timesteps
        M = self.M
        sigma = self.sigma

        payoff = np.maximum(SS[:, -1] - K, 0)

        VV = np.zeros((M, timesteps))
        VV[:, -1] = payoff

        discount_factor = np.exp(-r * dt)

        for j in tqdm(range(self._timesteps - 2, -1, -1)):

            #filter = SS[:, j] - K >= 0

            filter = [True] * len(SS[:, j])

            beta = np.polyfit(SS[:, j][filter], discount_factor * VV[:, j+1][filter], self.polinomial_fitting_order)

            continuation_value = np.polyval(beta, SS[:, j])

            self.plot(SS, VV, continuation_value, filter, j, discount_factor)

            VV[:, j] = np.where(SS[:, j] - K > continuation_value,
                                SS[:, j] - K,
                                VV[:, j + 1] * discount_factor)


        return VV[:,0].mean()

    @staticmethod
    def plot(SS, VV, continuation_value, filter, j, discount_factor):
        """
        plt.plot(SS[:, j][filter], VV[:, j + 1][filter], 'ko', label='True'.format(j))
        plt.plot(SS[:, j][filter], continuation_value[filter], 'ro', label='Estimated'.format(j))
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()
        """
        plt.plot(SS[:, j], VV[:, j + 1] * discount_factor, 'ko', label='True'.format(j))
        plt.plot(SS[:, j], continuation_value, 'ro', label='Estimated'.format(j))

        #plt.plot(SS[:, j], continuation_value - VV[:, j + 1] * discount_factor, 'go', label='Estimated'.format(j))

        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()



if __name__ == '__main__':
    print(AmericanOptionLeastSquareMonteCarloPricer().run())