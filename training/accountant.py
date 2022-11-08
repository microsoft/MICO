from typing import List, Optional

from prv_accountant.dpsgd import DPSGDAccountant

from opacus.accountants.accountant import IAccountant

class PRVAccountant(IAccountant):
    def __init__(self, noise_multiplier, sample_rate, max_steps, eps_error = 0.1, delta_error = 1e-9):
        super().__init__()
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.max_steps = max_steps
        self.eps_error = eps_error
        self.delta_error = delta_error
        self.accountant = DPSGDAccountant(
            noise_multiplier=noise_multiplier, 
            sampling_probability=sample_rate, 
            max_steps=max_steps,
            eps_error=eps_error, 
            delta_error=delta_error)

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if not (noise_multiplier == self.noise_multiplier and sample_rate == self.sample_rate):
            raise ValueError("Noise multplier and sample rate must be constant for DPSGDAccountant")

        if len(self.history) > 0:
            _, _, num_steps = self.history.pop()
            self.history.append((noise_multiplier, sample_rate, num_steps + 1))
        else:
            self.history.append((noise_multiplier, sample_rate, 1))
    
    def get_epsilon(self, delta: float, *, eps_error: float = 0.1, delta_error: float = 1e-9) -> float:
        """
        Compute upper bound for epsilon
        :param float delta: Target delta
        :return: Returns upper bound for $\varepsilon$
        :rtype: float
        """
        if not (eps_error == self.eps_error and delta_error == self.delta_error):
            raise ValueError("Attempted to override eps_error and delta_error which are fixed at initialization")

        if len(self.history) == 0:
            return 0

        _, _, num_steps = self.history[-1]
        _, _, eps = self.accountant.compute_epsilon(delta=delta, num_steps=num_steps)
        return eps

    @classmethod
    def mechanism(cls) -> str:
        return "PRV"

    def __len__(self):
        return len(self.history)