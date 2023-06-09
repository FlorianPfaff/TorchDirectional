# @author Florian Pfaff, pfaff@kit.edu
# @date 2021
from math import pi
import scipy.special
import torch

class VonMisesDistribution:
    def __init__(self, mu, kappa, norm_const = None):
        assert kappa >= 0
        self.mu = mu
        self.kappa = kappa
        self.norm_const = norm_const
    def calculate_norm_const(self):
        # Need to go to CPU to use scipy.special.iv
        self.norm_const = (2 * pi * scipy.special.iv(0, self.kappa.cpu())).type_as(self.kappa)
        return self.norm_const
    def get_params(self):
        return self.mu,self.kappa
    def pdf(self, xs):
        if self.norm_const == None:
            self.calculate_norm_const()
        p = torch.exp(self.kappa * torch.cos(xs - self.mu)) / self.norm_const
        return p