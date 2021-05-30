# @author Florian Pfaff, pfaff@kit.edu
# @date 2021
import torch
from matplotlib.pyplot import plot
from math import sqrt

class WDDistribution:
    def __init__(self, d, w = None):
        self.d = d
        self.w = w
    def pdf(self, xs):
        raise Exception("Pdf is not defined for WDDistribution.")
    def plot(self):
        plot(self.d,self.w,'r+')
    def plot_interpolated(self, plot_string = '-'):
        raise Exception("No interpolation available for WDDistribution.")
    def trigonometric_moment(self, n, whole_range = False, transformation = 'identity'):
        if not whole_range:
            exponent_part=torch.exp(1j*n*self.d)
        else:
            ind_range = torch.arange(n+1).type_as(self.d)
            exponent_part = torch.exp(1j*(self.d.view(1,-1)*ind_range.view(-1,1)))
            
        if self.w == None:
            if transformation == 'identity':
                m = exponent_part.mean(-1)
            elif transformation == 'sqrt':
                m = torch.sum(exponent_part,dim=1)*sqrt(1/self.d.shape[0])
            elif transformation == 'log':
                m = torch.sum(exponent_part,dim=1).mean(-1)
            else:
                raise Exception("Transformation not supported")
        else:
            m = exponent_part@self.w.type_as(exponent_part)
        return m
