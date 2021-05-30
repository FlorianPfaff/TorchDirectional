# @author Florian Pfaff, pfaff@kit.edu
# @date 2021
import torch
from torch.fft import rfft, irfft
from torch import real, imag, cos, sin
from matplotlib.pyplot import plot, show
from WDDistributionTorch import WDDistribution
import numpy as np
from numpy import pi
import scipy.integrate as integrate
import math

"""
Fourier Distribution. The real and complex ones and the option to have it multiplied_by_n or not serve to use the minimum number of operations and thus optimize the graph
"""

class FourierDistribution:
    def __init__(self, transformation='sqrt', c = None, a = None, b = None, n = None, multiplied_by_n = True):
        assert (a==None) == (b==None)
        assert (a==None) != (c==None)
        if not c==None:
            self.c = c # Assumed as result from rfft
            self.a = None
            self.b = None
            self.n = None # It is not clear for complex ones since they may include another coefficient or not (imaginary part of the last coefficient)
        elif a!=None and b!=None:
            self.a = a
            self.b = b
            self.c = None
            self.n = a.size(0) + b.size(0)
        else:
            raise Exception("Need to provide either c or a and b.")

        self.multiplied_by_n = multiplied_by_n
        self.transformation = transformation
        if n != None:
            if (a!=None):
                assert self.n == n
            else:
                self.n = n
        
    def __sub__(self, other):
        # If transformed will not yield minus of pdfs!
        assert self.a!=None and other.a!=None or self.c!=None and other.c!=None
        assert self.transformation==other.transformation
        assert self.n==other.n
        assert self.multiplied_by_n==other.multiplied_by_n
        if self.a!=None:
            aNew = self.a - other.a
            bNew = self.b - other.b
            fdNew = FourierDistribution(a=aNew, b=bNew, transformation=self.transformation,multiplied_by_n=self.multiplied_by_n)
        else:
            cNew = self.c - other.c
            fdNew = FourierDistribution(c=cNew, transformation=self.transformation,multiplied_by_n=self.multiplied_by_n)
        fdNew.n = self.n # The number should not change! We store it if we use a complex one now and set it to None if we falsely believe we know the number (it is not clear for complex ones)
        return fdNew
    def pdf(self, xs):
        assert xs.ndim <= 2
        xs = xs.view(-1,1)
        a,b = self.get_a_b()

        range = torch.arange(1,a.shape[0]).type_as(xs)
        p = a[0]/2+torch.sum(a[1:].view(1,-1) * cos(xs*range)+b.view(1,-1)*sin(xs*range),dim=1)
        if self.multiplied_by_n:
            p = p/self.n
        if self.transformation == 'sqrt':
            p = p**2
        elif self.transformation == 'identity':
            pass
        else:
            raise Exception("Transformation not supported.")
        return p
    def normalize(self):
        if self.a != None and self.b != None:
            if self.multiplied_by_n:
                a = self.a*(1/self.n)
                b = self.b*(1/self.n)
            else:
                a = self.a
                b = self.b
            if self.transformation=='identity':
                a0_non_rooted = a[0]
                a_new = self.a / (a0_non_rooted*pi)
                b_new = self.b / (a0_non_rooted*pi)
            elif self.transformation=='sqrt':
                from_a0 = a[0]**2/2
                from_a1_to_end_and_b = torch.sum(a[1:]**2)+torch.sum(b**2)
                a0_non_rooted = from_a0 + from_a1_to_end_and_b
                a_new = self.a / torch.sqrt(a0_non_rooted*pi)
                b_new = self.b / torch.sqrt(a0_non_rooted*pi)
            fd_normalized = FourierDistribution(a=a_new,b=b_new,transformation=self.transformation,n=self.n,multiplied_by_n=self.multiplied_by_n)
        elif self.c!= None:
            if self.transformation=='identity':
                if self.multiplied_by_n:
                    c0 = torch.real(self.c[0])*(1/self.n)
                else:
                    c0 = torch.real(self.c[0])
                c_new = self.c/(2*pi*c0)
            elif self.transformation=='sqrt':
                if self.multiplied_by_n:
                    c = self.c*(1/self.n)
                else:
                    c = self.c
                
                # Alternative without complex numbers (even **2 yields complex parts)
                # from_c0 = (torch.real(c[0]))**2
                # from_c1_to_end = torch.sum((torch.real(c[1:]))**2) + torch.sum((torch.imag(c[1:]))**2)
                # c_new = self.c/torch.sqrt(from_c0 + 2*from_c1_to_end)*(1/math.sqrt(2*pi))
                c_norm_div_by_sqrt2 = torch.linalg.norm(torch.cat(((1/math.sqrt(2))*c[0].unsqueeze(-1),c[1:]),dim=-1))
                c_new = self.c/c_norm_div_by_sqrt2*(1/(2*math.sqrt(pi)))
                
            fd_normalized = FourierDistribution(c=c_new,transformation=self.transformation,n=self.n,multiplied_by_n=self.multiplied_by_n)
        else:
            raise Exception('Need either a and b or c.')
        return fd_normalized

    def integral(self):
        if self.a != None and self.b != None:
            if self.multiplied_by_n:
                a = self.a*(1/self.n)
                b = self.b*(1/self.n)
            else:
                a = self.a
                b = self.b
            if self.transformation=='identity':
                a0_non_rooted = a[0] 
            elif self.transformation=='sqrt':
                from_a0 = a[0]**2 * 0.5
                from_a1_to_end_and_b = torch.sum(a[1:]**2)+torch.sum(b**2)
                a0_non_rooted = from_a0 + from_a1_to_end_and_b
            else:
                raise Exception('Transformation not supported.')
            integral = a0_non_rooted * pi
        elif self.c!= None:
            if self.transformation=='identity':
                if self.multiplied_by_n:
                    c0 = torch.real(self.c[0])*(1/self.n)
                else:
                    c0 = torch.real(self.c[0])
                integral = 2*pi*c0
            elif self.transformation=='sqrt':
                if self.multiplied_by_n:
                    c = self.c*(1/self.n)
                else:
                    c = self.c
                from_c0 = (torch.real(c[0]))**2
                # Alternative: torch.sum(((self.c[1:])/self.n)**2) - but this yields a complex part and I think should involve more (complicated) operations
                from_c1_to_end = torch.sum((torch.real(c[1:]))**2) + torch.sum((torch.imag(c[1:]))**2)
                
                a0_non_rooted = 2*from_c0 + 4*from_c1_to_end
                integral = a0_non_rooted * pi
            else:
                raise Exception('Transformation not supported.')
        else:
            raise Exception('Need either a and b or c.')
        return integral
    def integral_numerical(self):
        if self.a != None:
            a = self.a
        else:
            a = 2*real(self.c)
        integral = integrate.quad(lambda x: self.pdf(torch.tensor(x).type_as(a).view(1,-1)),0,2*pi)
        return integral
    def plot_grid(self):
        grid_values = irfft(self.get_c(), self.n)
        xs = np.linspace(0,2*pi,grid_values.shape[0],endpoint=False)
        vals = grid_values.detach().squeeze().cpu()
        if self.transformation=='sqrt':
            p = vals**2
        elif self.transformation=='log':
            p = torch.exp(vals)
        elif self.transformation=='identity':
            p = vals
        else:
            raise Exception("Transformation not supported.")

        plot(xs,p,'r+')
        show
    def plot(self, plot_string='-', **kwargs):
        xs = torch.linspace(0,2*pi,100).view([100,1])
        if self.a!= None:
            xs = xs.type_as(self.a)
        else:
            xs = xs.type_as(torch.real(self.c))
        pdf_vals = self.pdf(xs).detach().cpu()
        
        plot(xs.detach().cpu(), pdf_vals, plot_string, **kwargs)
        show

        return torch.max(pdf_vals)
    def get_a_b(self):
        if self.a != None:
            a = self.a
            b = self.b
        elif self.c != None:
            a = 2*real(self.c)
            b = -2*imag(self.c[1:])
        assert self.n == None or (a.size(0)+b.size(0))==self.n # Other case not implemented yet!
        return a,b
    def get_c(self):
        if self.a != None:
            c = torch.cat((self.a[0].unsqueeze(-1),self.a[1:]+1j*self.b))*0.5
        elif self.c != None:
            c = self.c
        return c
    def to_real_fd(self):
        if self.a != None:
            fd = self
        elif self.c != None:
            a,b = self.get_a_b()
            fd = FourierDistribution(transformation=self.transformation,a=a,b=b,n=self.n,multiplied_by_n=self.multiplied_by_n)
        return fd
        
    @staticmethod
    def from_distribution(dist, n, transformation, store_values_multiplied_by_n = True):
        if isinstance(dist,WDDistribution):
            fd = FourierDistribution(torch.conj(dist.trigonometric_moment(n, whole_range=True))/(2*pi),transformation,multiplied_by_n=False)
            if store_values_multiplied_by_n:
                raise Warning('Scaling up for WD (this not recommended).')
                fd.c = fd.c*fd.n
        else:
            xs = torch.linspace(0,2*pi,n+1).type_as(dist.get_params()[0])
            fvals = dist.pdf(xs[0:-1])
            if transformation=='identity':
                pass
            elif transformation=='sqrt':
                fvals = torch.sqrt(fvals)
            else:
                raise Exception("Transformation not supported.")
            c = rfft(fvals)
            if not store_values_multiplied_by_n:
                c = c*(1/n)
                
            fd = FourierDistribution(c=c,transformation=transformation,n=n,multiplied_by_n=store_values_multiplied_by_n)
        return fd