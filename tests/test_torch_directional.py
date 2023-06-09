# @author Florian Pfaff, pfaff@kit.edu
# @date 2021-2023
import unittest
from torch_directional import FourierDistribution
from torch_directional import VonMisesDistribution
import torch
import numpy as np
from numpy import pi
import copy
import scipy.integrate as integrate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestVMDistribution(unittest.TestCase):
    def test_vm_init(self):
        dist1 = VonMisesDistribution(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        dist2 = VonMisesDistribution(torch.tensor(2.0,device=device), torch.tensor(1.0, device=device))
        self.assertEqual(dist1.kappa,dist2.kappa)
        self.assertNotEqual(dist1.mu,dist2.mu)
    def test_pdf(self):
        dist = VonMisesDistribution(torch.tensor(2.0, device=device), torch.tensor(1.0, device=device))
        xs = torch.linspace(1,7,7,device=device)
        np.testing.assert_array_almost_equal(dist.pdf(xs).cpu().numpy(), np.array([ 0.215781465110296, 0.341710488623463, 0.215781465110296, 0.0829150854731715, 0.0467106111086458, 0.0653867888824553, 0.166938593220285]))
class TestFourierDistribution(unittest.TestCase):
    def test_vm_to_fourier(self):
        for mult_by_n in [True, False]:
            for transformation in ['identity','sqrt']:
                xs = torch.linspace(0, 2*pi, 100, device=device)
                dist = VonMisesDistribution(torch.tensor(2.5, device=device),torch.tensor(1.5, device=device))
                fd = FourierDistribution.from_distribution(dist, n=31, transformation=transformation,store_values_multiplied_by_n=mult_by_n)
                np.testing.assert_array_almost_equal(dist.pdf(xs).numpy(),fd.pdf(xs).numpy())
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(dist.pdf(xs).numpy(),fd_real.pdf(xs).numpy())
    def test_integral_numerical(self):
        scale_by = 2/5
        for mult_by_n in [True, False]:
            for transformation in ['identity']:#,'sqrt']:
                dist = VonMisesDistribution(torch.tensor(1.5, device=device),torch.tensor(2.5, device=device))
                fd = FourierDistribution.from_distribution(dist, n=31, transformation=transformation,store_values_multiplied_by_n=mult_by_n)
                np.testing.assert_array_almost_equal(fd.integral_numerical()[0],1)
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(fd_real.integral_numerical()[0],1)
                fd_unnorm = copy.copy(fd)
                fd_unnorm.c = fd.c*(scale_by)
                if transformation == 'identity':
                    expected_val = scale_by
                else:
                    expected_val = (scale_by)**2
                np.testing.assert_array_almost_equal(fd_unnorm.integral_numerical()[0],expected_val)
                fd_unnorm_real = fd_unnorm.to_real_fd()
                np.testing.assert_array_almost_equal(fd_unnorm_real.integral_numerical()[0],expected_val)
    def test_integral(self):
        scale_by = 1/5
        for mult_by_n in [True, False]:
            for transformation in ['identity','sqrt']:
                dist = VonMisesDistribution(torch.tensor(2.5, device=device),torch.tensor(1.5, device=device))
                fd = FourierDistribution.from_distribution(dist, n=31, transformation=transformation,store_values_multiplied_by_n=mult_by_n)
                np.testing.assert_array_almost_equal(fd.integral().cpu(),1)
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(fd_real.integral().cpu(),1)
                fd_unnorm = copy.copy(fd)
                fd_unnorm.c = fd.c*(scale_by)
                if transformation == 'identity':
                    expected_val = scale_by
                else:
                    expected_val = (scale_by)**2
                np.testing.assert_array_almost_equal(fd_unnorm.integral().cpu(),expected_val)
                fd_unnorm_real = fd_unnorm.to_real_fd()
                np.testing.assert_array_almost_equal(fd_unnorm_real.integral().cpu(),expected_val)
    def test_normalize(self):
        scale_by = 0.3
        for mult_by_n in [False, True]:
            for transformation in ['identity','sqrt']:
                dist = VonMisesDistribution(torch.tensor(2.0, device=device),torch.tensor(2.0, device=device))
                fd_unnorm = FourierDistribution.from_distribution(dist, n=31, transformation=transformation,store_values_multiplied_by_n=mult_by_n)
                fd_unnorm.c = fd_unnorm.c*scale_by
                fd_norm = fd_unnorm.normalize()
                fd_unnorm_real = fd_unnorm.to_real_fd()
                fd_norm_real = fd_unnorm_real.normalize()
                np.testing.assert_array_almost_equal(fd_norm.integral(),1)
                np.testing.assert_array_almost_equal(fd_norm_real.integral(),1)
    def test_distance(self):
        dist1 = VonMisesDistribution(torch.tensor(0.0, device=device),torch.tensor(1.0, device=device))
        dist2 = VonMisesDistribution(torch.tensor(2.0, device=device),torch.tensor(1.0, device=device))
        for mult_by_n in [False, True]:
            fd1 = FourierDistribution.from_distribution(dist1,n=31,transformation='sqrt',store_values_multiplied_by_n=mult_by_n)
            fd2 = FourierDistribution.from_distribution(dist2,n=31,transformation='sqrt',store_values_multiplied_by_n=mult_by_n)
            hel_like_distance,_ = integrate.quad(lambda x: (torch.sqrt(dist1.pdf(torch.tensor(x).type_as(dist1.mu).view(1,-1)))-torch.sqrt(dist2.pdf(torch.tensor(x).type_as(dist1.mu).view(1,-1))))**2,0,2*pi)
            fd_diff = fd1-fd2
            np.testing.assert_array_almost_equal(fd_diff.integral().cpu(),hel_like_distance)