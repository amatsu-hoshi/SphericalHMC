import torch
from torch import nn
from torch.autograd import grad


class SphericalHMC(nn.Module):
    def __init__(self, logpdf, dim, L, eps):
        super(SphericalHMC, self).__init__()
        self.logpdf = logpdf
        self.dim = dim
        self.L = L
        self.eps = eps
        self.I_a = nn.parameter.Parameter(
            torch.cat((torch.eye(self.dim), torch.zeros(1, self.dim)), dim=0), requires_grad=False)

    @staticmethod
    def batch_square_norm(input):
        return torch.sum(torch.pow(input, 2), -1)

    @staticmethod
    def batch_augment(theta):
        return torch.cat(
            (theta, torch.unsqueeze(torch.sqrt(1. - SphericalHMC.batch_square_norm(theta)), -1)), -1)

    @staticmethod
    def batch_proj(theta_a):
        return theta_a[..., :-1].detach().clone()

    @staticmethod
    def batch_outer(input1, input2):
        return torch.einsum('bi,bj->bij', input1, input2)

    @staticmethod
    def batch_mvp(input1, input2):
        return torch.einsum('bij,bj->bi', input1, input2)

    def H(self, theta_a, v):
        return -self.logpdf(SphericalHMC.batch_proj(theta_a)) + self.batch_square_norm(v) / 2.

    def sample(self, theta_0):
        theta_a = SphericalHMC.batch_augment(theta_0)
        v = torch.randn_like(theta_a)
        v -= SphericalHMC.batch_mvp(
            SphericalHMC.batch_outer(theta_a, theta_a), v)
        h_0 = self.H(theta_a, v)
        for step in range(self.L):
            theta = SphericalHMC.batch_proj(theta_a)
            theta.requires_grad = True
            u = -self.logpdf(theta)
            g = grad(u, theta, torch.ones_like(u))[0]
            theta.requires_grad = False
            v -= self.eps / 2. * SphericalHMC.batch_mvp(
                self.I_a - SphericalHMC.batch_outer(theta_a, theta), g)
            v_norm = torch.unsqueeze(torch.norm(v, dim=-1), -1)
            theta_a_new = theta_a * torch.cos(v_norm * self.eps) + \
                v / v_norm * torch.sin(v_norm * self.eps)
            v = -theta_a * v_norm * torch.sin(v_norm * self.eps) + \
                v * torch.cos(v_norm * self.eps)
            theta_a = theta_a_new
            theta = SphericalHMC.batch_proj(theta_a)
            theta.requires_grad = True
            u = -self.logpdf(theta)
            g = grad(u, theta, torch.ones_like(u))[0]
            theta.requires_grad = False
            v -= self.eps / 2. * SphericalHMC.batch_mvp(
                self.I_a - SphericalHMC.batch_outer(theta_a, theta), g)
        h = self.H(theta_a, v)
        delta_H = h - h_0
        mask = torch.unsqueeze(torch.rand_like(delta_H) < torch.exp(-delta_H), -1)
        return theta * mask + theta_0 * ~mask
