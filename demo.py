import torch
from torch.distributions import MultivariateNormal
from shmc import SphericalHMC
from matplotlib import pyplot


def uniform_sample(nsamples):
    r_scale = torch.unsqueeze(torch.rand(
        (nsamples), device='cuda'), -1) ** (1. / 2)
    proj = torch.randn((nsamples, 2),
                       device='cuda')
    proj = proj / torch.unsqueeze(torch.norm(proj, dim=-1), -1)
    return proj * r_scale

pyplot.gca().set_aspect(1.)
# generating truncated normal distribution inside a circle from uniform distribution
target_dist = MultivariateNormal(torch.zeros(
    2, device='cuda'), torch.eye(2, device='cuda'))
sampler = SphericalHMC(lambda q:target_dist.log_prob(q * 10), 2, 20, 1e-2).cuda()
q_0 = uniform_sample(10000)
q = sampler.sample(q_0).cpu().numpy()
x = q * 10
pyplot.scatter(x[..., 0], x[..., 1], s=1.)
pyplot.title('Trucated Normal Distribution generated by SHMC')
pyplot.show()