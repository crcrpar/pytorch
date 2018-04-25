r"""
Spectral Normalization from:
    - https://openreview.net/forum?id=B1QRgziT-
    - https://openreview.net/forum?id=ByS1VpgRZ
"""
import torch
from torch.nn.parameter import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm(2) + eps)


class SpectralNorm(object):

    def __init__(self, name='weight', n_power_iterations=1,
                 use_gamma=False, factor=None, eps=1e-12):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.use_gamma = use_gamma
        self.factor = factor
        self.eps = eps

    def compute_weight(self, module):
        # u, v, w are parameter
        W = getattr(module, self.name + "_org")
        u = getattr(module, self.name + "_u")

        weight_mat = W.data.view(W.size(0), -1)
        for _ in range(self.n_power_iterations):
            v = l2normalize(
                torch.mv(torch.t(weight_mat), u), self.eps)
            u = l2normalize(torch.mv(weight_mat, v), self.eps)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        if self.factor is not None:
            sigma = sigma / self.factor
        if self.use_gamma:
            W.data = getattr(module, self.name + '_gamma').data * W.data
        W.data = W.data / sigma
        return W, u

    def remove(self, module):
        del module._buffers[self.name + '_u']
        if self.use_gamma:
            delattr(module, self.name + '_gamma')

    def __call__(self, module, inputs):
        weight, u = self.compute_weight(module)
        setattr(module, self.name, weight)
        setattr(module, self.name + "_u", u)

    @staticmethod
    def apply(module, name, n_power_iterations, use_gamma, factor, eps):
        fn = SpectralNorm(name, n_power_iterations, use_gamma, factor, eps)
        W = getattr(module, name)
        height = W.size(0)

        u = l2normalize(W.data.new(height).normal_(0, 1), fn.eps)
        module.register_parameter(fn.name + "_org", W)
        module.register_buffer(fn.name + "_u", u)

        if use_gamma:
            _, s, _ = torch.svd(W.view(height, -1).data)
            module.register_parameter(
                fn.name + '_gamma', Parameter(W.data.clone().fill_(s[0])))

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1,
                  use_gamma=False, factor=None, eps=1e-12):
    """Apply spectral normalization to given module."""
    SpectralNorm.apply(module=module,
                       name=name,
                       n_power_iterations=n_power_iterations,
                       use_gamma=use_gamma,
                       factor=factor,
                       eps=eps)

    return module


def remove_weight_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}"
                     .format(name, module))
