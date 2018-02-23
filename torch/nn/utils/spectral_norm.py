"""
Spectral Normalization from Spectral Normalization for Generative Adversarial Networks from https://arxiv.org/abs/1802.05957
"""
import torch

from torch.autograd.variable import Variable
from torch.nn.parameter import Parameter


def _init_u_v_gamma(weight, use_gamma):
    w_mat = weight.view(weight.size(0), -1)
    u = _normalize(w_mat.data.new(w_mat.size(0)).normal_(0, 1))
    v = _normalize(w_mat.data.new(w_mat.size(1)).normal_(0, 1))

    if use_gamma:
        _, s, _ = torch.svd(w_mat.data)
        gamma = s.new(1).fill_(s[0])
        for _ in range(weight.data.dim()):
            gamma.unsqueeze(-1)
    else:
        gamma = None

    return u, v, gamma


def _normalize(v, eps=1e-12):
    return v / (v.norm(2) + eps)


def _variable(tensor):
    return Variable(tensor)


class SpectralNorm(object):

    eps = 1e-12

    def __init__(self, name='weight', Ip=1, use_gamma=False, factor=None):
        self.name = name
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor

    def compute_weight(self, module):
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        w = getattr(module, self.name)
        gamma = getattr(module, self.name + '_gamma')

        print(type(u), type(v), type(w), type(gamma))
        w_mat_height = w.size(0)
        w_mat = w.data.view(w_mat_height, -1)
        for _ in range(self.Ip):
            v = _normalize(torch.mv(w_mat.t(), u), SpectralNorm.eps)
            u = _normalize(torch.mv(w_mat, v), SpectralNorm.eps)

        getattr(module, self.name + '_u').copy_(u)
        getattr(module, self.name + '_v').copy_(v)
        sigma = torch.dot(u, torch.mv(w_mat, v))
        print(type(sigma))
        if self.factor is not None:
            sigma = sigma / self.factor
        if self.use_gamma:
            gamma = gamma.expand_as(w)
            w = w * gamma
        w = w / sigma
        print(type(w))
        return w

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_gamma']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, Ip, use_gamma, factor):
        fn = SpectralNorm(name, Ip, use_gamma, factor)
        weight = getattr(module, name)
        # add _u, _v and gamma
        u, v, gamma = _init_u_v_gamma(weight, use_gamma)
        module.register_buffer(name + '_u', u)
        module.register_buffer(name + '_v', v)
        module.register_parameter(name + '_gamma', gamma)
        module.register_parameter(name, fn.compute_weight(module))

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', Ip=1, use_gamma=False, factor=None):
    r"""Applies spectral normalization to a weight parameter in the given module.

    .. math::
        \mathbf{w} = \dfrac{\mathbf{w}}{\sigma(\mathbf{w})}
        \text{where, } \sigma(\mathbf{w}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \|A\mathbf{h}\|_2

    Spectral Normalization controls the Lipschitz constant of the discriminator function f by literally
    constraining the spectral norm of each layer using spectrl norm of weight.
    Spectral norms are computed by "power iteration method".

    See https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        Ip (int, optional): the number of power iteration
        use_gamma (float, optional): scale parameter applied to spectrally normalized weight
        factor (float, optional): scale parameter applied to spectral norm

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Conv2d(3, 16, 3, 1, 1))
        Conv2d (3 -> 16)
        >>> m.weight_u.size()
        torch.Size([])
        >>> m.weight_v.size()
        torch.Size([])
    """
    SpectralNorm.apply(module=module,
                       name=name,
                       Ip=Ip,
                       use_gamma=use_gamma,
                       factor=factor)

    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Returns:
        The original module

    Example:
        >>> m = spectral_norm(nn.Conv2d(3, 16, 1, 1))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}"
                     .format(name, module))
