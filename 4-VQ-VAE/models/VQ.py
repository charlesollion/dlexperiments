import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        latents = vq(z_e_x, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x)

        return z_q_x, z_q_x_bar


class VQEmbeddingGumbel(nn.Module):
    # Simple Gumbel Softmax to get indices of embedding
    def __init__(self, K, D, tau, normalize=False):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.tau = tau
        self.normalize = False

    def straight_through(self, z_e_x):
        if self.normalize:
            # Cosine similarity instead of dot product
            self.embedding.weight.data = F.normalize(self.embedding.weight, p=2, dim=1)
            z_e_x = F.normalize(z_e_x, p=2, dim=1)
        sims = torch.matmul(z_e_x, self.embedding.weight.T)
        hard = F.gumbel_softmax(sims, tau=self.tau, dim=-1, hard=True)
        z_q_x = torch.matmul(hard, self.embedding.weight)
        return z_q_x, z_q_x

    def forward(self, z_e_x):
        if self.normalize:
            # Cosine similarity instead of dot product
            self.embedding.weight.data = F.normalize(self.embedding.weight, p=2, dim=1)
            z_e_x = F.normalize(z_e_x, p=2, dim=1)
        sims = torch.matmul(z_e_x, self.embedding.weight.T)
        hard = F.gumbel_softmax(sims, tau=self.tau, dim=-1, hard=True)
        return torch.argmax(hard,dim=-1)
