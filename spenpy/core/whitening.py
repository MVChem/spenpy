"""Apply the same coil whitening to measurements and forward/adjoint physics."""

import torch
from torch import nn


class CoilWhitenedOperator(nn.Module):
    def __init__(self, operator, covariance):
        super().__init__()
        self.operator = operator
        covariance = torch.as_tensor(
            covariance, device=operator.coil_maps.device, dtype=operator.coil_maps.dtype
        )
        nc = operator.measurement_shape[-1]
        if covariance.shape != (nc, nc) or not torch.allclose(
            covariance, covariance.mH
        ):
            raise ValueError("covariance must be Hermitian [coil,coil]")
        lower = torch.linalg.cholesky(covariance)
        w = torch.linalg.solve_triangular(
            lower, torch.eye(nc, device=lower.device, dtype=lower.dtype), upper=False
        )
        self.register_buffer("whitening_matrix", w)

    @property
    def image_shape(self):
        return self.operator.image_shape

    @property
    def measurement_shape(self):
        return self.operator.measurement_shape

    def whiten(self, measurements):
        return torch.einsum("dc,...mkc->...mkd", self.whitening_matrix, measurements)

    def forward(self, image):
        return self.whiten(self.operator(image))

    def adjoint(self, data):
        return self.operator.adjoint(
            torch.einsum("dc,...mkd->...mkc", self.whitening_matrix.conj(), data)
        )

    def normal(self, image):
        return self.adjoint(self(image))

    def data_gradient(self, image, measurements):
        return self.adjoint(self(image) - measurements)
