# ------------------------------------------------------------------------ #
# Copyright 2022 SPTK Working Group                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from ..misc.utils import check_size, to_3d


def prod2pol(p1, p2):
    if p1.shape[-1] > p2.shape[-1]:
        p1, p2 = p2, p1
    *orig_shape, p1_order = p1.shape
    p1_flip = to_3d(p1).flip(2)
    p2 = to_3d(p2).transpose(0, 1)

    prod = F.conv1d(
        p2,
        p1_flip,
        padding=p1_order - 1,
        groups=p2.shape[0],
    ).reshape(*orig_shape, -1)
    return prod


class RootsToPolynomial(nn.Module):
    """This is the opposite module to :func:`~diffsptk.PolynomialToRoots`.

    order : int >= 1 [scalar]
        Order of coefficients.

    """

    def __init__(self, order):
        super(RootsToPolynomial, self).__init__()

        self.order = order

        assert 1 <= self.order

    def forward(self, x):
        """Convert roots to polynomial coefficients.

        Parameters
        ----------
        x : Tensor [shape=(..., M)]
            Complex roots.

        Returns
        -------
        a : Tensor [shape=(..., M+1)]
            Polynomial coefficients.

        Examples
        --------
        >>> x = torch.tensor([3, 4, -1])
        >>> pol_root = diffsptk.RootsToPolynomial(x.size(-1))
        >>> a = pol_root(x)
        >>> a
        tensor([ 1, -6,  5, 12])

        """
        check_size(x.size(-1), self.order, "number of roots")

        a = torch.stack([torch.ones_like(x), -x], dim=-1)

        left_pols = []
        while a.shape[-2] > 1:
            d, m = divmod(a.shape[-2], 2)
            if m:
                left_pols.append(a[..., -1, :])
                a = a[..., :-1, :]
            a = prod2pol(a[..., :d, :], a[..., d:, :])

        a = reduce(prod2pol, left_pols[::-1], a.squeeze(-2))
        return a
