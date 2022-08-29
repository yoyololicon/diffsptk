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

from ..misc.utils import check_size
from ..misc.utils import get_gamma
from .linear_intpl import LinearInterpolation
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class PseudoMGLSADigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsadf.html>`_
    for details. The exponential filter is approximated by the Taylor expansion.

    Parameters
    ----------
    filter_order : int >= 0 [scalar]
        Order of filter coefficients, :math:`M`.

    cep_order : int >= filter_order [scalar]
        Order of linear cepstrum.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    gamma : float [-1 <= gamma <= 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 [scalar]
        Number of stages.

    taylor_order : int >= 0 [scalar]
        Order of Taylor series expansion, :math:`L`.

    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    ignore_gain : bool [scalar]
        If True, perform filtering without gain.

    phase : ['minimum', 'zero']
        Filter type.

    """

    def __init__(
        self,
        filter_order,
        cep_order=200,
        alpha=0,
        gamma=0,
        c=None,
        taylor_order=30,
        frame_period=1,
        ignore_gain=False,
        phase="minimum",
    ):
        super(PseudoMGLSADigitalFilter, self).__init__()

        self.filter_order = filter_order
        self.taylor_order = taylor_order
        self.frame_period = frame_period
        self.ignore_gain = ignore_gain
        self.phase = phase

        assert 0 <= self.taylor_order

        if self.phase == "minimum":
            self.pad = nn.ConstantPad1d((cep_order, 0), 0)
        elif self.phase == "zero":
            self.pad = nn.ConstantPad1d((cep_order, cep_order), 0)
        else:
            raise ValueError(f"phase {phase} is not supported")

        self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            filter_order,
            cep_order,
            in_alpha=alpha,
            in_gamma=get_gamma(gamma, c),
        )
        self.linear_intpl = LinearInterpolation(frame_period)

    def forward(self, x, mc):
        """Apply an MGLSA digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Excitation signal.

        mc : Tensor [shape=(..., T/P, M+1)]
            Mel-generalized cepstrum, not MLSA digital filter coefficients.

        Returns
        -------
        y : Tensor [shape=(..., T)]
            Output signal.

        Examples
        --------
        >>> M = 4
        >>> x = diffsptk.step(3)
        >>> mc = torch.randn(2, M + 1)
        >>> mc
        tensor([[-0.9134, -0.5774, -0.4567,  0.7423, -0.5782],
                [ 0.6904,  0.5175,  0.8765,  0.1677,  2.4624]])
        >>> mglsadf = diffsptk.PseudoMGLSADigitalFilter(M, frame_period=2)
        >>> y = mglsadf(x.view(1, -1), mc.view(1, 2, M + 1))
        >>> y
        tensor([[0.4011, 0.8760, 3.5677, 4.8725]])

        """
        check_size(mc.size(-1), self.filter_order + 1, "dimension of mel-cepstrum")
        check_size(x.size(-1), mc.size(-2) * self.frame_period, "sequence length")

        c = self.mgc2c(mc)
        if self.ignore_gain:
            c[..., 0] = 0

        if self.phase == "minimum":
            c = c.flip(-1)
        elif self.phase == "zero":
            c1 = 0.5 * c[..., 1:]
            c = torch.cat((c1.flip(-1), c[..., :1], c1), dim=-1)
        c = self.linear_intpl(c)

        y = x.clone()
        for a in range(1, self.taylor_order + 1):
            x = self.pad(x)
            x = x.unfold(-1, c.size(-1), 1)
            x = (x * c).sum(-1) / a
            y += x
        return y
