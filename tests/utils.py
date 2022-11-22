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

import functools
import subprocess
import time
import warnings

import numpy as np
import soundfile as sf
import torch


def is_array(x):
    return type(x) is list or type(x) is tuple


def compose(*fs):
    def compose2_outer_kwargs(f, g):
        return lambda *args, **kwargs: f(g(*args), **kwargs)

    return functools.reduce(compose2_outer_kwargs, fs)


def call(cmd, get=True):
    if get:
        res = subprocess.run(
            cmd + " | x2x +da -f %.12f",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        is_double = torch.get_default_dtype() == torch.float64
        data = np.fromstring(
            res.stdout, sep="\n", dtype=np.float64 if is_double else np.float32
        )
        assert len(data) > 0, f"Failed to run command {cmd}"
        return data
    else:
        res = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
        )
        return None


def check_compatibility(
    device,
    modules,
    setup,
    inputs,
    target,
    teardown,
    dx=None,
    dy=None,
    eq=None,
    opt={},
    sr=None,
    verbose=False,
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    for cmd in setup:
        call(cmd, get=False)

    if not is_array(modules):
        modules = [modules]
    if not is_array(inputs):
        inputs = [inputs]

    x = []
    for i, cmd in enumerate(inputs):
        x.append(torch.from_numpy(call(cmd)).to(device))
        if is_array(dx):
            if dx[i] is not None:
                x[-1] = x[-1].reshape(-1, dx[i])
        elif dx is not None:
            x[-1] = x[-1].reshape(-1, dx)
        else:
            pass

    if len(setup) == 0:
        y = call(f"{inputs[0]} | {target}")
    else:
        y = call(target)
    if dy is not None:
        y = y.reshape(-1, dy)

    for cmd in teardown:
        call(cmd, get=False)

    module = compose(*[m.to(device) if hasattr(m, "to") else m for m in modules])
    y_hat = module(*x, **opt).cpu().numpy()

    if sr is not None:
        sf.write("output.wav", y_hat / 32768, sr)
        sf.write("target.wav", y / 32768, sr)

    if verbose:
        print(f"Output: {y_hat}")
        print(f"Target: {y}")

    if eq is None:
        assert np.allclose(y_hat, y), f"Output: {y_hat}\nTarget: {y}"
    else:
        assert eq(y_hat, y), f"Output: {y_hat}\nTarget: {y}"


def check_differentiable(device, modules, shapes, opt={}, load=1):
    if device == "cuda" and not torch.cuda.is_available():
        return

    if not is_array(modules):
        modules = [modules]
    if not is_array(shapes[0]):
        shapes = [shapes]

    x = []
    for shape in shapes:
        x.append(torch.randn(*shape, requires_grad=True, device=device))

    module = compose(*[m.to(device) if hasattr(m, "to") else m for m in modules])
    optimizer = torch.optim.SGD(x, lr=0.01)

    s = time.process_time()
    for _ in range(load):
        y = module(*x, **opt)
        optimizer.zero_grad()
        loss = y.mean()
        loss.backward()
        optimizer.step()
    e = time.process_time()

    if load > 1:
        print(f"time: {e - s}")

    for i in range(len(x)):
        g = x[i].grad.cpu().numpy()
        if not np.any(g):
            warnings.warn(f"detect zero gradient at {i}-th input")
        if np.any(np.isnan(g)):
            warnings.warn(f"detect NaN-gradient at {i}-th input")
        if np.any(np.isinf(g)):
            warnings.warn(f"detect Inf-gradient at {i}-th input")


def check_various_shape(module, shapes):
    x = torch.randn(*shapes[0])
    for i, shape in enumerate(shapes):
        x = x.view(shape)
        y = module(x).view(-1)
        if i == 0:
            target = y
        else:
            assert torch.allclose(y, target)
