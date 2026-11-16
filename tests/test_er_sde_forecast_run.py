from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from comfyui_spectrum_h3 import sampling
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.er_sde_stochastic import ERSDEStepDescriptor, ERSDEStochasticTracker, _DenoisedAnchor
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from test_max_forecast_run import _run_euler, _streaks


def tracker(gap):
    return ERSDEStochasticTracker(noise_sampler=lambda s, t: torch.ones(1),
        noise_scaler=lambda x: x*x, effective_s_noise=1, max_stage=3,
        debug=False, run_id=7, max_dense_gap=gap)


def desc(step, mode='forecast'):
    return ERSDEStepDescriptor(7, step, mode, False if mode == 'replay' else None, mode != 'actual')


def test_native_gate(monkeypatch):
    native = ModuleType('comfy.k_diffusion.sampling')
    def sample_er_sde(): pass
    def sample_euler(): pass
    native.sample_er_sde, native.sample_euler = sample_er_sde, sample_euler
    comfy = ModuleType('comfy'); kd = ModuleType('comfy.k_diffusion')
    comfy.k_diffusion = kd; kd.sampling = native
    for name, module in [('comfy', comfy), ('comfy.k_diffusion', kd), ('comfy.k_diffusion.sampling', native)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sampling, '_er_sde_tracking_contract', lambda s: (True, None))
    c = SpectrumH3Config(max_forecast_run=3)
    make = lambda f, o={}: SimpleNamespace(sampler_function=f, extra_options=o)
    assert sampling.max_consecutive_forecasts(make(sample_er_sde), c) == 2
    assert sampling.max_consecutive_forecasts(make(sample_euler), c) == 3
    for value in [0, -1, float('nan'), 'bad', None, True]:
        assert sampling.max_consecutive_forecasts(make(sample_er_sde, {'s_noise': value}), c) == 1
    def other(): pass
    other.__name__ = 'sample_er_sde'
    assert sampling.max_consecutive_forecasts(make(other), c) == 1
    other.__name__ = 'sample_refdelta_er_sde'
    assert sampling.max_consecutive_forecasts(make(other), c) == 1
    monkeypatch.setattr(sampling, '_er_sde_tracking_contract', lambda s: (False, 'bad'))
    assert sampling.max_consecutive_forecasts(make(sample_er_sde), c) == 1


def setup_dense(gap):
    t = tracker(gap)
    t._denoised_anchors = [_DenoisedAnchor(0, torch.tensor([0.])), _DenoisedAnchor(2, torch.tensor([2.]))]
    t._solver_coordinates = {0: torch.tensor(0.), 2: torch.tensor(2.), 3: torch.tensor(2.5), 4: torch.tensor(3.)}
    return t


def test_dense_gaps_and_legacy_guard():
    t = setup_dense(2)
    a = t.predict_causal_denoised(torch.zeros(1), desc(3))
    b = t.predict_causal_denoised(torch.zeros(1), desc(4))
    torch.testing.assert_close((b.value-a.value)/0.5, torch.ones(1))
    t = setup_dense(1); t._solver_coordinates[3] = torch.tensor(5.)
    assert t.predict_causal_denoised(torch.zeros(1), desc(3)).value.item() == 2
    t = setup_dense(2); t._solver_coordinates[3] = torch.tensor(5.); t._solver_coordinates[4] = torch.tensor(2.5)
    assert t.predict_causal_denoised(torch.zeros(1), desc(3)).value.item() == 4
    assert t.predict_causal_denoised(torch.zeros(1), desc(4)).value.item() == 4
    t = setup_dense(2); t._solver_coordinates[3] = torch.tensor(1.)
    assert t.predict_causal_denoised(torch.zeros(1), desc(3)).value.item() == 2
    assert t.predict_causal_denoised(torch.zeros(1), desc(4)).value.item() == 2


def draw(t):
    t.noise_scaler(torch.tensor(0.5)); t.noise_scaler(torch.tensor(1.))
    t.noise_sampler(torch.tensor(0.8), torch.tensor(0.4))
    return t._pending.value.clone()


def test_cumulative_compensation_and_single_cadence():
    t = tracker(2)
    q1 = draw(t); t.consume(torch.tensor([10.]), desc(1, 'replay'))
    q2 = draw(t)
    # alpha_t / alpha_s = 1 and r = 0.25
    torch.testing.assert_close(t.consume(torch.tensor([10.]), desc(2, 'replay')), 10 - (0.25*q1+q2))
    t = tracker(1); q1 = draw(t)
    torch.testing.assert_close(t.consume(torch.tensor([10.]), desc(1, 'replay')), 10-q1)
    draw(t); t.consume(torch.tensor([10.]), desc(2, 'actual'))
    q3 = draw(t)
    torch.testing.assert_close(t.consume(torch.tensor([10.]), desc(3, 'replay')), 10-q3)


def test_first_streak_and_late_guard(monkeypatch):
    original = SpectrumH3Runtime.start_run
    def start(self, *args, **kwargs):
        kwargs['multi_forecast_requires_prior_streak'] = True
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SpectrumH3Runtime, 'start_run', start)
    _, modes = _run_euler(SpectrumH3Config(max_forecast_run=2, window_size=6), 2)
    streaks = _streaks(modes)
    assert streaks[0][1] == 1
    assert any(length == 2 for _, length in streaks[1:])
    assert all(length == 1 for start, length in streaks if start >= 15)


def _native_er_sde_oracle(modes, *, track, noise_cut=None, d_fixed=None, seed=0):
    """Replay native sample_er_sde (stages 1-3) with a VP-like schedule whose alpha varies."""
    gen = torch.Generator().manual_seed(seed)
    n = len(modes)
    sig = torch.tensor(torch.linspace(1.0, 0.05, n).tolist() + [0.0], dtype=torch.float64)
    alphas = 1.0 / (1.0 + sig ** 2).sqrt()
    lam = sig / alphas
    offsets = torch.randn(n, 4, dtype=torch.float64, generator=gen)
    actual_values = torch.randn(n, 4, dtype=torch.float64, generator=gen)
    noises = [torch.randn(4, dtype=torch.float64, generator=gen) for _ in range(n)]
    x = torch.randn(4, dtype=torch.float64, generator=gen)
    if noise_cut is not None:
        noises = [v if i < noise_cut else torch.zeros_like(v) for i, v in enumerate(noises)]

    def scaler(value):
        return value * ((value ** 0.3).exp() + 10.0)

    draws = iter(noises)
    tracker_obj = None
    if track:
        tracker_obj = ERSDEStochasticTracker(
            noise_sampler=lambda s, t: next(draws), noise_scaler=scaler,
            effective_s_noise=1.0, max_stage=3, debug=False, run_id=1, max_dense_gap=2)
    call_scaler = tracker_obj.noise_scaler if tracker_obj else scaler
    old_d = old_dd = None
    xs, ds = [], []
    for i in range(n):
        xs.append(x.clone())
        if d_fixed is not None:
            d = d_fixed[i]
        else:
            d = actual_values[i].clone() if modes[i] == "A" else x - offsets[i]
            if tracker_obj is not None:
                # Exercise the exact-q path only (no dense anchors retained).
                tracker_obj._denoised_anchors.clear()
                mode = "actual" if modes[i] == "A" else "forecast"
                d = tracker_obj.consume(d, ERSDEStepDescriptor(1, i, mode, None, modes[i] != "A"))
        ds.append(d.clone())
        if sig[i + 1] == 0:
            break
        ls, lt = lam[i], lam[i + 1]
        r = call_scaler(lt.clone()) / call_scaler(ls.clone())
        x = (alphas[i + 1] / alphas[i]) * r * x + alphas[i + 1] * (1 - r) * d
        stage = min(3, i + 1)
        if stage >= 2:
            dt = lt - ls
            step = -dt / 200
            pos = lt + torch.arange(200, dtype=torch.float64) * step
            scaled = scaler(pos)
            dd = (d - old_d) / (ls - lam[i - 1])
            x = x + alphas[i + 1] * (dt + torch.sum(1 / scaled) * step * scaler(lt)) * dd
            if stage >= 3:
                su = torch.sum((pos - ls) / scaled) * step
                du = (dd - old_dd) / ((ls - lam[i - 2]) / 2)
                x = x + alphas[i + 1] * ((dt ** 2) / 2 + su * scaler(lt)) * du
            old_dd = dd
        draw = tracker_obj.noise_sampler(sig[i], sig[i + 1]) if tracker_obj else noises[i]
        x = x + alphas[i + 1] * draw * (lt ** 2 - ls ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_d = d
    return xs, ds, offsets, alphas


def test_cumulative_compensation_matches_native_solver_oracle():
    modes = ["A", "A", "A", "A", "F", "F", "A", "F", "F", "A"]
    _, ds, offsets, alphas = _native_er_sde_oracle(modes, track=True)
    # The schedule must actually vary alpha, or a missing r_alpha factor would pass.
    assert not torch.allclose(alphas[5] / alphas[4], torch.tensor(1.0, dtype=torch.float64))
    for cut, steps in ((4, (4, 5)), (7, (7, 8))):
        # Reference: same denoised values, but no noise injected from the streak's source step on.
        ref_xs, _, _, _ = _native_er_sde_oracle(modes, track=False, noise_cut=cut - 1, d_fixed=ds)
        for j in steps:
            assert torch.allclose(ds[j], ref_xs[j] - offsets[j], atol=1e-10, rtol=0)
