from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import max_consecutive_forecasts

TOPOLOGY = (
    ("video", (1, 24, 2, 4, 4)),
    ("audio", (1, 32, 2, 8)),
    ("hidden", 4),
    ("target_audio_rows", 1),
    ("target_video_rows", 2),
)
LABEL = ((0, "positive"),)


def _run_euler(config, cap, steps=20):
    runtime = SpectrumH3Runtime(config)
    runtime.start_run(
        torch.linspace(1.0, 0.0, steps + 1),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=cap,
        min_actual_steps_after_forecast=1,
    )
    modes = []
    for step, sigma in enumerate(torch.linspace(1.0, 0.05, steps)):
        decision = runtime.begin_step(sigma)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            runtime.observe_actual(
                decision["run_id"],
                decision["step_id"],
                call_id,
                torch.full((1, 3, 4), float(step)),
            )
        else:
            runtime.predict(
                decision["run_id"],
                decision["step_id"],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        modes.append("A" if actual else "F")
        runtime.finalize_step(decision["run_id"], decision["step_id"])
    return runtime, modes


def _streaks(modes):
    result = []
    start = None
    for index, mode in enumerate(modes + ["A"]):
        if mode == "F" and start is None:
            start = index
        elif mode != "F" and start is not None:
            result.append((start, index - start))
            start = None
    return result


def _forecast_indices(modes):
    return [i for i, m in enumerate(modes) if m == "F"]


def test_default_config_cap_one_reproduces_existing_schedule():
    runtime, modes = _run_euler(SpectrumH3Config(), 1)

    assert _forecast_indices(modes) == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert runtime.stats.actual_steps == 11
    assert runtime.stats.forecast_steps == 9


@pytest.mark.parametrize("cap", (2, 3, 4))
@pytest.mark.parametrize("window_size", (2.0, 6.0))
def test_forecast_runs_respect_cap_tail_and_late_schedule(cap, window_size):
    config = SpectrumH3Config(max_forecast_run=cap, window_size=window_size)
    runtime, modes = _run_euler(config, cap)
    streaks = _streaks(modes)
    late_start = -(-3 * 20 // 4)

    assert streaks
    assert max(length for _, length in streaks) <= cap
    assert max(length for _, length in streaks) > 1
    for start, length in streaks:
        end = start + length
        assert end < 20  # every streak is followed by an actual
        if length > 1:
            assert end <= late_start
    assert modes[-1] == "A"
    for index in range(late_start, 20):
        if modes[index] == "F":
            assert modes[index - 1] == "A"
    assert runtime.stats.actual_steps + runtime.stats.forecast_steps == 20


def test_cap_two_and_three_concrete_indices():
    got = {}
    for cap in (2, 3):
        for window_size in (2.0, 6.0):
            config = SpectrumH3Config(max_forecast_run=cap, window_size=window_size)
            _, modes = _run_euler(config, cap)
            got[(cap, window_size)] = _forecast_indices(modes)
    assert got == {
        (2, 2.0): [1, 3, 5, 7, 8, 10, 11, 13, 14, 16, 18],
        (2, 6.0): [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 17],
        (3, 2.0): [1, 3, 5, 7, 8, 10, 11, 12, 14, 16, 18],
        (3, 6.0): [1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 17],
    }


def _install_fake_native_euler(monkeypatch):
    def sample_euler():
        pass

    sampling = ModuleType("comfy.k_diffusion.sampling")
    sampling.sample_euler = sample_euler
    kdiff = ModuleType("comfy.k_diffusion")
    kdiff.sampling = sampling
    comfy = ModuleType("comfy")
    comfy.k_diffusion = kdiff
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion", kdiff)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", sampling)
    return sample_euler


def _named(function_name):
    def sampler_function():
        pass

    sampler_function.__name__ = function_name
    return SimpleNamespace(sampler_function=sampler_function, extra_options={})


def test_max_consecutive_forecasts_gate(monkeypatch):
    native = _install_fake_native_euler(monkeypatch)
    config = SpectrumH3Config(max_forecast_run=3)

    def euler(options):
        return SimpleNamespace(sampler_function=native, extra_options=options)

    assert max_consecutive_forecasts(euler({}), config) == 3
    assert max_consecutive_forecasts(euler({"s_churn": 0.0, "s_noise": 1.0}), config) == 3
    assert max_consecutive_forecasts(euler({}), None) == 1
    assert max_consecutive_forecasts(euler({}), SpectrumH3Config()) == 1
    assert max_consecutive_forecasts(euler({"s_churn": 5.0}), config) == 1
    assert max_consecutive_forecasts(euler({"s_churn": "bad"}), config) == 1
    assert max_consecutive_forecasts(euler({"mystery": 1}), config) == 1
    assert max_consecutive_forecasts(_named("sample_euler"), config) == 1
    rollback = SpectrumH3Config(
        max_forecast_run=3,
        selective_rollback_correction=True,
        offline_smoothing_replay=False,
    )
    assert max_consecutive_forecasts(euler({}), rollback) == 1
    feedback = SpectrumH3Config(
        max_forecast_run=3,
        anchor_residual_feedback=True,
        offline_smoothing_replay=False,
    )
    assert max_consecutive_forecasts(euler({}), feedback) == 1
    for mode in ("schedule", "schedule_confidence", "full"):
        model_aware = SpectrumH3Config(max_forecast_run=3, model_aware_mode=mode)
        assert max_consecutive_forecasts(euler({}), model_aware) == 1
    for name in (
        "sample_res_multistep",
        "sample_er_sde",
        "_turbo_sampler",
        "sample_seeds_2",
        "sample_seeds_3",
        "sample_sa_solver",
        "sample_sa_solver_pece",
    ):
        assert max_consecutive_forecasts(_named(name), config) == 1


def test_missing_native_sampling_module_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", None)
    config = SpectrumH3Config(max_forecast_run=3)
    assert max_consecutive_forecasts(_named("sample_euler"), config) == 1


@pytest.mark.parametrize("value", (0, 5, True, 1.0, "2"))
def test_config_rejects_bad_max_forecast_run(value):
    with pytest.raises(ValueError):
        SpectrumH3Config(max_forecast_run=value).validate()


@pytest.mark.parametrize("value", (1, 2, 3, 4))
def test_config_accepts_max_forecast_run(value):
    SpectrumH3Config(max_forecast_run=value).validate()


def test_node_exposes_max_forecast_run_last():
    types = SpectrumApplyMiniMaxH3.INPUT_TYPES()
    optional = types["optional"]
    assert list(optional)[-1] == "max_forecast_run"
    assert "max_forecast_run" not in types["required"]
    assert optional["max_forecast_run"][0] == "INT"
    spec = optional["max_forecast_run"][1]
    assert (spec["default"], spec["min"], spec["max"]) == (1, 1, 4)
