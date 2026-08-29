# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
from typing import Any

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

AUDIO_MODEL: dict[str, dict[str, int | None]] = {
    # The inference peak includes backend workspaces as well as model weights.
    # On ROCm, large MIOpen workspaces and allocator fragmentation can mask the
    # resident-weight reduction, so use a conservative floor that still catches
    # a disabled/no-op layerwise offloader.
    "stabilityai/stable-audio-open-1.0": {"cuda": 1500, "rocm": 512},
}

IMAGE_VIDEO_MODELS: dict[str, dict[str, int | None]] = {
    "riverclouds/qwen_image_random": {"cuda": 4500, "rocm": None},
    # "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"cuda": 45000, "rocm": None},
}

MODELS: dict[str, dict[str, int | None]] = {**AUDIO_MODEL, **IMAGE_VIDEO_MODELS}

MODEL_MARKS = {
    "riverclouds/qwen_image_random": pytest.mark.core_model,
    "stabilityai/stable-audio-open-1.0": pytest.mark.full_model,
}

AUDIO_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "runner_params": {},
    "sampler_params": {},
}

IMAGE_VIDEO_MODELS_PARAMS: dict[str, dict[str, Any]] = {
    "runner_params": {"boundary_ratio": 0.875, "flow_shift": 5.0},
    "sampler_params": {"height": 480, "width": 640, "num_frames": 5},
}


def check_audio_determinism(audio1, audio2, atol=1e-2):
    device = current_omni_platform.device_type
    if isinstance(audio1, np.ndarray):
        audio1 = torch.from_numpy(audio1).to(device)
    if isinstance(audio2, np.ndarray):
        audio2 = torch.from_numpy(audio2).to(device)

    if not torch.allclose(audio1, audio2, atol=atol):
        diff = torch.abs(audio1 - audio2)
        print(f"Max difference: {diff.max().item()}")
        print(f"Mean difference: {diff.mean().item()}")
        raise AssertionError(f"Audio outputs differ beyond tolerance atol={atol}")
    return True


def worker_peak_memory_mb(output: list[Any]) -> float:
    """Read the request-scoped peak reported by the diffusion worker."""
    assert output, "Diffusion worker returned no output"
    peak = float(getattr(output[0], "peak_memory_mb", 0.0) or 0.0)
    assert peak > 0, "Diffusion worker did not report request peak memory"
    return peak


def run_inference(
    model_name: str,
    layerwise_offload: bool = False,
    num_inference_steps: int = 3,
) -> tuple[float, Any]:
    current_omni_platform.empty_cache()

    if model_name in AUDIO_MODEL:
        params = AUDIO_MODEL_PARAMS
    else:
        params = IMAGE_VIDEO_MODELS_PARAMS

    with OmniRunner(
        model_name,
        enable_layerwise_offload=layerwise_offload,
        # TODO: we might want to add overlapped feature e2e tests
        # cache_backend="cache_dit",
        **params["runner_params"],
    ) as runner:
        # Refer to tests/e2e/offline_inference/test_wan22.py
        # Use minimal settings for testing
        output = runner.omni.generate(
            "A cat sitting on a table",
            OmniDiffusionSamplingParams(
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
                guidance_scale=1.0,
                num_inference_steps=num_inference_steps,
                **params["sampler_params"],
            ),
        )

    # Inference runs in StageDiffusionProc, not in this pytest process. The
    # worker resets its allocator peak immediately before pipeline.forward and
    # propagates that request-scoped value through OmniRequestOutput. A parent
    # process DeviceMemoryMonitor observes total device usage instead, including
    # unrelated CI processes and allocator state outside the worker, which can
    # invert a relatively small offload saving.
    peak = worker_peak_memory_mb(output)

    gc.collect()
    current_omni_platform.empty_cache()

    return peak, output


def test_worker_peak_memory_mb_uses_request_metric():
    output = [type("Output", (), {"peak_memory_mb": 4096.5})()]

    assert worker_peak_memory_mb(output) == 4096.5


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("model_name", list(MODELS.keys()))
def test_layerwise_offload_diffusion_model(model_name: str):
    """Test that layerwise offloading reduces GPU memory usage.

    This test verifies that layerwise offloading significantly reduces peak
    GPU memory usage compared to loading the entire model on GPU. The layerwise
    offloader keeps only a single transformer block on GPU at a time, with
    prefetching for compute-memory overlap.
    """
    try:
        # Run without layerwise offloading (baseline)
        no_offload_peak_memory, output_no_offload = run_inference(model_name, layerwise_offload=False)
        cleanup_dist_env_and_memory()

        # Run with layerwise offloading (1 layer on device)
        layerwise_offload_peak_memory, output_offload = run_inference(model_name, layerwise_offload=True)
        cleanup_dist_env_and_memory()
    except ValueError as exc:
        # omni_snapshot_download wraps GatedRepoError in a ValueError; skip instead of failing.
        if "Access to model" in str(exc) and "is restricted" in str(exc):
            pytest.skip(
                f"Skipping: gated HF repo {model_name!r} inaccessible "
                f"({exc}). See docs/contributing/ci/hf_credentials.md."
            )
        pytest.fail(f"Inference failed: {exc}")
    except Exception:
        pytest.fail("Inference failed")

    print(f"Layerwise offload peak memory (1 GPU layer): {layerwise_offload_peak_memory} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")

    if model_name == "stabilityai/stable-audio-open-1.0":
        audio_offload = output_offload[0].multimodal_output.get("audio")
        audio_no_offload = output_no_offload[0].multimodal_output.get("audio")
        # Match the sibling cpu-offload test's tolerance: layerwise offload moves
        # blocks across the PCIe bus on a side stream, which can perturb cuBLAS
        # algorithm selection and produce ~ULP-level drift larger than 1e-3.
        check_audio_determinism(audio_offload, audio_no_offload, atol=1e-2)

    is_rocm = torch.version.hip is not None
    platform = "rocm" if is_rocm else "cuda"
    expected_saved_memory = MODELS[model_name][platform]

    if expected_saved_memory is None:
        pytest.skip(f"Threshold not defined for {platform} on {model_name}")
    assert expected_saved_memory is not None

    # Verify that layerwise offloading significantly reduces memory usage
    # Passes only if the actual savings meets the expected savings
    actual_saved_memory = no_offload_peak_memory - layerwise_offload_peak_memory
    assert layerwise_offload_peak_memory + expected_saved_memory <= no_offload_peak_memory, (
        f"Layerwise offload peak memory {layerwise_offload_peak_memory} MB "
        f"should be at least {expected_saved_memory} MB less than no offload peak memory "
        f"{no_offload_peak_memory} MB (actual savings: {actual_saved_memory} MB)"
    )
