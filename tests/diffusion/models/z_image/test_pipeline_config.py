# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3Config

from vllm_omni.diffusion.model_loader import hub_prefetch
from vllm_omni.diffusion.models.z_image import pipeline_z_image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _ConfigLoadedError(Exception):
    """Stop initialization before allocating model weights."""


@pytest.mark.parametrize("failure", ["transient", "persistent", "local"])
def test_text_encoder_config_cache_failure(monkeypatch, tmp_path, failure):
    model = str(tmp_path) if failure == "local" else "Tongyi-MAI/Z-Image-Turbo"
    config = Qwen3Config()
    error = ValueError(f"Unrecognized model in {model}. Should have a `model_type` key in its config.json.")
    load_config = Mock(side_effect=[error, config] if failure == "transient" else error)
    prefetch = Mock()
    create_model = Mock(side_effect=_ConfigLoadedError)
    monkeypatch.setattr(pipeline_z_image, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_z_image, "prefetch_subfolders", prefetch)
    monkeypatch.setattr(hub_prefetch, "prefetch_subfolders", prefetch)
    monkeypatch.setattr(hub_prefetch.time, "sleep", lambda _: None)
    monkeypatch.setattr(pipeline_z_image.FlowMatchEulerDiscreteScheduler, "from_pretrained", Mock())
    monkeypatch.setattr(pipeline_z_image.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(pipeline_z_image, "create_transformers_model", create_model)

    od_config = SimpleNamespace(model=model, revision=None)
    if failure == "transient":
        with pytest.raises(_ConfigLoadedError):
            pipeline_z_image.ZImagePipeline(od_config=od_config)
        assert load_config.call_count == 2
        assert create_model.call_args.kwargs["hf_config"] is config
    else:
        with pytest.raises(ValueError, match="model_type") as exc_info:
            pipeline_z_image.ZImagePipeline(od_config=od_config)
        assert exc_info.value is error
        assert load_config.call_count == (1 if failure == "local" else 3)
        create_model.assert_not_called()

    assert prefetch.call_count == load_config.call_count
    for call in prefetch.call_args_list:
        assert call.args == (model, ["scheduler", "text_encoder", "vae", "tokenizer"])
    assert all(call.kwargs["subfolder"] == "text_encoder" for call in load_config.call_args_list)
    assert all(call.kwargs["local_files_only"] == (failure == "local") for call in load_config.call_args_list)
