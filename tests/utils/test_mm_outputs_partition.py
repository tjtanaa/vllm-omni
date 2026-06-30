import torch

from vllm_omni.utils.mm_outputs import partition_flat_payload, partition_payload_list


def test_partition_thinker_latent_payload():
    payload = {
        "hidden_states.layer_0": torch.zeros(2, 4),
        "hidden_states.layer_24": torch.zeros(2, 8),
        "embed.tts_bos": [torch.zeros(1, 1, 4)],
    }
    inter, client = partition_flat_payload(payload)
    assert inter == payload
    assert client == {}


def test_partition_talker_intermediate_codes():
    payload = {
        "codes.audio": torch.zeros(3, 2),
        "hidden": torch.zeros(3, 16),
    }
    inter, client = partition_flat_payload(payload)
    assert inter == payload
    assert client == {}


def test_partition_code2wav_client_audio():
    payload = {
        "model_outputs": torch.zeros(1, 2400),
        "sr": torch.tensor(24000, dtype=torch.int32),
    }
    inter, client = partition_flat_payload(payload)
    # Client gets the allowlisted final-output roots...
    assert client == payload
    # ...and the inter-stage payload is non-lossy: it keeps every key so a
    # downstream stage can still consume model_outputs/sr if needed.
    assert inter == payload


def test_partition_non_lossy_inter_stage_for_client_root():
    # Regression for #4527: a value under a client-facing root (model_outputs)
    # that a downstream stage also needs must stay in the inter-stage connector
    # payload, not be siphoned only to the client (which starved the next stage
    # and produced empty audio / 300s connector-input timeouts).
    payload = {
        "model_outputs": torch.zeros(1, 8),
        "talker_text_offset": torch.zeros(1, dtype=torch.int32),
    }
    inter, client = partition_flat_payload(payload)
    assert inter == payload
    assert client == {"model_outputs": payload["model_outputs"]}


def test_partition_payload_list_preserves_request_alignment():
    payloads = [
        {"hidden_states.layer_0": torch.zeros(1, 2)},
        {"model_outputs": torch.zeros(1, 10)},
    ]
    inter_list, client_list = partition_payload_list(payloads)
    # Inter-stage is non-lossy: every request keeps its full payload.
    assert inter_list == payloads
    # Client copy only carries allowlisted client-facing roots.
    assert client_list == [None, payloads[1]]
