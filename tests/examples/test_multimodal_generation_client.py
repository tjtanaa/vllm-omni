# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for displaying non-streaming multimodal chat responses."""

from argparse import Namespace
from unittest.mock import Mock

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionAudio, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from examples.online_serving.openai_chat_completion_client_for_multimodal_generation import (
    run_multimodal_generation,
)
from tests.examples.helpers import extract_content_after_keyword, strip_trailing_audio_saved_line

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("output", ["text", "audio", "combined", "separate"])
def test_non_streaming_response_preserves_text_and_audio(output, tmp_path, monkeypatch, capsys):
    text = "These are cherry blossoms."
    audio = ChatCompletionAudio(id="audio-test", data="UklGRg==", expires_at=0, transcript=text)
    messages = {
        "text": [ChatCompletionMessage(role="assistant", content=text)],
        "audio": [ChatCompletionMessage(role="assistant", audio=audio)],
        "combined": [ChatCompletionMessage(role="assistant", content=text, audio=audio)],
        "separate": [
            ChatCompletionMessage(role="assistant", content=text),
            ChatCompletionMessage(role="assistant", audio=audio),
        ],
    }[output]
    response = ChatCompletion(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=message) for message in messages],
    )
    client = Mock(spec=OpenAI)
    client.chat = Mock()
    client.chat.completions.create.return_value = response
    args = Namespace(
        model="test-model",
        modalities=None,
        num_concurrent_requests=1,
        speaker=None,
        query_type="text",
        stream=False,
    )
    monkeypatch.chdir(tmp_path)

    run_multimodal_generation(args, client)

    stdout = capsys.readouterr().out
    if output != "audio":
        content = extract_content_after_keyword("Chat completion output from text:", stdout)
        assert strip_trailing_audio_saved_line(content) == text
    else:
        assert "Chat completion output from text:" not in stdout
    if output != "text":
        assert "Audio saved to audio_chatcmpl-test_0.wav" in stdout
        assert (tmp_path / "audio_chatcmpl-test_0.wav").read_bytes() == b"RIFF"
    else:
        assert "Audio saved to" not in stdout
        assert not list(tmp_path.iterdir())
