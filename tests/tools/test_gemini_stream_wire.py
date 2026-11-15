"""Gemini streaming wire contract; no audio device, NumPy or paid request."""
import base64
import json
from unittest.mock import MagicMock

import pytest
import tools.tts_streaming as ts


@pytest.mark.parametrize("model, modern", [
    ("gemini-3.8-flash-tts", True),
    ("gemini-2.5-flash-preview-tts", False),
])
def test_gemini_stream_requests_matching_voice_and_headerless_audio(monkeypatch, model, modern):
    monkeypatch.setattr(ts, "_gemini_key", lambda: "unit-test-not-a-real-key")
    pcm = b"\x01\x00\x02\x00"
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = iter([
        "data: " + json.dumps({"candidates": [{"content": {"parts": [{"inlineData": {
            "data": base64.b64encode(pcm).decode(),
        }}]}}]}),
    ])
    post = MagicMock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    section = {"model": model, "voice": "sample-voice"}
    assert list(ts.GeminiStreamer({}, section).stream("Hello.")) == [pcm]
    payload = post.call_args.kwargs["json"]
    assert payload["contents"][0]["parts"] == [{"text": "Hello."}]
    config = payload["generationConfig"]
    if modern:
        assert config["speechConfig"] == {"voiceConfig": {"voice": "sample-voice"}}
        assert config["responseFormat"] == {"audio": {"mimeType": "AUDIO_L16", "sampleRate": 24000}}
    else:
        assert config["speechConfig"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "sample-voice"}}}
        assert "responseFormat" not in config
