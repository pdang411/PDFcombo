"""Piper TTS Web UI - Run directly"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import wave

import requests
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import VOICES_JSON, download_voice

from contextlib import asynccontextmanager

_LOGGER = logging.getLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_voice, MODEL
    logging.basicConfig(level=logging.INFO)

    download_dir = Path(DATA_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(
        download_dir.glob("*.onnx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if onnx_files:
        newest = onnx_files[0]
        model_id = newest.stem
        _LOGGER.info(f"Found existing voice model: {model_id}")
        try:
            default_voice = PiperVoice.load(newest, use_cuda=USE_CUDA)
            loaded_voices[model_id] = default_voice
            MODEL = model_id
            _LOGGER.info(f"Loaded voice: {model_id}")
        except Exception as e:
            _LOGGER.error(f"Failed to load voice model {model_id}: {e}")
            default_voice = None
    else:
        _LOGGER.info(f"No voice models found. Downloading default: {MODEL}")
        try:
            download_voice(MODEL, download_dir)
            model_path = download_dir / f"{MODEL}.onnx"
            if model_path.exists():
                default_voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                loaded_voices[MODEL] = default_voice
                _LOGGER.info(f"Downloaded and loaded voice: {MODEL}")
        except Exception as e:
            _LOGGER.error(f"Failed to download default voice model: {e}")
            default_voice = None

    yield


app = FastAPI(title="Piper TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

MODEL = os.getenv("MODEL", "en_GB-cori-medium")
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"
SSL_CERTFILE = os.getenv("SSL_CERTFILE", "")
SSL_KEYFILE = os.getenv("SSL_KEYFILE", "")

loaded_voices: Dict[str, PiperVoice] = {}
default_voice: Optional[PiperVoice] = None


class DownloadRequest(BaseModel):
    voice: str
    force_redownload: bool = False


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None
    response_format: Optional[str] = "wav"


@app.get("/")
async def root():
    return {"message": "Piper TTS API", "model": MODEL, "ui": "/ui"}


@app.get("/health")
async def health():
    return {
        "status": "ok" if default_voice is not None else "degraded",
        "message": "Ready"
        if default_voice is not None
        else "No voice model loaded - use /download to download a voice",
        "model": MODEL,
        "default_voice_ready": default_voice is not None,
        "loaded_voices": list(loaded_voices.keys()),
    }


@app.get("/ui", response_class=HTMLResponse)
async def webui():
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/voices")
async def list_voices() -> Dict[str, Any]:
    voices_dict: Dict[str, Any] = {}

    for onnx_path in Path(DATA_DIR).glob("*.onnx"):
        config_path = Path(f"{onnx_path}.onnx.json")
        if config_path.exists():
            model_id = onnx_path.stem
            with open(config_path, encoding="utf-8") as config_file:
                voices_dict[model_id] = json.load(config_file)

    return voices_dict


@app.get("/all-voices")
async def list_all_voices():
    try:
        with urlopen(VOICES_JSON, timeout=30) as response:
            return json.load(response)
    except Exception as e:
        _LOGGER.warning(f"Failed to fetch voices list: {e}")
        return {}


@app.post("/download")
async def download_voice_endpoint(req: DownloadRequest):
    global MODEL, default_voice
    download_dir = Path(DATA_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    voice_name = req.voice.strip('"').strip("'")

    try:
        download_voice(voice_name, download_dir, force_redownload=req.force_redownload)

        model_path = download_dir / f"{voice_name}.onnx"
        if model_path.exists():
            try:
                voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                loaded_voices[voice_name] = voice
                MODEL = voice_name
                # Always set the newly downloaded voice as default
                default_voice = voice

                return {
                    "status": "success",
                    "voice": voice_name,
                    "message": "Voice downloaded and loaded successfully",
                }
            except Exception as e:
                return {
                    "status": "loaded",
                    "voice": voice_name,
                    "message": f"Voice downloaded but failed to load: {e}",
                }
        else:
            raise HTTPException(
                status_code=500, detail="Download failed - model file not created"
            )

    except Exception as e:
        _LOGGER.error(f"Voice download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/synthesize")
async def synthesize_get(
    text: str,
    voice: Optional[str] = None,
    speaker: Optional[str] = None,
    speaker_id: Optional[int] = None,
    length_scale: Optional[float] = None,
    noise_scale: Optional[float] = None,
    noise_w_scale: Optional[float] = None,
) -> Response:
    req = SynthesizeRequest(
        text=text,
        voice=voice,
        speaker=speaker,
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )
    return await synthesize(req)


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    model_id = (req.voice if req.voice else MODEL).strip('"').strip("'")

    voice = loaded_voices.get(model_id)

    if voice is None:
        model_path = Path(DATA_DIR) / f"{model_id}.onnx"
        if model_path.exists():
            try:
                voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                loaded_voices[model_id] = voice
            except Exception as e:
                _LOGGER.error(f"Failed to load voice {model_id}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to load voice model: {str(e)}"
                )

    if voice is None:
        voice = default_voice

    if voice is None:
        raise HTTPException(
            status_code=503,
            detail="No voice model available. Please download a voice model using the /ui interface or POST to /download",
        )

    speaker_id = req.speaker_id
    if voice.config.num_speakers > 1 and speaker_id is None:
        if req.speaker:
            speaker_id = voice.config.speaker_id_map.get(req.speaker)
        if speaker_id is None:
            speaker_id = 0

    if speaker_id is not None and speaker_id >= voice.config.num_speakers:
        speaker_id = 0

    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=req.length_scale or voice.config.length_scale,
        noise_scale=req.noise_scale or voice.config.noise_scale,
        noise_w_scale=req.noise_w_scale or voice.config.noise_w_scale,
    )

    if req.response_format == "pcm":
        audio_chunks = []
        first = True
        for audio_chunk in voice.synthesize(req.text, syn_config):
            if first:
                sr = audio_chunk.sample_rate
                sw = audio_chunk.sample_width
                sc = audio_chunk.sample_channels
                first = False
            audio_chunks.append(audio_chunk.audio_int16_bytes)
        return Response(
            content=b"".join(audio_chunks),
            media_type="audio/L16",
            headers={
                "X-Piper-Sample-Rate": str(sr),
                "X-Piper-Sample-Width": str(sw),
                "X-Piper-Channels": str(sc),
            },
        )

    with io.BytesIO() as wav_io:
        wav_file = wave.open(wav_io, "wb")
        with wav_file:
            wav_params_set = False
            for i, audio_chunk in enumerate(voice.synthesize(req.text, syn_config)):
                if not wav_params_set:
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)
                    wav_params_set = True
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        return Response(content=wav_io.getvalue(), media_type="audio/wav")


def _get_voice_and_config(req_or_text) -> tuple:
    if isinstance(req_or_text, SynthesizeRequest):
        req = req_or_text
        model_id = (req.voice if req.voice else MODEL).strip('"').strip("'")
        text = req.text
        speaker_id = req.speaker_id
        speaker = req.speaker
        length_scale = req.length_scale
        noise_scale = req.noise_scale
        noise_w_scale = req.noise_w_scale
    else:
        text, voice_name, speaker_id_val, speaker_val, ls_val, ns_val, nws_val = req_or_text
        model_id = (voice_name if voice_name else MODEL).strip('"').strip("'")
        text = text
        speaker_id = speaker_id_val
        speaker = speaker_val
        length_scale = ls_val
        noise_scale = ns_val
        noise_w_scale = nws_val

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    voice = loaded_voices.get(model_id)
    if voice is None:
        model_path = Path(DATA_DIR) / f"{model_id}.onnx"
        if model_path.exists():
            try:
                voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                loaded_voices[model_id] = voice
            except Exception as e:
                _LOGGER.error(f"Failed to load voice {model_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load voice model: {str(e)}")

    if voice is None:
        voice = default_voice

    if voice is None:
        raise HTTPException(status_code=503, detail="No voice model available. Please download a voice model using the /ui interface or POST to /download")

    sid = speaker_id
    if voice.config.num_speakers > 1 and sid is None:
        if speaker:
            sid = voice.config.speaker_id_map.get(speaker)
        if sid is None:
            sid = 0

    if sid is not None and sid >= voice.config.num_speakers:
        sid = 0

    syn_config = SynthesisConfig(
        speaker_id=sid,
        length_scale=length_scale or voice.config.length_scale,
        noise_scale=noise_scale or voice.config.noise_scale,
        noise_w_scale=noise_w_scale or voice.config.noise_w_scale,
    )
    return voice, syn_config, text


@app.get("/synthesize/stream")
async def synthesize_stream(
    text: str,
    voice: Optional[str] = None,
    speaker_id: Optional[int] = None,
    length_scale: Optional[float] = None,
    noise_scale: Optional[float] = None,
    noise_w_scale: Optional[float] = None,
):
    voice_model, syn_config, clean_text = _get_voice_and_config(
        (text, voice, speaker_id, None, length_scale, noise_scale, noise_w_scale)
    )

    async def generate() -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def fill_queue():
            try:
                for chunk in voice_model.synthesize(clean_text, syn_config):
                    queue.put_nowait(chunk)
            except Exception as e:
                queue.put_nowait(e)
            finally:
                queue.put_nowait(sentinel)

        loop.run_in_executor(None, fill_queue)

        first = True
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            chunk = item
            if first:
                header = json.dumps({
                    "sample_rate": chunk.sample_rate,
                    "sample_width": chunk.sample_width,
                    "channels": chunk.sample_channels,
                }).encode() + b"\n"
                yield header
                first = False
            yield chunk.audio_int16_bytes

    return StreamingResponse(
        generate(),
        media_type="audio/L16",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def audio_speech(req: SynthesizeRequest) -> Response:
    return await synthesize(req)





def _pipeline_synthesize(text: str):
    if not text.strip():
        return
    try:
        voice, syn_config, _ = _get_voice_and_config(
            (text, None, None, None, None, None, None)
        )
        for audio_chunk in voice.synthesize(text, syn_config):
            yield {
                "sr": audio_chunk.sample_rate,
                "data": audio_chunk.audio_int16_bytes,
            }
    except Exception as e:
        _LOGGER.error(f"Pipeline TTS error: {e}")



@app.websocket("/ws/voice")
async def voice_pipeline(websocket: WebSocket):
    await websocket.accept()
    _LOGGER.info("Voice pipeline WebSocket connected")

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")
            data = msg.get("data", "").strip()

            if msg_type == "cancel":
                _LOGGER.info("Pipeline cancelled by user")
                continue

            if msg_type != "text" or not data:
                await websocket.send_json({"type": "error", "data": "Expected text message"})
                continue

            await websocket.send_json({"type": "status", "state": "thinking"})

            full_text = ""
            tts_buffer = ""

            for llm_chunk in _llm_stream(data):
                full_text += llm_chunk
                tts_buffer += llm_chunk

                await websocket.send_json({"type": "text", "data": llm_chunk})

                if len(tts_buffer) > 5 and any(c in tts_buffer for c in ".!?"):
                    last_boundary = max(
                        tts_buffer.rfind(c) for c in ".!?" if c in tts_buffer
                    )
                    sentence = tts_buffer[: last_boundary + 1]
                    tts_buffer = tts_buffer[last_boundary + 1 :]

                    if sentence.strip():
                        await websocket.send_json({"type": "status", "state": "speaking"})
                        for chunk in _pipeline_synthesize(sentence):
                            pcm_b64 = base64.b64encode(chunk["data"]).decode("ascii")
                            await websocket.send_json({
                                "type": "audio",
                                "pcm": pcm_b64,
                                "sr": chunk["sr"],
                            })

            if tts_buffer.strip():
                await websocket.send_json({"type": "status", "state": "speaking"})
                for chunk in _pipeline_synthesize(tts_buffer):
                    pcm_b64 = base64.b64encode(chunk["data"]).decode("ascii")
                    await websocket.send_json({
                        "type": "audio",
                        "pcm": pcm_b64,
                        "sr": chunk["sr"],
                    })

            await websocket.send_json({"type": "status", "state": "done"})

    except WebSocketDisconnect:
        _LOGGER.info("Voice pipeline WebSocket disconnected")
    except Exception as e:
        _LOGGER.error(f"Voice pipeline error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


PIPELINE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Pipeline</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #111; color: #eee; height: 100vh; display: flex; flex-direction: column; }
  #header { padding: 16px 20px; background: #1a1a2e; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #333; }
  #orb { width: 32px; height: 32px; border-radius: 50%; transition: all 0.3s; background: #555; flex-shrink: 0; }
  #orb.idle { background: #555; box-shadow: none; }
  #orb.listening { background: #22c55e; box-shadow: 0 0 20px #22c55e88; }
  #orb.thinking { background: #f59e0b; box-shadow: 0 0 20px #f59e0b88; animation: pulse 1s infinite; }
  #orb.speaking { background: #3b82f6; box-shadow: 0 0 30px #3b82f688; animation: pulse 0.5s infinite; }
  @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.15); } }
  #status-label { font-size: 14px; color: #999; }
  #chat { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { padding: 10px 14px; border-radius: 12px; max-width: 80%; line-height: 1.5; font-size: 15px; }
  .msg.user { background: #1e3a5f; align-self: flex-end; }
  .msg.assistant { background: #2a2a3e; align-self: flex-start; }
  #controls { padding: 16px 20px; background: #1a1a2e; border-top: 1px solid #333; display: flex; gap: 10px; align-items: center; }
   #ptt-btn { padding: 10px 18px; border: none; border-radius: 20px; font-size: 14px; cursor: pointer; background: #dc2626; color: white; flex-shrink: 0; }
   #ptt-btn:hover { background: #b91c1c; }
   #ptt-btn:disabled { opacity: 0.5; cursor: not-allowed; }
   #ptt-btn.listening { background: #22c55e; }
  #input-area { flex: 1; display: flex; gap: 8px; }
  #text-input { flex: 1; padding: 10px 14px; border-radius: 20px; border: 1px solid #444; background: #222; color: #eee; font-size: 15px; outline: none; }
  #text-input:focus { border-color: #3b82f6; }
  #send-btn { padding: 10px 20px; border: none; border-radius: 20px; background: #3b82f6; color: white; font-size: 15px; cursor: pointer; }
  #send-btn:hover { background: #2563eb; }
  #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .controls-row { display: flex; gap: 8px; align-items: center; }
  #clear-btn { padding: 8px 16px; border: 1px solid #555; border-radius: 16px; background: transparent; color: #999; cursor: pointer; font-size: 13px; }
  #clear-btn:hover { background: #333; color: #eee; }
</style>
</head>
<body>
<div id="header">
  <div id="orb" class="idle"></div>
  <span id="status-label">Idle</span>
  <span style="flex:1"></span>
  <span style="font-size:13px;color:#555;">Voice Pipeline</span>
</div>
<div id="chat"></div>
<div id="controls">
  <button id="ptt-btn">Push-to-Talk</button>
  <div id="input-area">
    <input id="text-input" type="text" placeholder="Type a message..." autofocus>
    <button id="send-btn">Send</button>
  </div>
  <button id="clear-btn">Clear</button>
</div>
<script>
let ws = null;
const orb = document.getElementById('orb');
const statusLabel = document.getElementById('status-label');
const chat = document.getElementById('chat');
const pttBtn = document.getElementById('ptt-btn');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');

let audioCtx = null;
let pttRecognition = null;
let isConnected = false;
let isAssSpeaking = false;

function setState(state) {
  orb.className = state;
  const labels = { idle: 'Idle', listening: 'Listening...', thinking: 'Thinking...', speaking: 'Speaking' };
  statusLabel.textContent = labels[state] || state;
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'assistant') {
    const last = chat.lastElementChild;
    if (last && last.classList.contains('assistant') && !last.dataset.final) {
      last.textContent += text;
      chat.scrollTop = chat.scrollHeight;
      return;
    }
  }
  div.textContent = text;
  div.dataset.final = 'true';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function finalizeLastAssistant() {
  const last = chat.lastElementChild;
  if (last && last.classList.contains('assistant')) {
    last.dataset.final = 'true';
  }
}

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws/voice');

  ws.onopen = () => {
    isConnected = true;
    sendBtn.disabled = false;
    setState('idle');
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case 'status':
        isAssSpeaking = (msg.state === 'speaking' || msg.state === 'thinking');
        setState(msg.state);
        if (msg.state === 'done') finalizeLastAssistant();
        if (msg.state === 'done' || msg.state === 'idle') isAssSpeaking = false;
        break;
      case 'text':
        addMessage('assistant', msg.data);
        break;
      case 'audio': {
        const binary = atob(msg.pcm);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        playPcm(bytes, msg.sr);
        break;
      }
      case 'error':
        addMessage('assistant', '[Error: ' + msg.data + ']');
        setState('idle');
        break;
    }
  };

  ws.onclose = () => {
    isConnected = false;
    sendBtn.disabled = true;
    setState('idle');
    statusLabel.textContent = 'Disconnected';
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => ws.close();
}

function playPcm(pcmData, sampleRate) {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const numFrames = Math.floor(pcmData.length / 2);
  const buffer = audioCtx.createBuffer(1, numFrames, sampleRate);
  const channel = buffer.getChannelData(0);
  const view = new Int16Array(pcmData.buffer, pcmData.byteOffset, numFrames);
  for (let i = 0; i < numFrames; i++) channel[i] = view[i] / 32768;
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();
}

function sendText(text) {
  if (!text.trim() || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMessage('user', text.trim());
  ws.send(JSON.stringify({ type: 'text', data: text.trim() }));
  textInput.value = '';
}

function startPtt() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert('Speech recognition not supported.'); return; }
  pttRecognition = new SR();
  pttRecognition.lang = 'en-US';
  pttRecognition.continuous = false;
  pttRecognition.interimResults = false;
  pttBtn.textContent = 'Listening...';
  pttBtn.style.background = '#22c55e';
  setState('listening');
  pttRecognition.onresult = (e) => {
    sendText(e.results[0][0].transcript);
    pttDone();
  };
  pttRecognition.onerror = pttDone;
  pttRecognition.onend = pttDone;
  pttRecognition.start();
}
function pttDone() {
  pttBtn.textContent = 'Push-to-Talk';
  pttBtn.style.background = '#dc2626';
  setState('idle');
}
pttBtn.addEventListener('click', startPtt);

sendBtn.addEventListener('click', () => sendText(textInput.value));
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendText(textInput.value);
});

clearBtn.addEventListener('click', () => {
  chat.innerHTML = '';
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'cancel' }));
  }
});

connectWebSocket();
</script>
</body>
</html>"""


@app.get("/voice-pipeline", response_class=HTMLResponse)
async def voice_pipeline_page():
    return HTMLResponse(PIPELINE_HTML)


def run_dual_server():
    import threading
    import uvicorn

    def run_piper():
        ssl_kwargs = {}
        if SSL_CERTFILE and SSL_KEYFILE:
            ssl_kwargs["ssl_certfile"] = SSL_CERTFILE
            ssl_kwargs["ssl_keyfile"] = SSL_KEYFILE
        uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=300, **ssl_kwargs)

    piper_thread = threading.Thread(target=run_piper, daemon=True)
    piper_thread.start()

    import streamlit.web.cli as stcli
    import sys

    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
    ]
    stcli.main()


if __name__ == "__main__":
    import uvicorn

    ssl_kwargs = {}
    if SSL_CERTFILE and SSL_KEYFILE:
        ssl_kwargs["ssl_certfile"] = SSL_CERTFILE
        ssl_kwargs["ssl_keyfile"] = SSL_KEYFILE
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=300, **ssl_kwargs)
