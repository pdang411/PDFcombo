

# --- Imports ---
import base64
import html
import os
import re
import struct
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import requests
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
from pydub.utils import which
from pipertalk.tts_utils import clean_text_for_tts, pcm_to_wav

import PyPDF2
import fitz  # PyMuPDF

# --- Optional Dependencies ---
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# --- Configuration ---
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:5000")

TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "10000"))
PDF_RENDER_ZOOM = float(os.getenv("PDF_RENDER_ZOOM", "1.6"))
PDF_READ_CHUNK_SIZE = int(os.getenv("PDF_READ_CHUNK_SIZE", str(1024 * 1024)))


# --- Piper TTS HTTP Client ---
class PiperHTTPClient:
    """HTTP client for Piper TTS server. Synthesizes text to WAV audio
    via the /synthesize endpoint. Handles PCM-to-WAV conversion internally."""
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=4, max_retries=1
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def synthesize(
        self,
        text: str,
        voice: str = None,
        speaker_id: int = None,
        length_scale: float = None,
        noise_scale: float = None,
        noise_w_scale: float = None,
    ) -> bytes:
        url = f"{self.base_url}/synthesize"
        payload = {"text": text, "response_format": "pcm"}
        if voice:
            payload["voice"] = voice
        if speaker_id is not None:
            payload["speaker_id"] = speaker_id
        if length_scale is not None:
            payload["length_scale"] = length_scale
        if noise_scale is not None:
            payload["noise_scale"] = noise_scale
        if noise_w_scale is not None:
            payload["noise_w_scale"] = noise_w_scale
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        sr = int(resp.headers.get("X-Piper-Sample-Rate", 22050))
        return pcm_to_wav(resp.content, sr)


# --- Cached TTS Resource & Voice Fetching ---
@st.cache_resource
def get_piper_client():
    """Returns cached PiperHTTPClient singleton for the session.
    Creates one if not yet initialized."""
    try:
        return PiperHTTPClient(TTS_BASE_URL, timeout=60)
    except Exception as e:
        print(f"Failed to initialize Piper HTTP client: {e}")
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_voices() -> dict:
    """Fetches available voices from Piper TTS /voices endpoint.
    Returns dict of voice_name -> config, or empty dict on failure.
    Cached for 30 seconds."""
    try:
        r = requests.get(f"{TTS_BASE_URL}/voices", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


@st.cache_data(max_entries=32, show_spinner=False)
def speak(text: str):
    """Cached wrapper: synthesizes text via PiperHTTPClient.
    Returns WAV bytes or None on failure."""
    client = get_piper_client()
    if not client:
        print("Piper TTS not available")
        return

    try:
        return client.synthesize(text)
    except Exception as e:
        print(f"Piper TTS failed: {e}")


def time_now():
    """Returns current time as formatted string (hh:mm AM/PM)."""
    import datetime
    return datetime.datetime.now().strftime("%I:%M %p")


def listen(timeout: float = 5.0) -> str:
    """Stub: audio recording requires browser mic access via UI button.
    Raises RuntimeError with instructions."""
    raise RuntimeError(
        "Audio recording requires browser microphone access. Use the record button in the UI."
    )


# --- PDF File Reading ---
from functools import lru_cache


@st.cache_data(show_spinner=False)
def _read_pdf_bytes(path_or_file):
    """Reads PDF content from file path (str) or uploaded file object.
    Returns raw bytes for further processing."""
    if isinstance(path_or_file, str):
        with open(path_or_file, "rb") as fh:
            return fh.read()
    return path_or_file.read()


# --- OCR Support for scanned PDFs ---
def ocr_image_to_text(image_bytes: bytes) -> str:
    """Runs Tesseract OCR on a single PNG/JPEG image.
    Returns extracted text string, or empty string on failure.
    Requires pytesseract + PIL."""
    if pytesseract is None or Image is None:
        return ""
    try:
        if TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""


def ocr_pdf_page(pdf_bytes: bytes, page_index: int, zoom: float = 2.0) -> str:
    """Renders a single PDF page as PNG via PyMuPDF, then runs OCR.
    Returns extracted text string, or empty string on failure."""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            p = d.load_page(page_index)
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
        finally:
            d.close()
        return ocr_image_to_text(img_bytes)
    except Exception as e:
        print(f"PDF page OCR failed: {e}")
        return ""


def ocr_pdf_file(pdf_bytes: bytes, zoom: float = 2.0) -> str:
    """Runs OCR on every page of a PDF via PyMuPDF rendering + Tesseract.
    Returns all text joined by double newlines, or empty string on failure."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = len(doc)
            if page_count == 0:
                return ""
            results = []
            for i in range(page_count):
                page = doc.load_page(i)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                text = ocr_image_to_text(img_bytes)
                results.append(text)
        finally:
            doc.close()
        return "\n\n".join(results)
    except Exception as e:
        print(f"PDF file OCR failed: {e}")
        return ""


# --- Streamlit App Initialization ---
st.set_page_config(layout="wide")
try:
    st.set_option("server.maxUploadSize", 500)
except Exception:
    pass

st.title("PDF Page-by-Page Audiobook Reader")

st.markdown(
    """<style>
    img { max-width: 100% !important; height: auto !important; }
    </style>""",
    unsafe_allow_html=True,
)


# --- FFmpeg Setup ---
def _set_ffmpeg():
    """Finds ffmpeg binary for pydub AudioSegment.
    Checks imageio_ffmpeg, FFMPEG_PATH env var, system PATH,
    and common install locations on Windows."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            AudioSegment.converter = exe
            return
    except Exception:
        pass

    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and os.path.isfile(env_ff):
        AudioSegment.converter = env_ff
        return

    sys_ff = which("ffmpeg") or which("avconv")
    if sys_ff:
        AudioSegment.converter = sys_ff
        return

    candidates = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.getcwd(), ".venv", "Scripts", "ffmpeg.exe"),
        os.path.join(os.getcwd(), ".venv", "Library", "bin", "ffmpeg.exe"),
    ]
    for c in candidates:
        try:
            if os.path.isfile(c):
                AudioSegment.converter = c
                return
        except Exception:
            continue
    return


_set_ffmpeg()

_conv = getattr(AudioSegment, "converter", None)
if not (_conv and os.path.isfile(_conv)) and not (which("ffmpeg") or which("avconv")):
    st.sidebar.warning(
        "ffmpeg/avconv not found. Audio functions may fail. "
        "Install ffmpeg and add to PATH, or set FFMPEG_PATH to full ffmpeg.exe path."
    )

if pytesseract is None:
    st.sidebar.warning(
        "Tesseract OCR not installed. For scanned PDFs, install Tesseract and add to PATH. "
        "Download: https://github.com/UB-Mannheim/tesseract/wiki"
    )
elif TESSERACT_PATH and not os.path.isfile(TESSERACT_PATH):
    st.sidebar.warning(
        f"Tesseract OCR not found at: {TESSERACT_PATH}. "
        "Set TESSERACT_PATH environment variable to the tesseract.exe path."
    )


# --- PDF Extraction & Rendering ---
@st.cache_data(show_spinner=False)
def load_pdf_texts(pdf_bytes: bytes):
    """Extracts text from every page of a PDF using PyMuPDF (primary)
    with PyPDF2 fallback. Returns list of strings, one per page."""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        try:
            for i in range(d.page_count):
                p = d.load_page(i)
                texts.append(p.get_text("text") or "")
        finally:
            d.close()
        return texts
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return texts
        except Exception as e2:
            print(f"PyPDF2 also failed: {e2}")
            return [""]


@st.cache_data(show_spinner=False)
def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Returns total number of pages in PDF via PyMuPDF with PyPDF2 fallback."""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return d.page_count
        finally:
            d.close()
    except Exception as e:
        print(f"PyMuPDF page count failed: {e}")
        try:
            return len(PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages)
        except Exception:
            return 0


@st.cache_data(show_spinner=False)
def get_pdf_bookmarks(pdf_bytes: bytes) -> list:
    """Extracts PDF table of contents / bookmarks via PyMuPDF.
    Returns list of dicts: {'title': str, 'page': int (0-based)}."""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            toc = d.get_toc()
            bookmarks = []
            for item in toc:
                level, title, page_num = item[:3]
                bookmarks.append({"title": title.strip(), "page": page_num - 1})
            return bookmarks
        finally:
            d.close()
    except Exception as e:
        print(f"PDF bookmark extraction failed: {e}")
        return []


@st.cache_data(show_spinner=False)
def get_pdf_page_text(pdf_bytes: bytes, page_index: int, use_ocr: bool = False) -> str:
    """Extracts text from a single PDF page via PyMuPDF (primary)
    with PyPDF2 fallback. Optionally runs OCR if use_ocr=True."""
    text = ""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            p = d.load_page(page_index)
            text = p.get_text("text") or ""
        finally:
            d.close()
    except Exception as e:
        print(f"PyMuPDF page text failed: {e}")
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            if 0 <= page_index < len(reader.pages):
                text = reader.pages[page_index].extract_text() or ""
        except Exception:
            pass

    if use_ocr and pytesseract is not None:
        ocr_text = ocr_pdf_page(pdf_bytes, page_index, zoom=2.0)
        if ocr_text.strip():
            text = ocr_text

    return text


def get_current_page_text(pdf_bytes: bytes, use_ocr: bool = False) -> str:
    """Returns text for the currently selected page only."""
    page_index = int(st.session_state.get("page_index", 0))
    total_pages = get_pdf_page_count(pdf_bytes)
    if total_pages <= 0:
        return ""
    page_index = max(0, min(page_index, total_pages - 1))
    st.session_state["page_index"] = page_index
    return get_pdf_page_text(pdf_bytes, page_index, use_ocr=use_ocr) or ""


@st.cache_data(show_spinner=False)
def cached_page_image(pdf_bytes: bytes, page_index: int, zoom: float = 1.6):
    """Renders a single PDF page as PNG image via PyMuPDF.
    Returns PNG bytes or None on failure. Results cached."""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            p = d.load_page(page_index)
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        finally:
            d.close()
    except Exception as e:
        print(f"PDF page image rendering failed: {e}")
        return None


# --- TTS Audio Pipeline ---
def _split_text(text: str, max_len: int) -> list[str]:
    """Splits text into chunks at sentence boundaries (period+space),
    then newlines, then spaces, finally hard cut at max_len.
    Used to stay under TTS_MAX_CHARS per request."""
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind(". ", 0, max_len)
        if cut == -1:
            cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = text.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        part = text[:cut].strip()
        if part:
            parts.append(part)
        text = text[cut:].lstrip()
    return parts


@st.cache_data(show_spinner=False, max_entries=20)
def _synthesize_cached(
    clean_text: str,
    voice: str = None,
    speaker_id: int = None,
    length_scale: float = None,
    noise_scale: float = None,
    noise_w_scale: float = None,
) -> bytes:
    """Synthesizes text to WAV via Piper HTTP client, with chunking.
    Splits text longer than TTS_MAX_CHARS, joins PCM segments,
    and re-wraps in a single WAV. Results cached for reuse."""
    if not clean_text or not clean_text.strip():
        return b""

    client = get_piper_client()
    if not client:
        raise RuntimeError("Piper TTS not available")

    def _synth(txt):
        return client.synthesize(
            txt,
            voice=voice,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

    if len(clean_text) <= TTS_MAX_CHARS:
        return _synth(clean_text)

    chunks = [c for c in _split_text(clean_text, TTS_MAX_CHARS) if c.strip()]
    if not chunks:
        return b""

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_synth, chunks))

    pcm_parts = []
    sample_rate = 22050
    for wav_bytes in results:
        if not wav_bytes or len(wav_bytes) <= 44:
            continue
        if sample_rate == 22050:
            sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
        pcm_parts.append(wav_bytes[44:])

    if not pcm_parts:
        return b""

    pcm_data = b"".join(pcm_parts)
    return pcm_to_wav(pcm_data, sample_rate)


@st.cache_data(max_entries=32, show_spinner=False)
def _cached_text_to_audio(
    clean_text: str,
    rate: float,
    voice: str = None,
    speaker_id: int = None,
    length_scale: float = None,
    noise_scale: float = None,
    noise_w_scale: float = None,
):
    """Cached wrapper: synthesizes clean text then applies playback speed.
    Uses pydub to adjust frame rate for speed changes. Returns BytesIO of WAV."""
    audio_bytes = _synthesize_cached(
        clean_text,
        voice=voice,
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )
    if not audio_bytes:
        return None

    if abs(rate - 1.0) < 1e-6:
        return BytesIO(audio_bytes)

    try:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
        audio = audio._spawn(
            audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * rate)}
        )
        audio = audio.set_frame_rate(44100)
        out = BytesIO()
        audio.export(out, format="wav")
        out.seek(0)
        return out
    except Exception:
        return BytesIO(audio_bytes)


def convert_text_to_audio(
    text: str,
    rate: float = 1.0,
    voice: str = None,
    speaker_id: int = None,
    length_scale: float = None,
    noise_scale: float = None,
    noise_w_scale: float = None,
):
    """Public entry point: cleans text, applies voice settings,
    synthesizes to audio with optional speed adjustment.
    Returns BytesIO of WAV data or None on failure."""
    if not text or not str(text).strip():
        return None
    clean_text = clean_text_for_tts(str(text))
    try:
        r = float(rate)
    except Exception:
        r = 1.0
    return _cached_text_to_audio(
        clean_text, r,
        voice=voice,
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )


# --- Voice Settings Helper ---
def _voice_kwargs() -> dict:
    """Collects current voice/speed settings from session_state
    into a kwargs dict for TTS functions."""
    return dict(
        voice=st.session_state.get("tts_voice"),
        speaker_id=st.session_state.get("tts_speaker_id"),
        length_scale=st.session_state.get("tts_length_scale"),
        noise_scale=st.session_state.get("tts_noise_scale"),
        noise_w_scale=st.session_state.get("tts_noise_w"),
    )


# --- Word-Sync Audio Player ---
def build_sync_player_html(
    audio_bytes: bytes, text: str, element_id: str | None = None
):
    """Builds HTML/CSS/JS for a word-level sync audio player.
    Highlights each word as the audio plays, using estimated
    per-word timestamps proportional to character count."""
    if element_id is None:
        element_id = f"syncplayer_{uuid.uuid4().hex}"

    raw = text.replace("\r\n", "\n")
    lines = [ln for ln in raw.split("\n")]
    words = []
    for ln in lines:
        for w in ln.split():
            words.append(w)

    if not words:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return f'<audio controls src="data:audio/wav;base64,{b64}"></audio>'

    total_chars = sum(len(w) for w in words)
    try:
        from pydub import AudioSegment as _AS

        seg = _AS.from_file(BytesIO(audio_bytes), format="wav")
        total_ms = len(seg)
    except Exception:
        total_ms = max(1000, int(total_chars * 30))

    per_word_ms = [int(len(w) / total_chars * total_ms) for w in words]
    diff = total_ms - sum(per_word_ms)
    i = 0
    while diff > 0:
        per_word_ms[i % len(per_word_ms)] += 1
        diff -= 1
        i += 1

    timestamps = []
    cum = 0
    for ms in per_word_ms:
        timestamps.append(round(cum / 1000.0, 3))
        cum += ms

    out_lines = []
    widx = 0
    for ln in lines:
        parts = []
        for w in ln.split():
            safe = html.escape(w)
            parts.append(f'<span class="word" data-idx="{widx}">{safe}</span>')
            widx += 1
        out_lines.append(" ".join(parts))
    body_html = "<br/>".join(out_lines)

    b64 = base64.b64encode(audio_bytes).decode("ascii")

    js = f'''<style>
.word {{ color: #222; }}
.word.active {{ background: #fffa8b; color: #000; }}
</style>
<div id="{element_id}_container">
    <audio id="{element_id}_audio" controls src="data:audio/wav;base64,{b64}"></audio>
  <div id="{element_id}_text" style="margin-top:12px; font-size:16px; line-height:1.5;">{body_html}</div>
</div>
<script>
(function(){{
  const audio = document.getElementById('{element_id}_audio');
  const words = Array.from(document.querySelectorAll('#{element_id}_text .word'));
  const timestamps = {timestamps};
  function highlight(idx){{
    words.forEach((w,i)=>{{w.classList.toggle('active', i===idx);}});
  }}
  audio.addEventListener('timeupdate', ()=>{{
    const t = audio.currentTime;
    let idx = 0;
    for(let i=0;i<timestamps.length;i++){{ if(t >= timestamps[i]) idx = i; else break; }}
    highlight(idx);
  }});
}})();
</script>'''

    return js


# --- Voice Command Parser ---
def parse_voice_command(cmd: str) -> dict:
    """Parses a natural-language voice command for PDF navigation.
    Recognizes: page N, chapter N, next/prev chapter, next/prev page,
    read, speed N, go to N, chapter by name.
    Returns dict with 'type' and optional 'value'/'read_after'/'text'."""
    cmd = cmd.lower().strip()

    if cmd.startswith("pdf "):
        cmd = cmd[4:].strip()
    elif cmd.startswith("pdf"):
        cmd = cmd[3:].strip()

    if not cmd:
        return {"type": "unknown", "text": ""}

    _nums = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40",
        "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
        "ninety": "90",
    }
    for word, digit in _nums.items():
        cmd = re.sub(r"\b" + word + r"\b", digit, cmd)

    has_read = bool(re.search(r"read|play|speak|repeat|reread|start|narrate|listen", cmd))

    if (m := re.search(r"(?:go\s+)?chapter\s*(\d+)", cmd)):
        return {"type": "chapter_num", "value": int(m.group(1))}

    if re.search(r"(?:next|forward)\s+chapter", cmd):
        return {"type": "next_chapter"}

    if re.search(r"(?:prev|previous|back)\s+chapter", cmd):
        return {"type": "prev_chapter"}

    if "chapter" in cmd:
        return {"type": "chapter_name", "text": cmd}

    if (m := re.search(r"page\s*(\d+)", cmd)):
        rv = {"type": "page", "value": int(m.group(1))}
        if has_read:
            rv["read_after"] = True
        return rv

    if (m := re.search(r"go\s+to\s+(\d+)", cmd)):
        rv = {"type": "page", "value": int(m.group(1))}
        if has_read:
            rv["read_after"] = True
        return rv

    if (m := re.search(r"(?:^|\s)(\d+)$", cmd)):
        rv = {"type": "page", "value": int(m.group(1))}
        if has_read:
            rv["read_after"] = True
        return rv

    if re.search(r"next|forward|turn|flip", cmd):
        return {"type": "next"}

    if re.search(r"prev|previous|back", cmd):
        return {"type": "prev"}

    if (m := re.search(r"speed\s*([\d.]+)", cmd)):
        return {"type": "speed", "value": float(m.group(1))}

    if has_read:
        return {"type": "read"}

    return {"type": "unknown", "text": cmd}


# --- Voice Command Executor ---
def _read_current_page(pdf_bytes=None, total_pages=None, enable_ocr=False, rate=1.0, sync_highlight=False):
    """Generates audio for the current page, plays it in sidebar,
    and optionally shows a word-highlight sync player."""
    if st.session_state.get("page_index", -1) < 0:
        st.sidebar.warning("No page selected.")
        return
    page_text = (get_pdf_page_text(pdf_bytes, st.session_state["page_index"], use_ocr=enable_ocr) or "")
    if not page_text.strip():
        st.sidebar.info("No extractable text on this page (try enabling OCR).")
        return
    with st.spinner("Generating audio for current page..."):
        try:
            buf = convert_text_to_audio(page_text, st.session_state.get("reading_rate", rate), **_voice_kwargs())
            if not buf:
                st.sidebar.warning("No text found on this page to generate audio.")
                return
            st.session_state["last_audio_bytes"] = buf.getvalue()
            st.sidebar.audio(st.session_state["last_audio_bytes"], format="audio/wav", autoplay=True)
            if sync_highlight:
                try:
                    components.html(build_sync_player_html(st.session_state["last_audio_bytes"], page_text), height=260)
                except Exception:
                    pass
        except Exception as e:
            st.sidebar.error(f"Audio generation failed: {e}")


def exec_voice_command(cmd: str, pdf_bytes, total_pages, enable_ocr, rate, sync_highlight):
    """Executes a parsed voice command: navigates pages/chapters,
    adjusts speed, reads aloud, or sends unknown commands to LLM."""
    parsed = parse_voice_command(cmd)
    t = parsed["type"]

    if t == "next":
        if st.session_state["page_index"] < total_pages - 1:
            st.session_state["page_index"] += 1

    elif t == "prev":
        if st.session_state["page_index"] > 0:
            st.session_state["page_index"] -= 1

    elif t == "page":
        p = parsed["value"] - 1
        if 0 <= p < total_pages:
            st.session_state["page_index"] = p
        if parsed.get("read_after"):
            _read_current_page(pdf_bytes, total_pages, enable_ocr, rate, sync_highlight)

    elif t == "chapter_num":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            ch = parsed["value"] - 1
            if 0 <= ch < len(bookmarks):
                st.session_state["page_index"] = bookmarks[ch]["page"]
                st.sidebar.success(f"Went to chapter {ch+1}: {bookmarks[ch]['title']}")
            else:
                st.sidebar.info(f"Chapter {parsed['value']} not found. Max: {len(bookmarks)}")
        else:
            st.sidebar.info("No chapters in this PDF")

    elif t == "next_chapter":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            for bm in bookmarks:
                if bm["page"] > st.session_state["page_index"]:
                    st.session_state["page_index"] = bm["page"]
                    st.sidebar.success(f"Next chapter: {bm['title']}")
                    break

    elif t == "prev_chapter":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            for bm in reversed(bookmarks):
                if bm["page"] < st.session_state["page_index"]:
                    st.session_state["page_index"] = bm["page"]
                    st.sidebar.success(f"Previous chapter: {bm['title']}")
                    break

    elif t == "chapter_name":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            for bm in bookmarks:
                if bm["title"].lower() in parsed.get("text", ""):
                    st.session_state["page_index"] = bm["page"]
                    st.sidebar.success(f"Went to: {bm['title']}")
                    break
            else:
                names = ", ".join(b['title'][:30] for b in bookmarks[:5])
                st.sidebar.info(f"Chapters: {names}")
        else:
            st.sidebar.info("No chapters in this PDF")

    elif t == "speed":
        st.session_state["reading_rate"] = parsed["value"]
        st.sidebar.success(f"Reading speed set to {parsed['value']}")

    elif t == "read":
        _read_current_page(pdf_bytes, total_pages, enable_ocr, rate, sync_highlight)

    elif t == "unknown":
        st.sidebar.info(f"Unrecognised command: '{cmd}'")


# --- UI: Sidebar uploader / speed controls ---
use_local_file = st.sidebar.checkbox(
    "Load PDF from server",
    help="Use a PDF already present in the app folder instead of uploading one.",
)
st.sidebar.caption("Server PDFs are usually faster than browser uploads.")
local_pdf_choice = None
if use_local_file:
    import glob

    pdf_files = glob.glob(os.path.join(os.getcwd(), "*.pdf"))
    if not pdf_files:
        st.sidebar.info(
            "No PDF files found in project root. Place your PDF in the project folder."
        )
    else:
        names = [os.path.basename(p) for p in pdf_files]
        sel = st.sidebar.selectbox("Choose PDF from server", names)
        local_pdf_choice = os.path.join(os.getcwd(), sel)
        st.sidebar.caption(f"Selected: {local_pdf_choice}")

uploaded_pdf = (
    None if use_local_file else st.sidebar.file_uploader("Upload PDF", type=["pdf"])
)

sidebar_rate = st.sidebar.slider(
    "Reading speed (1.0 = normal)", 0.5, 2.0, 1.0, 0.1, key="sidebar_rate"
)
st.session_state["reading_rate"] = float(
    st.session_state.get("sidebar_rate", sidebar_rate)
)
rate = st.session_state.get("reading_rate", float(sidebar_rate))
st.sidebar.write(f"Current speed: {rate}")

# --- Voice Settings ---
st.sidebar.divider()
st.sidebar.subheader("Voice Settings")
_voices_data = _fetch_voices()
_vnames = list(_voices_data.keys())
if _vnames:
    _sel_voice = st.sidebar.selectbox("Voice model", _vnames, key="tts_voice_name")
    _vcfg = _voices_data.get(_sel_voice, {})
    _num_spk = _vcfg.get("num_speakers", 1)
    if _num_spk and int(_num_spk) > 1:
        _spk_map = _vcfg.get("speaker_id_map", {})
        if _spk_map:
            _spk_name = st.sidebar.selectbox(
                "Speaker", list(_spk_map.keys()), key="tts_spk_name"
            )
            _spk_id = int(_spk_map[_spk_name])
        else:
            _spk_id = int(
                st.sidebar.number_input(
                    "Speaker ID", 0, max(0, int(_num_spk) - 1), 0, key="tts_spk_num"
                )
            )
    else:
        _spk_id = None
    st.session_state["tts_voice"] = _sel_voice
    st.session_state["tts_speaker_id"] = _spk_id
else:
    st.sidebar.caption("No voices found on TTS server - using server default.")
    st.session_state["tts_voice"] = None
    st.session_state["tts_speaker_id"] = None

st.sidebar.slider(
    "Length scale (pace)", 0.5, 2.0, 1.0, 0.05,
    key="tts_length_scale",
    help="Lower = faster speech. 1.0 = model default.",
)
st.sidebar.slider(
    "Noise scale (expressiveness)", 0.0, 1.0, 0.667, 0.05,
    key="tts_noise_scale",
    help="Higher = more pitch variation in voice.",
)
st.sidebar.slider(
    "Noise_w (phoneme timing)", 0.0, 1.0, 0.8, 0.05,
    key="tts_noise_w",
    help="Higher = more variation in phoneme duration.",
)

sync_highlight = st.sidebar.checkbox(
    "Enable sync highlighting (word-by-word)", value=True
)

enable_ocr = False
if uploaded_pdf or local_pdf_choice:
    if local_pdf_choice:
        pdf_bytes = _read_pdf_bytes(local_pdf_choice)
    else:
        pdf_bytes = _read_pdf_bytes(uploaded_pdf)

    if not pdf_bytes:
        st.sidebar.error("No PDF data received. Please try uploading a valid PDF file.")
        st.info("Upload a PDF in the sidebar to begin.")
        st.stop()

    total_pages = get_pdf_page_count(pdf_bytes)

    st.sidebar.write(f"Pages: {total_pages}")

    show_diag = st.sidebar.checkbox("Show diagnostics")
    if show_diag:
        st.sidebar.write(f"pdf_bytes size: {len(pdf_bytes)} bytes")
        try:
            d_check = fitz.open(stream=pdf_bytes, filetype="pdf")
            raw_count = d_check.page_count
            d_check.close()
        except Exception:
            try:
                raw_count = len(PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages)
            except Exception:
                raw_count = "(error)"
        st.sidebar.write(f"Reader page count: {raw_count}")
        current_page_number = st.session_state.get("page_index", 0) + 1
        current_page_text = get_current_page_text(pdf_bytes, use_ocr=enable_ocr)
        st.sidebar.write("Current page diagnostics:")
        st.sidebar.write(
            {
                "page": current_page_number,
                "chars": len(current_page_text),
            }
        )
        st.sidebar.write(f"session page_index: {st.session_state.get('page_index')}")

    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0
    if st.session_state["page_index"] >= total_pages:
        st.session_state["page_index"] = max(0, total_pages - 1)

    audio_container = st.empty()

    st.sidebar.subheader("Navigation")

    st.sidebar.metric("Current Page", f"{st.session_state['page_index'] + 1} / {total_pages}")

    page_input = st.sidebar.number_input(
        "Page number",
        min_value=1,
        max_value=max(1, total_pages),
        value=st.session_state["page_index"] + 1,
        step=1,
        key="page_input",
    )
    if page_input - 1 != st.session_state["page_index"]:
        st.session_state["page_index"] = page_input - 1

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if col1.button("Prev", key="nav_prev"):
            if st.session_state["page_index"] > 0:
                st.session_state["page_index"] -= 1
    with col2:
        if col2.button("Next", key="nav_next"):
            if st.session_state["page_index"] < total_pages - 1:
                st.session_state["page_index"] += 1

    colf, coll = st.sidebar.columns([1, 1])
    with colf:
        if colf.button("First", key="nav_first"):
            st.session_state["page_index"] = 0
    with coll:
        if coll.button("Last", key="nav_last"):
            st.session_state["page_index"] = max(0, total_pages - 1)

    enable_ocr = st.sidebar.toggle(
        "Enable OCR for scanned PDFs",
        value=False,
        help="Turn on to extract text from scanned/image PDFs",
    )

    # --- Voice Assistant (hidden form pattern) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Voice Assistant")

    st.markdown(
        """<style>[data-testid="stForm"]{position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;overflow:hidden;}</style>""",
        unsafe_allow_html=True,
    )
    with st.form("pdf_vf"):
        st.text_input("", key="pdf_voice_cmd", label_visibility="collapsed")
        st.form_submit_button("Send", use_container_width=True)

    _vc_text = st.session_state.get("pdf_voice_cmd", "").strip()
    _vc_last = st.session_state.get("_pdf_last_cmd", "")
    if _vc_text and _vc_text != _vc_last:
        st.session_state["_pdf_last_cmd"] = _vc_text
        _vc_cmd = _vc_text.lower()
        if not _vc_cmd.startswith("pdf"):
            _vc_cmd = "pdf " + _vc_cmd
        st.sidebar.info(f"Voice: '{_vc_cmd}'")
        exec_voice_command(_vc_cmd, pdf_bytes, total_pages, enable_ocr, rate, sync_highlight)

    # Hands-free voice mode
    if "_pdf_vm" not in st.session_state:
        st.session_state["_pdf_vm"] = False
    if "_pdf_toggle" not in st.session_state:
        st.session_state["_pdf_toggle"] = False

    if st.session_state.pop("_stop_pdf_vm", False):
        st.session_state["_pdf_toggle"] = False

    st.sidebar.toggle(
        "Hands-Free Voice Mode",
        key="_pdf_toggle",
        help="Continuous listening for voice commands (no button needed)",
    )
    st.session_state["_pdf_vm"] = st.session_state["_pdf_toggle"]

    if st.session_state["_pdf_vm"]:
        st.sidebar.html(
            """
<div id="pdf-vm-status" style="font-size:12px;color:#4ade80;margin-bottom:4px;">Initializing...</div>
<script>
(function() {
    if (window.__pdfRec) {
        var el = document.getElementById('pdf-vm-status');
        if (el) el.textContent = 'Listening...';
        return;
    }
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { var el = document.getElementById('pdf-vm-status'); if (el) el.textContent = 'Not supported'; return; }
    var rec, running = true;
    var upd = function(t) { var el = document.getElementById('pdf-vm-status'); if (el) el.textContent = t; };
    var sub = function(text) {
        text = (text||'').trim();
        if (!text) return;
        upd('Cmd: ' + text.slice(0,40));
        var form = document.querySelector('[data-testid="stForm"]');
        if (!form) return;
        var input = form.querySelector('input');
        if (!input) return;
        var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        s.call(input, text);
        input.dispatchEvent(new Event('input', {bubbles:true}));
        setTimeout(function() { var btn = form.querySelector('button'); if (btn && !btn.disabled) btn.click(); }, 100);
    };
    function start() {
        if (!running) return;
        rec = new SR();
        window.__pdfRec = rec;
        rec.continuous = true;
        rec.interimResults = false;
        rec.lang = 'en-US';
        rec.onresult = function(e) {
            for (var i = e.resultIndex; i < e.results.length; i++) {
                if (e.results[i].isFinal) { sub(e.results[i][0].transcript); }
            }
        };
        rec.onerror = function() { if (running) setTimeout(start, 1000); };
        rec.onend = function() { if (running) setTimeout(start, 500); };
        try { rec.start(); upd('Listening...'); } catch(e) {
            upd('Click page to enable mic...');
            var f = function() { document.removeEventListener('click', f); setTimeout(start, 200); };
            document.addEventListener('click', f);
        }
    }
    if (!window.__pdfRec) setTimeout(start, 500);
    window.addEventListener('beforeunload', function(){running=false; if(rec) try{rec.stop();}catch(e){}});
})();
</script>
""",
            unsafe_allow_javascript=True,
        )
        if st.sidebar.button("Stop Voice Mode"):
            st.session_state["_stop_pdf_vm"] = True
            st.session_state["_pdf_vm"] = False
            st.rerun()
    else:
        st.sidebar.caption("'next page', 'back', 'page 5', 'read page 3', 'chapter 1', 'read', 'speed 1.5'")
        st.sidebar.html(
            """
            <style>
                #voice-btn {
                    padding: 8px 16px; font-size: 16px; cursor: pointer;
                    border-radius: 4px; border: 1px solid #ccc; background: #4CAF50; color: #fff;
                    width: 100%;
                }
                #voice-btn:hover { background: #45a049; }
                #voice-btn:disabled { opacity: 0.6; cursor: not-allowed; }
                #voice-status { margin-top: 5px; font-size: 13px; min-height: 20px; }
            </style>
            <button id="voice-btn">Start Voice Command</button>
            <div id="voice-status"></div>
            <script>
            (function() {
                var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                var btn = document.getElementById('voice-btn');
                var status = document.getElementById('voice-status');
                if (!SR) { status.textContent = 'Not supported. Try Chrome.'; btn.disabled = true; return; }
                btn.addEventListener('click', function() {
                    if (btn.disabled) return;
                    btn.disabled = true;
                    btn.textContent = 'Listening...';
                    status.textContent = 'Speak now...';
                    var r = new SR();
                    r.lang = 'en-US';
                    r.continuous = false;
                    r.interimResults = false;
                    var timeout = setTimeout(function() {
                        try { r.stop(); } catch(e) {}
                        status.textContent = 'Timed out. Try again.';
                        btn.disabled = false;
                        btn.textContent = 'Start Voice Command';
                    }, 10000);
                    r.onresult = function(e) {
                        clearTimeout(timeout);
                        var text = e.results[0][0].transcript;
                        status.textContent = 'Got: ' + text;
                        var form = document.querySelector('[data-testid="stForm"]');
                        if (form) {
                            var inp = form.querySelector('input');
                            if (inp) {
                                var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                                setter.call(inp, text);
                                inp.dispatchEvent(new Event('input', {bubbles:true}));
                                setTimeout(function() {
                                    var btn2 = form.querySelector('button');
                                    if (btn2 && !btn2.disabled) btn2.click();
                                }, 100);
                            }
                        }
                        btn.disabled = false;
                        btn.textContent = 'Start Voice Command';
                    };
                    r.onerror = function(e) {
                        clearTimeout(timeout);
                        status.textContent = 'Error: ' + e.error;
                        btn.disabled = false;
                        btn.textContent = 'Start Voice Command';
                    };
                    r.onend = function() {
                        clearTimeout(timeout);
                        btn.disabled = false;
                        btn.textContent = 'Start Voice Command';
                    };
                    try { r.start(); } catch(err) {
                        status.textContent = 'Error: ' + err.message;
                        btn.disabled = false;
                        btn.textContent = 'Start Voice Command';
                        clearTimeout(timeout);
                    }
                });
            })();
            </script>
            """,
            unsafe_allow_javascript=True,
        )

    # Legacy query-param fallback
    recognized = st.query_params.get("voice")
    if recognized and isinstance(recognized, str) and recognized.strip():
        _qp_cmd = recognized.strip().lower()
        if not _qp_cmd.startswith("pdf"):
            _qp_cmd = "pdf " + _qp_cmd
        st.sidebar.info(f"Voice: '{_qp_cmd}'")
        st.query_params.clear()
        exec_voice_command(_qp_cmd, pdf_bytes, total_pages, enable_ocr, rate, sync_highlight)

    st.sidebar.subheader("Select Page")

    page_index = st.session_state["page_index"]
    page_image = None
    try:
        page_image = cached_page_image(pdf_bytes, page_index, zoom=1.6)
    except Exception:
        page_image = None

    if page_image is not None:
        st.image(page_image, caption=f"Page {page_index + 1}")
    else:
        st.info("Page image unavailable (render failed). Showing text-only view.")

    page_text = get_current_page_text(pdf_bytes, use_ocr=enable_ocr)
    escaped = html.escape(page_text)
    html_content = f"""
    <div style='width:100%; max-width:100%; height:600px; overflow-y:auto; border:1px solid #ddd; padding:12px; background:#fff; box-sizing:border-box;'>
      <pre style='white-space:pre-wrap; font-family:inherit; font-size:16px; line-height:1.6; margin:0; word-wrap:break-word;'>
{escaped}
      </pre>
    </div>
    """
    components.html(html_content, height=600)

    if "last_audio_bytes" in st.session_state and st.session_state["last_audio_bytes"]:
        try:
            st.audio(st.session_state["last_audio_bytes"], format="audio/wav")
        except Exception:
            pass

    audio_buffer = None
    if st.sidebar.button("Start Reading", key="start_reading"):
        with st.spinner("Generating audio for this page..."):
            try:
                audio_buffer = convert_text_to_audio(page_text, rate, **_voice_kwargs())
            except Exception as e:
                st.sidebar.error(f"Audio generation failed: {e}")
                audio_buffer = None

    if audio_buffer is not None:
        audio_container.audio(audio_buffer, format="audio/wav", autoplay=True)
        try:
            if sync_highlight:
                html_player = build_sync_player_html(audio_buffer.getvalue(), page_text)
                components.html(html_player, height=260)
        except Exception:
            pass
        st.sidebar.download_button(
            "Download This Page as WAV",
            data=audio_buffer,
            file_name=f"page_{page_index + 1}.wav",
            mime="audio/wav",
            key=f"download_play_{page_index}_{uuid.uuid4().hex}",
        )

    if st.sidebar.button("Generate (no play)", key="gen_no_play"):
        with st.spinner("Generating audio for this page (no play)..."):
            try:
                audio_buffer = convert_text_to_audio(page_text, rate, **_voice_kwargs())
            except Exception as e:
                st.sidebar.error(f"Audio generation failed: {e}")
                audio_buffer = None

    if audio_buffer is not None:
        st.sidebar.download_button(
            "Download This Page as WAV",
            data=audio_buffer,
            file_name=f"page_{page_index + 1}.wav",
            mime="audio/wav",
            key=f"download_noplay_{page_index}_{uuid.uuid4().hex}",
        )
else:
    st.info("Upload a PDF in the sidebar to begin.")
