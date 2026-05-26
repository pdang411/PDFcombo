"""Shared TTS/text utilities for PiperTTS-based apps.

This module keeps Markdown cleanup and PCM/WAV conversion in one place so
the PDF reader, tutor, and Piper agent all speak text consistently.
"""

import base64
import re
import struct


def clean_text_for_tts(text: str) -> str:
    """Strip markdown, emoji, and formatting for clean TTS synthesis."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'(?m)^#{1,6}\s+', '', text)
    text = re.sub(r'(?m)^>\s+', '', text)
    text = re.sub(r'(?m)^[\*\-\+]\s+', '', text)
    text = re.sub(r'(?m)^\d+\.\s+', '', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'[\u2600-\u27BF\U0001F000-\U0001FFFF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050) -> bytes:
    """Wrap raw PCM int16 data in a WAV container."""
    data_size = len(pcm_data)
    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 36 + data_size)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)
    struct.pack_into("<H", header, 20, 1)
    struct.pack_into("<H", header, 22, 1)
    struct.pack_into("<I", header, 24, sample_rate)
    struct.pack_into("<I", header, 28, sample_rate * 2)
    struct.pack_into("<H", header, 32, 2)
    struct.pack_into("<H", header, 34, 16)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, data_size)
    return bytes(header) + pcm_data


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (.!?)"""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def wav_to_base64(wav_bytes: bytes) -> str:
    """Encode WAV bytes to a single-quote-safe base64 string for st.html injection."""
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return b64.replace("'", "\\'")


def pcm_to_wav_base64(pcm_data: bytes, sample_rate: int = 22050) -> str:
    """PCM to WAV to base64 in one call."""
    return wav_to_base64(pcm_to_wav(pcm_data, sample_rate))
