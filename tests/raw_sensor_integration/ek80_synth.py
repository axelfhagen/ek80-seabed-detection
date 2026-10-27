"""Builders for tiny synthetic Simrad EK80 .raw byte streams.

Framing (Simrad EK80 interface spec): every datagram is
    <uint32 LE length> <4-byte type> <int64 LE NT time> <payload> <uint32 LE length>
where ``length`` counts the type, timestamp and payload bytes (not the two
length fields themselves).
"""

import struct
from datetime import datetime

import numpy as np

# 100 ns ticks between 1601-01-01 and 1970-01-01
NT_EPOCH_OFFSET = 116444736000000000


def nt_time(dt: datetime) -> int:
    """Windows FILETIME for a naive-UTC datetime (computed independently)."""
    epoch = datetime(1601, 1, 1)
    delta = dt - epoch
    return (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10


def datagram(dtype: bytes, timestamp: datetime, payload: bytes) -> bytes:
    assert len(dtype) == 4
    body = dtype + struct.pack("<q", nt_time(timestamp)) + payload
    n = struct.pack("<I", len(body))
    return n + body + n


def xml0(timestamp: datetime, xml_text: str) -> bytes:
    # EK80 XML0 payloads are null-terminated; pad with two nulls so the
    # terminator is unambiguous.
    return datagram(b"XML0", timestamp, xml_text.encode("ascii") + b"\x00\x00")


def nme0(timestamp: datetime, sentence: str) -> bytes:
    return datagram(b"NME0", timestamp, sentence.encode("ascii") + b"\x00")


def raw3(
    timestamp: datetime,
    channel_id: str,
    samples,
    data_type: int,
    sample_offset: int = 0,
    sample_count=None,
) -> bytes:
    """RAW3 datagram.  ``samples`` is serialised verbatim as float32/complex64."""
    samples = np.asarray(samples)
    if sample_count is None:
        sample_count = samples.shape[0]
    chan = channel_id.encode("ascii").ljust(128, b"\x00")
    header = (
        chan
        + struct.pack("<H", data_type)
        + b"\x00\x00"  # spare
        + struct.pack("<i", sample_offset)
        + struct.pack("<i", sample_count)
    )
    return datagram(b"RAW3", timestamp, header + samples.tobytes())
