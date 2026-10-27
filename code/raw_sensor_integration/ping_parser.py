"""
Ping extraction and parsing for Simrad EK80 .raw files.
"""

import struct
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from . import raw_parser


# Windows NT epoch offset (difference between 1601 and 1970 in 100ns units)
NT_EPOCH_OFFSET = 116444736000000000


def nt_to_datetime(nt_time: int) -> datetime:
    """Convert Windows NT timestamp to Python datetime."""
    unix_time = (nt_time - NT_EPOCH_OFFSET) / 10000000
    return datetime.utcfromtimestamp(unix_time)


def decode_data_type(data_type: int) -> Dict[str, Any]:
    """
    Decode RAW3 DataType bitmask.

    Returns dict with:
    - base_type: Power, Angle, PowerAngle, or Complex
    - complex_float32: True if data is complex float32
    - has_angle: True if angle data is included
    """
    base_type = data_type & 0x07
    types = ["Power", "Angle", "PowerAngle", "Complex"]
    base = types[base_type] if base_type < 4 else f"Unknown({base_type})"

    return {
        "base_type": base,
        "complex_float32": bool((data_type >> 3) & 1),
        "has_angle": bool((data_type >> 8) & 1),
        "has_complex": bool((data_type >> 9) & 1),
        "raw": data_type,
    }


class Ping:
    """Represents a single ping from an echosounder."""

    def __init__(
        self,
        channel_id: str,
        timestamp: datetime,
        samples: np.ndarray,
        sample_offset: int,
        data_type_info: Dict[str, Any],
    ):
        self.channel_id = channel_id
        self.timestamp = timestamp
        self.samples = samples  # Complex samples
        self.sample_offset = sample_offset
        self.data_type_info = data_type_info

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def power(self) -> np.ndarray:
        """Get power (magnitude squared) in linear units."""
        return np.abs(self.samples) ** 2

    @property
    def power_db(self) -> np.ndarray:
        """Get power in dB."""
        power = self.power
        # Avoid log(0)
        power = np.maximum(power, 1e-20)
        return 10 * np.log10(power)

    def __repr__(self):
        return (
            f"Ping(channel={self.channel_id}, time={self.timestamp}, samples={self.sample_count})"
        )


def parse_raw3_datagram(data: bytes, offset: int) -> Tuple[Ping, int]:
    """
    Parse a single RAW3 datagram.

    Args:
        data: File bytes
        offset: Offset to start of datagram (at 'RAW3')

    Returns:
        (Ping object, next offset after datagram)
    """
    # Get datagram length
    dgram_len = struct.unpack("<I", data[offset - 4 : offset])[0]

    pos = offset + 4  # Skip 'RAW3'

    # Timestamp (8 bytes)
    nt_time = struct.unpack("<q", data[pos : pos + 8])[0]
    timestamp = nt_to_datetime(nt_time)
    pos += 8

    # Channel ID (128 bytes)
    channel_id = data[pos : pos + 128].split(b"\x00")[0].decode("ascii")
    pos += 128

    # Data type (2 bytes)
    data_type = struct.unpack("<H", data[pos : pos + 2])[0]
    data_type_info = decode_data_type(data_type)
    pos += 2

    # NComplex (1 byte) + Spare (1 byte)
    pos += 2

    # Sample offset (4 bytes)
    sample_offset = struct.unpack("<i", data[pos : pos + 4])[0]
    pos += 4

    # Sample count (4 bytes)
    sample_count = struct.unpack("<i", data[pos : pos + 4])[0]
    pos += 4

    # Calculate header size and data size
    header_size = pos - offset
    data_size = dgram_len - header_size

    # Read sample data
    sample_bytes = data[pos : pos + data_size]

    # Parse samples based on data type
    if data_type_info["complex_float32"]:
        # Complex float32 data
        samples = np.frombuffer(sample_bytes, dtype=np.complex64)

        # If has_complex is set, data includes additional complex samples
        if data_type_info["has_complex"]:
            # Data is interleaved: complex sample + angle data
            bytes_per_sample = data_size // sample_count
            if bytes_per_sample == 24:
                # 8 bytes complex + 8 bytes complex + 8 bytes angle?
                # Extract just the first complex sample for now
                samples = samples[::3][:sample_count]
    else:
        # Power/angle as float32
        samples = np.frombuffer(sample_bytes, dtype=np.float32)
        # Convert to complex for consistency
        samples = samples.astype(np.complex64)

    # Ensure correct sample count
    if len(samples) > sample_count:
        samples = samples[:sample_count]

    ping = Ping(
        channel_id=channel_id,
        timestamp=timestamp,
        samples=samples,
        sample_offset=sample_offset,
        data_type_info=data_type_info,
    )

    # Next offset is after datagram + trailing length field
    next_offset = offset + dgram_len + 4

    return ping, next_offset


def extract_pings(
    path: str, channel_filter: Optional[str] = None, max_pings: Optional[int] = None
) -> List[Ping]:
    """
    Extract all pings from a .raw file.

    Args:
        path: Path to .raw file
        channel_filter: Only extract pings from channels containing this string
        max_pings: Maximum number of pings to extract (None for all)

    Returns:
        List of Ping objects
    """
    data = raw_parser.read_raw_file(path)
    datagram_types = raw_parser.find_datagram_types(data)

    if "RAW3" not in datagram_types:
        return []

    pings = []
    for offset in datagram_types["RAW3"]:
        try:
            ping, _ = parse_raw3_datagram(data, offset)

            # Apply channel filter
            if channel_filter and channel_filter not in ping.channel_id:
                continue

            pings.append(ping)

            if max_pings and len(pings) >= max_pings:
                break

        except Exception:
            # Skip malformed datagrams
            continue

    return pings


def get_ping_matrix(pings: List[Ping], channel_id: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Convert list of pings to a 2D matrix.

    Args:
        pings: List of Ping objects
        channel_id: Channel to extract (must match exactly or be substring)

    Returns:
        (matrix, timestamps) where matrix is [samples x pings]
    """
    # Filter pings by channel
    filtered = [p for p in pings if channel_id in p.channel_id]

    if not filtered:
        return np.array([]), []

    # Get dimensions
    sample_counts = [p.sample_count for p in filtered]
    max_samples = max(sample_counts)
    n_pings = len(filtered)

    # Build matrix
    matrix = np.zeros((max_samples, n_pings), dtype=np.complex64)
    timestamps = []

    for i, ping in enumerate(filtered):
        matrix[: ping.sample_count, i] = ping.samples
        timestamps.append(ping.timestamp)

    return matrix, timestamps


def pings_to_power_image(pings: List[Ping], channel_id: str) -> np.ndarray:
    """
    Convert pings to a power image (dB scale).

    Args:
        pings: List of Ping objects
        channel_id: Channel to extract

    Returns:
        2D array of power in dB [samples x pings]
    """
    matrix, _ = get_ping_matrix(pings, channel_id)

    if matrix.size == 0:
        return np.array([])

    # Convert to power in dB
    power = np.abs(matrix) ** 2
    power = np.maximum(power, 1e-20)  # Avoid log(0)
    power_db = 10 * np.log10(power)

    return power_db
