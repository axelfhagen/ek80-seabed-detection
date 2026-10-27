"""
Core parsing functions for Simrad .raw files.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple


def read_raw_file(path: str) -> bytes:
    """Load a .raw file as bytes."""
    with open(path, "rb") as f:
        return f.read()


def find_text_strings(data: bytes, min_length: int = 4) -> List[Tuple[int, str]]:
    """
    Find ASCII text strings in binary data.
    Returns list of (offset, string) tuples.
    """
    strings = []
    current = []
    start = 0

    for i, byte in enumerate(data):
        if 32 <= byte <= 126:  # Printable ASCII
            if not current:
                start = i
            current.append(chr(byte))
        else:
            if len(current) >= min_length:
                strings.append((start, "".join(current)))
            current = []

    if len(current) >= min_length:
        strings.append((start, "".join(current)))

    return strings


def find_datagram_types(data: bytes) -> Dict[str, List[int]]:
    """
    Search for common Simrad datagram type identifiers.
    Returns dict mapping type string to list of offsets.
    """
    # Common Simrad datagram types
    known_types = [
        b"CON0",
        b"CON1",  # Configuration
        b"RAW0",
        b"RAW3",  # Raw data
        b"XML0",  # XML configuration
        b"NME0",  # NMEA data
        b"TAG0",  # Annotations
        b"FIL1",  # Filter
        b"MRU0",  # Motion reference unit
    ]

    found = {}
    for dtype in known_types:
        offsets = []
        start = 0
        while True:
            pos = data.find(dtype, start)
            if pos == -1:
                break
            offsets.append(pos)
            start = pos + 1

        if offsets:
            found[dtype.decode("ascii")] = offsets

    return found


def parse_datagram_header(data: bytes, offset: int) -> Dict[str, Any]:
    """
    Parse a datagram header at the given offset.
    Simrad format typically has length before type.

    Returns dict with header info.
    """
    # Try to read length (4 bytes before type identifier)
    if offset >= 4:
        length = int.from_bytes(data[offset - 4 : offset], "little")
    else:
        length = None

    # Read type (4 bytes)
    dtype = data[offset : offset + 4].decode("ascii", errors="replace")

    return {
        "offset": offset,
        "type": dtype,
        "length": length,
    }


def extract_xml_content(data: bytes) -> List[str]:
    """
    Extract XML sections from raw data.
    Many Simrad files contain XML configuration.
    """
    xml_sections = []

    # Find XML0 datagrams
    start = 0
    while True:
        pos = data.find(b"XML0", start)
        if pos == -1:
            break

        # Try to find XML content after the header
        # Look for <?xml or <Configuration or similar
        search_start = pos + 4
        search_end = min(search_start + 10000, len(data))
        section = data[search_start:search_end]

        # Find XML start
        xml_start = section.find(b"<?xml")
        if xml_start == -1:
            xml_start = section.find(b"<")

        if xml_start != -1:
            # Find end of XML (look for closing tag or null bytes)
            xml_data = section[xml_start:]
            end = xml_data.find(b"\x00\x00")
            if end != -1:
                xml_text = xml_data[:end].decode("utf-8", errors="replace")
                xml_sections.append(xml_text)

        start = pos + 1

    return xml_sections


def get_file_info(path: str) -> Dict[str, Any]:
    """
    Get basic info about a .raw file.
    """
    p = Path(path)
    data = read_raw_file(path)

    return {
        "filename": p.name,
        "size_bytes": len(data),
        "size_mb": len(data) / (1024 * 1024),
        "first_bytes": data[:100].hex(),
        "datagram_types": find_datagram_types(data),
    }
