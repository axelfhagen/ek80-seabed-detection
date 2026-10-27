"""GPS/NMEA extraction from Simrad .raw files."""

import math
import struct
from typing import Optional

import pandas as pd

from .raw_parser import find_datagram_types
from .ping_parser import nt_to_datetime


# NMEA sentence prefixes that carry GPS position
GPS_PREFIXES = ("$GPGGA", "$GPRMC", "$GPGLL", "$INGGA", "$GNRMC", "$GNGGA")


def parse_nme0_datagrams(data: bytes) -> list:
    """Extract all NME0 datagrams as (timestamp, sentence) records."""
    offsets = find_datagram_types(data).get("NME0", [])
    records = []
    for off in offsets:
        # 8-byte FILETIME timestamp starts 4 bytes after the 'NME0' tag
        ts_bytes = data[off + 4 : off + 12]
        if len(ts_bytes) < 8:
            continue
        nt_time = struct.unpack("<Q", ts_bytes)[0]
        timestamp = nt_to_datetime(nt_time)

        # NMEA sentence starts 12 bytes after the tag, null-terminated
        sentence_start = off + 12
        null_pos = data.find(b"\x00", sentence_start)
        if null_pos == -1 or null_pos - sentence_start > 512:
            null_pos = sentence_start + 256  # safety cap
        sentence = data[sentence_start:null_pos].decode("ascii", errors="replace").strip()
        if sentence:
            records.append({"timestamp": timestamp, "sentence": sentence})
    return records


def nmea_to_decimal(value: str, direction: str) -> Optional[float]:
    """Convert NMEA DDmm.mmmm + N/S/E/W to decimal degrees."""
    if not value or "." not in value:
        return None
    dot = value.index(".")
    degrees = float(value[: dot - 2])
    minutes = float(value[dot - 2 :])
    decimal = degrees + minutes / 60.0
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal


def parse_gps_sentences(records: list) -> pd.DataFrame:
    """Filter NME0 records for GPS position sentences and decode lat/lon."""
    rows = []
    for r in records:
        s = r["sentence"]
        # Strip checksum (*XX) before splitting
        if "*" in s:
            s = s[: s.index("*")]
        if not any(s.startswith(p) for p in GPS_PREFIXES):
            continue
        fields = s.split(",")
        try:
            msg_type = s[3:6]
            if msg_type == "GGA":
                lat = nmea_to_decimal(fields[2], fields[3])
                lon = nmea_to_decimal(fields[4], fields[5])
            elif msg_type == "GLL":
                lat = nmea_to_decimal(fields[1], fields[2])
                lon = nmea_to_decimal(fields[3], fields[4])
            elif msg_type == "RMC":
                lat = nmea_to_decimal(fields[3], fields[4])
                lon = nmea_to_decimal(fields[5], fields[6])
            else:
                continue
            if lat is not None and lon is not None:
                rows.append(
                    {
                        "timestamp": r["timestamp"],
                        "lat": lat,
                        "lon": lon,
                        "sentence_type": r["sentence"][:6],
                    }
                )
        except (IndexError, ValueError):
            continue
    df = pd.DataFrame(rows)
    if len(df) > 0:
        # De-duplicate: GGA/GLL/RMC fire at the same second — keep one per timestamp
        df = df.drop_duplicates(subset=["timestamp", "lat", "lon"])
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in meters."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
