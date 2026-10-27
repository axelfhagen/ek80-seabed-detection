"""
Metadata extraction utilities for Simrad .raw files.
"""

import re
import struct
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
from . import raw_parser


def extract_metadata(path: str) -> Dict[str, Any]:
    """
    Extract metadata from a .raw file.

    Returns dict with available metadata fields.
    """
    data = raw_parser.read_raw_file(path)

    metadata = {
        "file_info": raw_parser.get_file_info(path),
        "xml_config": {},
        "range_info": {},
    }

    # Try to parse XML configuration
    xml_sections = raw_parser.extract_xml_content(data)
    if xml_sections:
        for i, xml_text in enumerate(xml_sections):
            parsed = parse_xml_config(xml_text)
            if parsed:
                metadata["xml_config"][f"section_{i}"] = parsed

    # Extract range-related parameters
    metadata["range_info"] = extract_range_parameters(metadata["xml_config"])

    return metadata


def parse_xml_config(xml_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse XML configuration text into a dictionary.
    """
    try:
        # Clean up XML text
        xml_text = xml_text.strip()
        if not xml_text.startswith("<?xml") and not xml_text.startswith("<"):
            return None

        root = ET.fromstring(xml_text)
        return xml_element_to_dict(root)
    except ET.ParseError:
        # Try to extract key-value pairs with regex
        return extract_key_values_regex(xml_text)


def xml_element_to_dict(element) -> Dict[str, Any]:
    """
    Convert XML element to dictionary recursively.
    """
    result = {}

    # Add attributes
    if element.attrib:
        result["@attributes"] = element.attrib

    # Add children
    for child in element:
        child_data = xml_element_to_dict(child)

        if child.tag in result:
            # Convert to list if multiple children with same tag
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data

    # Add text content
    if element.text and element.text.strip():
        if result:
            result["#text"] = element.text.strip()
        else:
            return element.text.strip()

    return result


def extract_key_values_regex(text: str) -> Dict[str, str]:
    """
    Extract key-value pairs from text using regex.
    Useful for malformed XML or other structured text.
    """
    result = {}

    # Pattern for <Tag>Value</Tag>
    pattern = r"<(\w+)>([^<]+)</\1>"
    matches = re.findall(pattern, text)

    for key, value in matches:
        result[key] = value.strip()

    return result


def extract_range_parameters(xml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract range-related parameters from parsed XML config.

    Looks for common parameter names related to range/depth.
    """
    range_params = {}

    # Keywords to search for
    keywords = [
        "range",
        "depth",
        "sample",
        "interval",
        "frequency",
        "pulse",
        "power",
        "gain",
        "tvg",
        "absorption",
        "soundvelocity",
        "sound_velocity",
        "soundspeed",
    ]

    def search_dict(d, path=""):
        if isinstance(d, dict):
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key

                # Check if key matches any keyword
                key_lower = key.lower()
                for kw in keywords:
                    if kw in key_lower:
                        range_params[current_path] = value
                        break

                # Recurse
                search_dict(value, current_path)
        elif isinstance(d, list):
            for i, item in enumerate(d):
                search_dict(item, f"{path}[{i}]")

    search_dict(xml_config)

    return range_params


def _get_xml_attr(attrs_bytes: bytes, name: str) -> Optional[str]:
    """Extract a single attribute value from a bytes snippet of XML attributes."""
    m = re.search(rb"(?:^|\s)" + name.encode() + rb'="([^"]+)"', attrs_bytes)
    return m.group(1).decode("ascii") if m else None


def _get_nominal_frequencies(data: bytes) -> Dict[str, int]:
    """
    Extract nominal frequency (Hz) per channel from the static config XML block.

    The config XML has non-self-closing Channel elements that contain a nested
    Transducer element with a Frequency attribute, e.g.:
        <Channel ChannelID="WBT 737172-7 ES38-18|200-18CR_ES" ...>
            <Transducer ... Frequency="38000" ... />
        </Channel>
    This gives the actual nominal frequency regardless of the FM sweep range.
    """
    config_end = data.find(b"</Configuration>")
    if config_end == -1:
        return {}
    config = data[: config_end + 20].decode("ascii", errors="replace")
    freq_map: Dict[str, int] = {}
    pattern = r'ChannelID="([^"]+)"[^>]*>.*?<Transducer\b[^>]*\bFrequency="(\d+)"'
    for m in re.finditer(pattern, config, re.DOTALL):
        channel_id = m.group(1)
        if channel_id not in freq_map:
            freq_map[channel_id] = int(m.group(2))
    return freq_map


def get_channel_settings(path: str) -> List[Dict[str, Any]]:
    """
    Extract channel settings from a .raw file.

    Returns list of dicts with:
    - channel_id: Channel identifier
    - sample_interval: Time between samples (seconds)
    - sound_velocity: Sound speed (m/s)
    - sample_count: Number of samples per ping
    - max_range: Calculated maximum range (meters)
    - frequency: Channel centre frequency (Hz), derived from FrequencyStart/FrequencyEnd
    """
    data = raw_parser.read_raw_file(path)

    # Ping-config Channel elements are self-closing and carry FrequencyStart/FrequencyEnd.
    # Example:
    #   <Channel ChannelID="..." FrequencyStart="35000" FrequencyEnd="38000"
    #            SampleInterval="0.000128" SoundVelocity="1500" ... />
    channel_settings = {}
    ping_channel_pattern = rb"<Channel\b([^>]+)/>"

    for match in re.finditer(ping_channel_pattern, data):
        attrs = match.group(1)
        channel_id = _get_xml_attr(attrs, "ChannelID")
        sample_interval_str = _get_xml_attr(attrs, "SampleInterval")

        if not channel_id or not sample_interval_str:
            continue
        if channel_id in channel_settings:
            continue  # keep first occurrence per channel

        sound_velocity_str = _get_xml_attr(attrs, "SoundVelocity")
        freq_start_str = _get_xml_attr(attrs, "FrequencyStart")
        freq_end_str = _get_xml_attr(attrs, "FrequencyEnd")

        freq_single_str = _get_xml_attr(attrs, "Frequency")
        if freq_start_str and freq_end_str:
            frequency = (float(freq_start_str) + float(freq_end_str)) / 2
        elif freq_single_str:
            frequency = float(freq_single_str)
        else:
            frequency = None

        channel_settings[channel_id] = {
            "channel_id": channel_id,
            "sample_interval": float(sample_interval_str),
            "sound_velocity": float(sound_velocity_str) if sound_velocity_str else None,
            "frequency": frequency,  # FM midpoint fallback; overridden below
        }

    # Override with nominal frequency from static config (authoritative)
    nominal_freqs = _get_nominal_frequencies(data)
    for channel_id, settings in channel_settings.items():
        if channel_id in nominal_freqs:
            settings["frequency"] = nominal_freqs[channel_id]

    # Get sample counts from RAW3 datagrams
    datagram_types = raw_parser.find_datagram_types(data)
    if "RAW3" not in datagram_types:
        return list(channel_settings.values())

    for raw3_offset in datagram_types["RAW3"][:50]:
        offset = raw3_offset + 4 + 8  # Skip type + timestamp
        channel_id = data[offset : offset + 128].split(b"\x00")[0].decode("ascii")
        offset += 128 + 4  # Skip channel + datatype/spare
        offset += 4  # Skip sample offset
        sample_count = struct.unpack("<i", data[offset : offset + 4])[0]

        if channel_id in channel_settings:
            settings = channel_settings[channel_id]
            if "sample_count" not in settings:
                settings["sample_count"] = sample_count
                sv = settings["sound_velocity"] or 1500.0
                max_range = sample_count * settings["sample_interval"] * sv / 2
                settings["max_range"] = max_range

    return list(channel_settings.values())


def get_pulse_duration(path: str, channel_id: Optional[str] = None) -> float:
    """
    Extract transmit pulse duration (seconds) from a .raw file's XML config.

    Searches for PulseDuration attributes in the binary XML headers.
    If channel_id is provided, tries to find the value near that channel's
    configuration block (best-effort). Falls back to 0.001024 s (1.024 ms)
    if not found.

    Parameters
    ----------
    path : str
        Path to the .raw file.
    channel_id : str, optional
        Channel identifier. If provided, restricts search to the relevant
        region of the XML config. If None, returns the first match found.

    Returns
    -------
    float
        Pulse duration in seconds.
    """
    data = raw_parser.read_raw_file(path)

    # EK80 XML config stores PulseDuration as an attribute: PulseDuration="0.001024"
    pattern = rb'PulseDuration="([^"]+)"'

    if channel_id is not None:
        # Try to find a match near the channel's config block
        channel_bytes = channel_id.encode("ascii", errors="ignore")
        ch_pos = data.find(channel_bytes)
        if ch_pos != -1:
            # Search in a 10 kB window around the channel entry
            search_start = max(0, ch_pos - 5000)
            search_end = min(len(data), ch_pos + 5000)
            for match in re.finditer(pattern, data[search_start:search_end]):
                try:
                    value = float(match.group(1))
                    if 0 < value < 1.0:
                        return value
                except ValueError:
                    continue

    # Fall back to first valid match anywhere in the file
    for match in re.finditer(pattern, data):
        try:
            value = float(match.group(1))
            if 0 < value < 1.0:
                return value
        except ValueError:
            continue

    return 0.001024  # default: 1.024 ms


def get_file_summary(path: str) -> Dict[str, Any]:
    """
    Get a summary of a .raw file's contents.

    Returns dict with:
    - filename
    - size_mb
    - ping_count: Number of RAW3 datagrams
    - channels: List of channel summaries with range info
    """
    file_info = raw_parser.get_file_info(path)

    # Get channel settings
    channels = get_channel_settings(path)

    # Count pings
    datagram_types = file_info["datagram_types"]
    ping_count = len(datagram_types.get("RAW3", []))

    return {
        "filename": file_info["filename"],
        "size_mb": file_info["size_mb"],
        "ping_count": ping_count,
        "channels": channels,
    }
