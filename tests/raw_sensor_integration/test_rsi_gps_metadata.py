"""Characterization tests: NMEA/GPS decoding and XML0 metadata extraction."""

from datetime import datetime

import numpy as np
import pytest

from raw_sensor_integration import gps_parser, metadata

from ek80_synth import nme0, raw3, xml0

T0 = datetime(2024, 2, 1, 3, 50, 2)
T1 = datetime(2024, 2, 1, 3, 50, 3)
CH38 = "WBT 737172-7 ES38-18|200-18CR_ES"

GGA = "$GPGGA,035002.00,5932.1234,N,01045.6789,E,1,08,0.9,10.0,M,,M,,*47"
RMC = "$GPRMC,035003.00,A,5932.5000,N,01046.0000,E,10.2,180.0,010224,,,A*6C"
GLL = "$GPGLL,3352.8000,S,15112.6000,W,035004.00,A,A*7A"
VTG = "$GPVTG,180.0,T,,M,10.2,N,18.9,K,A*00"

LAT_GGA = 59 + 32.1234 / 60
LON_GGA = 10 + 45.6789 / 60


# ── NMEA ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, hemi, expected", [
    ("5932.1234", "N", 59 + 32.1234 / 60),
    ("01045.6789", "E", 10 + 45.6789 / 60),
    ("3352.8000", "S", -(33 + 52.8 / 60)),
    ("15112.6000", "W", -(151 + 12.6 / 60)),
])
def test_nmea_to_decimal_converts_ddmm_to_signed_degrees(value, hemi, expected):
    assert gps_parser.nmea_to_decimal(value, hemi) == pytest.approx(expected, abs=1e-12)


def test_parse_nme0_datagrams_reads_timestamp_and_sentence():
    data = nme0(T0, GGA) + nme0(T1, VTG)
    recs = gps_parser.parse_nme0_datagrams(data)
    assert recs == [{"timestamp": T0, "sentence": GGA},
                    {"timestamp": T1, "sentence": VTG}]


def test_parse_gps_sentences_decodes_gga_rmc_gll_and_skips_others():
    recs = [
        {"timestamp": T0, "sentence": GGA},
        {"timestamp": T1, "sentence": RMC},
        {"timestamp": T1, "sentence": VTG},
        {"timestamp": datetime(2024, 2, 1, 3, 50, 4), "sentence": GLL},
    ]
    df = gps_parser.parse_gps_sentences(recs)
    assert list(df["sentence_type"]) == ["$GPGGA", "$GPRMC", "$GPGLL"]
    np.testing.assert_allclose(df["lat"], [LAT_GGA, 59 + 32.5 / 60, -(33 + 52.8 / 60)])
    np.testing.assert_allclose(df["lon"], [LON_GGA, 10 + 46.0 / 60, -(151 + 12.6 / 60)])


def test_parse_gps_sentences_drops_duplicate_fix_at_same_timestamp():
    same_fix_rmc = "$GPRMC,035002.00,A,5932.1234,N,01045.6789,E,0,0,010224,,,A*00"
    df = gps_parser.parse_gps_sentences([
        {"timestamp": T0, "sentence": GGA},
        {"timestamp": T0, "sentence": same_fix_rmc},
    ])
    assert len(df) == 1 and df["sentence_type"].iloc[0] == "$GPGGA"


# ── XML0 metadata ──────────────────────────────────────────────────────────

CONFIG_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<Configuration><Transceivers><Transceiver><Channels>'
    f'<Channel ChannelID="{CH38}" PulseDuration="0.000512">'
    '<Transducer TransducerName="ES38-18" Frequency="38000" />'
    '</Channel>'
    '</Channels></Transceiver></Transceivers></Configuration>'
)
PARAM_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    f'<Parameter><Channel ChannelID="{CH38}" FrequencyStart="35000" '
    'FrequencyEnd="45000" SampleInterval="0.000128" SoundVelocity="1490" '
    'PulseDuration="0.001024" /></Parameter>'
)


def _raw_file(tmp_path, n_samples=1000):
    data = (xml0(T0, CONFIG_XML) + xml0(T0, PARAM_XML)
            + raw3(T0, CH38, np.zeros(n_samples, dtype=np.complex64), 0x08))
    path = tmp_path / "meta.raw"
    path.write_bytes(data)
    return path


def test_get_channel_settings_prefers_nominal_transducer_frequency(tmp_path):
    (ch,) = metadata.get_channel_settings(str(_raw_file(tmp_path)))
    assert ch["channel_id"] == CH38
    assert ch["frequency"] == 38000          # not the FM midpoint 40000
    assert ch["sample_interval"] == pytest.approx(0.000128)
    assert ch["sound_velocity"] == pytest.approx(1490.0)
    assert ch["sample_count"] == 1000
    # 1000 * 128 us * 1490 m/s / 2
    assert ch["max_range"] == pytest.approx(95.36)


def test_get_channel_settings_falls_back_to_fm_midpoint_without_config(tmp_path):
    path = tmp_path / "noconfig.raw"
    path.write_bytes(xml0(T0, PARAM_XML))
    (ch,) = metadata.get_channel_settings(str(path))
    assert ch["frequency"] == pytest.approx(40000.0)
    assert "sample_count" not in ch


def test_get_pulse_duration_takes_first_match_and_defaults(tmp_path):
    assert metadata.get_pulse_duration(str(_raw_file(tmp_path))) == pytest.approx(0.000512)
    empty = tmp_path / "empty.raw"
    empty.write_bytes(xml0(T0, "<Configuration/>"))
    assert metadata.get_pulse_duration(str(empty)) == pytest.approx(0.001024)
