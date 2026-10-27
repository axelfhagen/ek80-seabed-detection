"""Characterization tests: EK80 datagram framing and RAW3 ping decoding."""

from datetime import datetime

import numpy as np
import pytest

from raw_sensor_integration import ping_parser, raw_parser

from ek80_synth import nme0, nt_time, raw3, xml0

T0 = datetime(2024, 2, 1, 3, 50, 2)
CH38 = "WBT 737172-7 ES38-18|200-18CR_ES"
CH200 = "WBT 737173-15 ES200-7CDK-Split"


def _write(tmp_path, *chunks):
    path = tmp_path / "synthetic.raw"
    path.write_bytes(b"".join(chunks))
    return path


def test_nt_to_datetime_inverts_filetime_encoding():
    dt = datetime(2024, 2, 1, 3, 50, 2, 500000)
    assert ping_parser.nt_to_datetime(nt_time(dt)) == dt


def test_find_datagram_types_reports_tag_offsets_in_file_order():
    x = xml0(T0, "<Configuration/>")
    n = nme0(T0, "$GPGGA,035002.00,5932.1234,N,01045.6789,E,1,08,0.9,10.0,M,,M,,*47")
    r = raw3(T0, CH38, np.ones(4, dtype=np.complex64), data_type=0x08)
    data = x + n + r + n
    found = raw_parser.find_datagram_types(data)

    # tag sits 4 bytes after the start of each datagram
    assert found["XML0"] == [4]
    assert found["NME0"] == [len(x) + 4, len(x) + len(n) + len(r) + 4]
    assert found["RAW3"] == [len(x) + len(n) + 4]
    assert set(found) == {"XML0", "NME0", "RAW3"}


def test_parse_raw3_decodes_complex_float32_ping(tmp_path):
    samples = np.array([1 + 2j, 3 - 4j, 0.5j, -1], dtype=np.complex64)
    d = raw3(T0, CH38, samples, data_type=0x08, sample_offset=7)
    ping, next_offset = ping_parser.parse_raw3_datagram(d, 4)

    assert ping.channel_id == CH38
    assert ping.timestamp == T0
    assert ping.sample_offset == 7
    assert ping.sample_count == 4
    np.testing.assert_array_equal(ping.samples, samples)
    # next_offset is the byte just past the trailing length field, i.e. the
    # *leading length field* of the next datagram (not its 4-byte tag, which is
    # the convention for the ``offset`` argument).
    assert next_offset == len(d)
    np.testing.assert_allclose(ping.power, [5.0, 25.0, 0.25, 1.0])
    np.testing.assert_allclose(ping.power_db, 10 * np.log10([5.0, 25.0, 0.25, 1.0]), rtol=1e-6)


def test_parse_raw3_promotes_real_float32_samples_to_complex():
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    d = raw3(T0, CH38, samples, data_type=0x01)
    ping, _ = ping_parser.parse_raw3_datagram(d, 4)
    assert ping.samples.dtype == np.complex64
    np.testing.assert_allclose(ping.samples.real, samples)
    np.testing.assert_array_equal(ping.samples.imag, 0)


def test_extract_pings_filters_by_channel_substring_and_respects_max(tmp_path):
    chunks = [xml0(T0, "<Configuration/>")]
    for k in range(3):
        t = datetime(2024, 2, 1, 3, 50, 2 + k)
        chunks.append(raw3(t, CH38, np.full(3, k + 1, dtype=np.complex64), 0x08))
        chunks.append(raw3(t, CH200, np.full(2, 10 * (k + 1), dtype=np.complex64), 0x08))
    path = _write(tmp_path, *chunks)

    all_pings = ping_parser.extract_pings(str(path))
    assert [p.channel_id for p in all_pings] == [CH38, CH200] * 3

    p38 = ping_parser.extract_pings(str(path), channel_filter="ES38")
    assert [p.samples[0].real for p in p38] == [1.0, 2.0, 3.0]
    assert [p.timestamp.second for p in p38] == [2, 3, 4]

    assert len(ping_parser.extract_pings(str(path), max_pings=2)) == 2


def test_extract_pings_returns_empty_list_without_raw3(tmp_path):
    path = _write(tmp_path, xml0(T0, "<Configuration/>"))
    assert ping_parser.extract_pings(str(path)) == []


def test_get_ping_matrix_zero_pads_shorter_pings_samples_by_pings(tmp_path):
    chunks = [
        raw3(T0, CH38, np.array([1, 2, 3], dtype=np.complex64), 0x08),
        raw3(T0, CH38, np.array([4, 5], dtype=np.complex64), 0x08),
    ]
    pings = ping_parser.extract_pings(str(_write(tmp_path, *chunks)))
    matrix, times = ping_parser.get_ping_matrix(pings, "ES38")
    np.testing.assert_array_equal(matrix, np.array([[1, 4], [2, 5], [3, 0]]))
    assert times == [T0, T0]

    img = ping_parser.pings_to_power_image(pings, "ES38")
    # |1|^2 -> 0 dB, |4|^2 -> 10log10(16); zero-padding clamped to -200 dB
    np.testing.assert_allclose(img[0], [0.0, 10 * np.log10(16)], atol=1e-5)
    assert img[2, 1] == pytest.approx(-200.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EK80 DataType bit0=Power, bit1=Angle (Simrad EK80 interface spec); "
        "decode_data_type maps value 1 to 'Angle' via a 0-based list index"
    ),
)
def test_decode_data_type_bit0_means_power():
    assert ping_parser.decode_data_type(0x01)["base_type"] == "Power"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EK80 complex data stores NComplex values per sample in bits 8-10 of "
        "DataType (e.g. 0x0408 = 4 sectors). parse_raw3_datagram truncates the "
        "interleaved buffer to sample_count, mixing sectors instead of returning "
        "one value per range sample"
    ),
)
def test_parse_raw3_multi_sector_complex_returns_one_value_per_range_sample():
    n_samples, n_sectors = 3, 4
    # value encodes (sample, sector) as sample*10 + sector
    grid = np.array(
        [[s * 10 + k for k in range(n_sectors)] for s in range(n_samples)], dtype=np.complex64
    )
    d = raw3(T0, CH38, grid.reshape(-1), data_type=0x0408, sample_count=n_samples)
    ping, _ = ping_parser.parse_raw3_datagram(d, 4)
    # consecutive entries must belong to consecutive range samples
    np.testing.assert_array_equal(np.diff(ping.samples.real // 10), [1, 1])


def test_extract_xml_content_returns_null_terminated_xml_text():
    xml = '<?xml version="1.0"?><Configuration><Header Version="1"/></Configuration>'
    data = xml0(T0, xml) + raw3(T0, CH38, np.ones(2, dtype=np.complex64), 0x08)
    assert raw_parser.extract_xml_content(data) == [xml]
