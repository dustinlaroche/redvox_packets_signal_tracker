import os
import sys
import glob
import math
import numpy as np

# ============================================================
# USER SETTINGS (EDIT THESE)
# ============================================================

# Path to extracted redvox-3.8.8 folder that CONTAINS "redvox/" package directory
REDVOX_ROOT = r"/home/dlaroche/Desktop/redvox-3.8.8"

# Either a single .rdvxm file OR a directory containing .rdvxm files (recursive)
INPUT_PATH = r"/home/dlaroche/Desktop/RedVox"

# Receiver geocoordinates (the "start point")
RECEIVER_NAME = "Receiver"
RECEIVER_LAT_DEG = 41.8781      # EDIT
RECEIVER_LON_DEG = -87.6298     # EDIT
RECEIVER_ALT_M = 539.0          # meters above sea level (used by altitude estimate)

# Magnetometer series to FFT: "mag" (magnitude) or "x" / "y" / "z"
AXIS = "mag"

# Preprocessing
DEMEAN = True

# Event detection in frequency domain:
# Threshold is applied to FFT magnitude (scaled like your code: |FFT| / ADC_GAIN)
ADC_GAIN = 1.0                  # set if you need additional scaling
MAGNITUDE_THRESHOLD = 15.0       # EDIT based on your data

# Limit which frequencies are considered "signals" (Hz)
MIN_HZ = 0.001                    # ignore DC and very low bins
MAX_HZ = 50.0                   # None for no upper limit

# --- NEW: Optional altitude filter for output placemarks ---
# If ENABLE_ALTITUDE_FILTER is True, only output placemarks with altitude_est_m <= MAX_ALTITUDE_M
ENABLE_ALTITUDE_FILTER = True
MAX_ALTITUDE_M = 1000.0         # EDIT (meters). Example: 2000m keeps "low-altitude" events.

# KML output path
OUTPUT_KML = r"/home/dlaroche/Desktop/signal_origins.kml"

# ============================================================
# CONSTANTS (from your C++)
# ============================================================
EARTH_RADIUS_METERS = 6371000.0
SPEED_OF_LIGHT = 299792458.0
PI = math.pi

# ============================================================
# REDVOX LOADING (API1000 .rdvxm)
# ============================================================

def add_redvox_to_path(redvox_root: str) -> None:
    redvox_root = os.path.abspath(redvox_root)
    if not os.path.isdir(os.path.join(redvox_root, "redvox")):
        raise RuntimeError(
            "REDVOX_ROOT must point to the folder that contains the 'redvox' package.\n"
            f"Given: {redvox_root}\n"
            f"Expected: {os.path.join(redvox_root, 'redvox')}"
        )
    if redvox_root not in sys.path:
        sys.path.insert(0, redvox_root)


def list_rdvxm_files(input_path: str):
    input_path = os.path.abspath(input_path)
    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "**", "*.rdvxm"), recursive=True))
    else:
        paths = [input_path]
    paths = [p for p in paths if p.lower().endswith(".rdvxm") and os.path.isfile(p)]
    if not paths:
        raise RuntimeError(f"No .rdvxm files found at: {input_path}")
    return paths


def load_mag_series_from_rdvxm(input_path: str, axis: str):
    """
    Loads concatenated magnetometer time series from API1000 (.rdvxm).
    Returns: t_us (float64, microseconds), v (float64)
    """
    from redvox.api1000.wrapped_redvox_packet.wrapped_packet import WrappedRedvoxPacketM

    rdvxm_paths = list_rdvxm_files(input_path)
    t_all, x_all, y_all, z_all = [], [], [], []

    for p in rdvxm_paths:
        try:
            pkt = WrappedRedvoxPacketM.from_compressed_path(p)
        except Exception:
            continue

        sensors = pkt.get_sensors()
        mag = sensors.get_magnetometer() if sensors is not None else None
        if mag is None:
            continue

        t = np.asarray(mag.get_timestamps().get_timestamps(), dtype=np.float64)
        x = np.asarray(mag.get_x_samples().get_values(), dtype=np.float64)
        y = np.asarray(mag.get_y_samples().get_values(), dtype=np.float64)
        z = np.asarray(mag.get_z_samples().get_values(), dtype=np.float64)

        n = min(len(t), len(x), len(y), len(z))
        if n < 2:
            continue

        t_all.append(t[:n]); x_all.append(x[:n]); y_all.append(y[:n]); z_all.append(z[:n])

    if not t_all:
        raise RuntimeError(f"No magnetometer samples found in: {input_path}")

    t_us = np.concatenate(t_all)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    z = np.concatenate(z_all)

    order = np.argsort(t_us)
    t_us = t_us[order]; x = x[order]; y = y[order]; z = z[order]

    uniq = np.ones_like(t_us, dtype=bool)
    uniq[1:] = t_us[1:] != t_us[:-1]
    t_us = t_us[uniq]; x = x[uniq]; y = y[uniq]; z = z[uniq]

    axis = axis.lower().strip()
    if axis == "x":
        v = x
    elif axis == "y":
        v = y
    elif axis == "z":
        v = z
    elif axis == "mag":
        v = np.sqrt(x*x + y*y + z*z)
    else:
        raise ValueError("AXIS must be one of: 'x', 'y', 'z', 'mag'")

    if DEMEAN:
        v = v - np.mean(v)

    return t_us, v


def resample_to_uniform(t_us: np.ndarray, v: np.ndarray):
    """
    Uniform resample using median dt and linear interpolation.
    Returns: v_uniform, fs_hz
    """
    dt_us = np.diff(t_us)
    dt_us = dt_us[dt_us > 0]
    if dt_us.size == 0:
        raise ValueError("Non-increasing timestamps.")
    dt_med = float(np.median(dt_us))
    fs = 1e6 / dt_med

    t0 = float(t_us[0])
    t1 = float(t_us[-1])
    t_uniform_us = np.arange(t0, t1, dt_med, dtype=np.float64)
    v_uniform = np.interp(t_uniform_us, t_us, v)
    return v_uniform, fs

# ============================================================
# C++-ALGORITHM PORTS
# ============================================================

def calculate_wavelength(frequency_hz: float) -> float:
    return SPEED_OF_LIGHT / frequency_hz


def doa_horizontal_angle_deg(fft_bin: complex, from_magnetic_north: bool = False) -> float:
    angle_deg = math.degrees(math.atan2(fft_bin.imag, fft_bin.real))
    if from_magnetic_north:
        angle_deg = (angle_deg + 360.0) % 360.0
    return angle_deg


def doa_vertical_angle_deg(fft_bin: complex) -> float:
    return math.degrees(math.atan2(fft_bin.imag, fft_bin.real))


def doa_distance_meters_from_phase(fft_bin: complex, frequency_hz: float):
    if frequency_hz == 0.0:
        return 0.0, 0.0
    wavelength = calculate_wavelength(abs(frequency_hz))
    phase_shift = math.atan2(fft_bin.imag, fft_bin.real)
    d = abs(phase_shift) * wavelength / (2.0 * PI)
    return d, d


def doa_source_altitude_m(fft_bin: complex, frequency_hz: float, receiver_m_asl: float):
    dist_m, _ = doa_distance_meters_from_phase(fft_bin, frequency_hz)
    v_ang_rad = math.radians(doa_vertical_angle_deg(fft_bin))
    return receiver_m_asl + dist_m * math.tan(v_ang_rad)


def get_magnitudes(fft_result: np.ndarray, adc_gain: float) -> np.ndarray:
    return (np.abs(fft_result) / float(adc_gain)).astype(np.float64)


def find_bins_above_threshold(magnitudes: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(magnitudes > threshold)[0]

# ============================================================
# GEO / KML
# ============================================================

def deg_to_rad(deg: float) -> float:
    return deg * PI / 180.0

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / PI


def calculate_destination(start_lat_deg: float, start_lon_deg: float, distance_m: float, bearing_deg: float):
    bearing_rad = deg_to_rad(bearing_deg)
    lat1 = deg_to_rad(start_lat_deg)
    lon1 = deg_to_rad(start_lon_deg)

    dr = distance_m / EARTH_RADIUS_METERS

    lat2 = math.asin(math.sin(lat1) * math.cos(dr) +
                     math.cos(lat1) * math.sin(dr) * math.cos(bearing_rad))

    lon2 = lon1 + math.atan2(math.sin(bearing_rad) * math.sin(dr) * math.cos(lat1),
                             math.cos(dr) - math.sin(lat1) * math.sin(lat2))

    return rad_to_deg(lat2), rad_to_deg(lon2)


def kml_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def write_kml(filename: str, receiver_lat: float, receiver_lon: float, receiver_alt_m: float, placemarks: list):
    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write("<Document>\n")
        f.write(f"  <name>{kml_escape(os.path.basename(filename))}</name>\n")

        # Receiver placemark
        f.write("  <Placemark>\n")
        f.write(f"    <name>{kml_escape(RECEIVER_NAME)}</name>\n")
        f.write("    <description>Receiver location</description>\n")
        f.write("    <Point>\n")
        f.write(f"      <coordinates>{receiver_lon:.8f},{receiver_lat:.8f},{receiver_alt_m:.2f}</coordinates>\n")
        f.write("    </Point>\n")
        f.write("  </Placemark>\n")

        # Signal placemarks
        for pm in placemarks:
            f.write("  <Placemark>\n")
            f.write(f"    <name>{kml_escape(pm['name'])}</name>\n")
            f.write(f"    <description>{kml_escape(pm['description'])}</description>\n")
            f.write("    <Point>\n")
            f.write(f"      <coordinates>{pm['lon']:.8f},{pm['lat']:.8f},{pm['alt']:.2f}</coordinates>\n")
            f.write("    </Point>\n")
            f.write("  </Placemark>\n")

        f.write("</Document>\n</kml>\n")

# ============================================================
# MAIN
# ============================================================

def main():
    # Do not name this script "redvox.py"
    add_redvox_to_path(REDVOX_ROOT)

    # Load + resample magnetometer
    t_us, v = load_mag_series_from_rdvxm(INPUT_PATH, AXIS)
    v_u, fs_hz = resample_to_uniform(t_us, v)

    n = len(v_u)
    if n < 16:
        raise RuntimeError("Not enough samples after resampling for FFT.")

    # FFT (full spectrum)
    V = np.fft.fft(v_u)
    freq = np.fft.fftfreq(n, d=1.0 / fs_hz)

    magnitudes = get_magnitudes(V, ADC_GAIN)
    bins = find_bins_above_threshold(magnitudes, MAGNITUDE_THRESHOLD)

    # Frequency filtering
    keep_bins = []
    for b in bins.tolist():
        f_hz = float(freq[b])
        af = abs(f_hz)
        if af < float(MIN_HZ):
            continue
        if MAX_HZ is not None and af > float(MAX_HZ):
            continue
        keep_bins.append(b)

    if not keep_bins:
        print("No FFT bins above threshold after frequency filtering.")
        write_kml(OUTPUT_KML, RECEIVER_LAT_DEG, RECEIVER_LON_DEG, RECEIVER_ALT_M, [])
        print(f"Wrote KML (receiver only): {OUTPUT_KML}")
        return

    placemarks = []
    rejected_alt = 0

    for b in keep_bins:
        f_hz = float(freq[b])
        bin_c = complex(V[b])

        bearing_deg = doa_horizontal_angle_deg(bin_c, from_magnetic_north=True)
        vangle_deg = doa_vertical_angle_deg(bin_c)

        wl_m = calculate_wavelength(abs(f_hz))
        dist_m, _ = doa_distance_meters_from_phase(bin_c, f_hz)
        alt_m = doa_source_altitude_m(bin_c, f_hz, RECEIVER_ALT_M)

        # --- NEW: altitude filter ---
        if ENABLE_ALTITUDE_FILTER and alt_m > float(MAX_ALTITUDE_M):
            rejected_alt += 1
            continue

        dest_lat, dest_lon = calculate_destination(
            RECEIVER_LAT_DEG, RECEIVER_LON_DEG, dist_m, bearing_deg
        )

        desc = (
            f"bin={b}, f={f_hz:.6f} Hz, |FFT|/gain={magnitudes[b]:.6g}\n"
            f"bearing(deg)={bearing_deg:.3f}, vertical(deg)={vangle_deg:.3f}\n"
            f"wavelength(m)={wl_m:.6g}, distance_est(m)={dist_m:.6g}, altitude_est(m)={alt_m:.2f}\n"
            f"altitude_filter={'ON' if ENABLE_ALTITUDE_FILTER else 'OFF'} (max={MAX_ALTITUDE_M} m)"
        )

        placemarks.append({
            "name": f"Signal bin {b} ({f_hz:.3f} Hz)",
            "description": desc,
            "lat": dest_lat,
            "lon": dest_lon,
            "alt": alt_m,
        })

    write_kml(OUTPUT_KML, RECEIVER_LAT_DEG, RECEIVER_LON_DEG, RECEIVER_ALT_M, placemarks)

    print(f"Bins above threshold (after freq filter): {len(keep_bins)}")
    if ENABLE_ALTITUDE_FILTER:
        print(f"Rejected by altitude filter: {rejected_alt}")
    print(f"Placemark count written: {len(placemarks)}")
    print(f"Wrote KML: {OUTPUT_KML}")


main()
