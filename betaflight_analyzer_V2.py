import streamlit as st
# Set page config as the very first Streamlit command
st.set_page_config(page_title="Advanced Betaflight Log Analyzer", layout="wide")

import os
import io
import json
import re
import tempfile
import pathlib
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
# Use INFO for general use, DEBUG for detailed development/troubleshooting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_pid_response(P: float, D: float, wn: float = 5.0, zeta: float = 0.7):
    """
    Simulate the closed-loop step response of a system controlled by a PD controller.
    The plant is assumed to be a second-order system: G(s) = 1 / (s^2 + 2*zeta*wn*s + wn^2)
    and the controller is C(s) = P + D*s.
    Returns: (time, response, overshoot, rise_time)
    """
    # Define plant transfer function
    num_plant = [1.0]
    den_plant = [1.0, 2 * zeta * wn, wn**2]
    # Define controller transfer function (PD)
    num_controller = [D, P]
    den_controller = [1.0]
    # Open-loop transfer function: L(s) = C(s)*G(s)
    num_open = np.polymul(num_controller, num_plant)
    den_open = np.polymul(den_controller, den_plant)
    # Closed-loop transfer function: T(s) = L(s) / (1 + L(s))
    den_closed = np.polyadd(den_open, num_open)
    system = signal.TransferFunction(num_open, den_closed)
    # Simulate step response
    t, y = signal.step(system)
    overshoot = max(y) - 1.0 if max(y) > 1.0 else 0.0
    # Calculate rise time between 10% and 90%
    try:
        t10 = t[np.where(y >= 0.1)[0][0]]
        t90 = t[np.where(y >= 0.9)[0][0]]
        rise_time = t90 - t10
    except Exception:
        rise_time = None
    return t, y, overshoot, rise_time

def optimize_pid_for_axis(current_P: float, current_D: float, desired_overshoot: float = 0.05, desired_rise_time: float = 1.0):
    """
    Sweeps a range around the current P and D values and simulates the step response.
    Returns the recommended P, D values and a cost score.
    """
    best_score = float('inf')
    best_P = current_P
    best_D = current_D
    for P in np.linspace(current_P * 0.8, current_P * 1.2, 10):
        for D in np.linspace(current_D * 0.8, current_D * 1.2, 10):
            t, y, overshoot, rise_time = simulate_pid_response(P, D)
            if rise_time is None:
                continue
            # Define a simple cost function that sums the overshoot and rise time errors
            score = abs(overshoot - desired_overshoot) + abs(rise_time - desired_rise_time)
            if score < best_score:
                best_score = score
                best_P = P
                best_D = D
    return best_P, best_D, best_score


# --- Helper Function for JSON Serialization ---
def make_serializable(obj: any) -> any:
    """
    Recursively convert non-serializable types (like numpy types, datetime, etc.)
    into serializable types for JSON storage. Handles common issues.
    """
    # Basic types first
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Numpy numeric types
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        # Convert numpy integer types to standard Python int
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        # Convert numpy float types to standard Python float
        if np.isnan(obj): return None # Represent NaN as null in JSON
        if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity" # Represent infinity as strings
        return float(obj)
    # Numpy boolean type
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Numpy arrays
    elif isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj): return [str(item) for item in obj] # Complex to string list
        return make_serializable(obj.tolist()) # Recursively handle arrays
    # Python lists/tuples
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj] # Recursively handle items
    # Dictionaries
    elif isinstance(obj, dict):
        # Recursively handle dictionaries with depth limit
        def serialize_dict_recursive(d, depth=0, max_depth=15): # Increased max_depth
            if depth > max_depth:
                logger.warning(f"Dictionary serialization truncated at depth {max_depth}")
                return f"Dict serialization skipped (max depth {max_depth} reached)"
            try:
                # Ensure keys are strings for JSON compatibility
                return {str(k): serialize_dict_recursive(v, depth+1, max_depth)
                        if isinstance(v, dict) else make_serializable(v)
                        for k, v in d.items()}
            except RecursionError:
                 logger.error("Recursion error during dict serialization.")
                 return "Dict serialization failed (RecursionError)"
        return serialize_dict_recursive(obj)
    # Datetime objects
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() # Datetime to ISO string
    # Plotly figures (avoid serializing)
    elif isinstance(obj, go.Figure):
        return "Plotly Figure Object (Not Serialized)"
    # Objects with to_dict (e.g., some Pandas objects)
    elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            # Avoid serializing large DataFrames via to_dict
            if isinstance(obj, pd.DataFrame) and len(obj) > 1000:
                 return f"Large DataFrame {obj.shape} (Not Serialized)"
            temp_dict = obj.to_dict()
            # Limit size of resulting dicts
            return make_serializable(temp_dict) if len(temp_dict) < 1000 else f"Large {type(obj).__name__} dict skipped"
        except Exception as e:
            logger.warning(f"Error serializing {type(obj).__name__} using to_dict: {e}")
            return f"Error serializing {type(obj).__name__} via to_dict: {e}"
    # Pandas NA
    elif pd.isna(obj):
        return None
    # Pathlib objects
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    # Fallback for other types
    else:
        try:
            if not isinstance(obj, bytes): return str(obj) # Try string conversion
            else: return f"Non-serializable type: {type(obj)}" # Bytes are not JSON serializable
        except Exception as e:
            logger.debug(f"Failed to serialize object of type {type(obj)}: {e}")
            return f"Non-serializable type: {type(obj)}"

# --- Main Class for Betaflight Log Analysis ---
class BetaflightLogAnalyzer:
    """
    Analyzes Betaflight blackbox log files, aligning with common tuning methodologies.
    Provides methods for parsing, analysis, comparison, and history tracking.
    """
    def __init__(self):
        # Use pathlib for path manipulation
        self.base_dir = pathlib.Path(".") # Use current directory or specify another
        self.logs_db_path = self.base_dir / "logs_database.json"
        self.tuning_history_path = self.base_dir / "tuning_history.json"
        self._ensure_db_files_exist()
        
    def get_tuning_history(self) -> list:
        """
        Retrieves the tuning history from the JSON file.
        Returns a list of historical log analysis entries.
        """
        try:
            with self.tuning_history_path.open('r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Validate history is a list
            if not isinstance(history, list):
                logger.warning(f"Invalid tuning history format in {self.tuning_history_path}. Expected a list.")
                return {"error": "Invalid history format"}
            
            return history
        except FileNotFoundError:
            logger.info(f"Tuning history file not found at {self.tuning_history_path}.")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON in tuning history file {self.tuning_history_path}")
            return {"error": "JSON decoding failed"}
        except Exception as e:
            logger.error(f"Unexpected error reading tuning history: {e}")
            return {"error": str(e)}

        # Initialize column descriptions dictionary for better field interpreation
        self.column_descriptions = {
            "loopIteration": "Counter for each flight controller loop iteration",
            "time": "Timestamp in microseconds since flight start",
            "axisP[0]": "Proportional PID term for Roll axis",
            "axisP[1]": "Proportional PID term for Pitch axis",
            "axisP[2]": "Proportional PID term for Yaw axis",
            "axisI[0]": "Integral PID term for Roll axis",
            "axisI[1]": "Integral PID term for Pitch axis",
            "axisI[2]": "Integral PID term for Yaw axis",
            "axisD[0]": "Derivative PID term for Roll axis",
            "axisD[1]": "Derivative PID term for Pitch axis",
            "axisD[2]": "Derivative PID term for Yaw axis",
            "axisF[0]": "Feedforward PID term for Roll axis",
            "axisF[1]": "Feedforward PID term for Pitch axis",
            "axisF[2]": "Feedforward PID term for Yaw axis",
            "rcCommand[0]": "RC input for Roll from transmitter",
            "rcCommand[1]": "RC input for Pitch from transmitter",
            "rcCommand[2]": "RC input for Yaw from transmitter",
            "rcCommand[3]": "RC input for Throttle from transmitter",
            "setpoint[0]": "Target rotation rate for Roll",
            "setpoint[1]": "Target rotation rate for Pitch",
            "setpoint[2]": "Target rotation rate for Yaw",
            "setpoint[3]": "Throttle setpoint (target throttle)",
            "vbatLatest": "Battery voltage reading",
            "amperageLatest": "Current draw (amperes * 100)",
            "baroAlt": "Altitude from barometer (if available)",
            "gyroADC[0]": "Raw gyro sensor data - Roll axis",
            "gyroADC[1]": "Raw gyro sensor data - Pitch axis",
            "gyroADC[2]": "Raw gyro sensor data - Yaw axis",
            "accSmooth[0]": "Filtered accelerometer data - Roll axis",
            "accSmooth[1]": "Filtered accelerometer data - Pitch axis",
            "accSmooth[2]": "Filtered accelerometer data - Yaw axis",
            "motor[0]": "Motor 0 throttle command",
            "motor[1]": "Motor 1 throttle command",
            "motor[2]": "Motor 2 throttle command",
            "motor[3]": "Motor 3 throttle command",
            "flightModeFlags": "Flags indicating active flight modes",
            "stateFlags": "Flags for system state (e.g., arming, failsafe)",
            "failsafePhase": "Current failsafe phase (if triggered)"
        }

    def _ensure_db_files_exist(self):
        """Creates empty database JSON files if they don't exist."""
        for db_path in [self.logs_db_path, self.tuning_history_path]:
            if not db_path.exists():
                try:
                    with db_path.open('w', encoding='utf-8') as f: # Specify encoding
                        # Initialize history as a list, database as a dict
                        json.dump([] if db_path == self.tuning_history_path else {}, f)
                    logger.info(f"Initialized empty database file: {db_path}")
                except Exception as e:
                    logger.error(f"Error initializing database file {db_path}: {e}")

    def _read_log_file(self, file_path: str) -> list:
        """Reads log file content, handling basic errors."""
        logger.debug(f"Attempting to read file: {file_path}")
        path = pathlib.Path(file_path)
        if not path.is_file(): # More specific check
            logger.error(f"File not found or is not a file: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            raise ValueError("File is empty.")
        try:
            # Read with error replacement for robustness
            with path.open('r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise IOError(f"Could not read file: {e}") from e

        if len(lines) < 10: # Basic sanity check for minimum content
            logger.error(f"File has very few lines (<10): {file_path}")
            raise ValueError("File has too few lines (<10).")
        logger.debug(f"Successfully read {len(lines)} lines from {file_path}")
        return lines

    def _find_header_and_data(self, lines: list) -> tuple[list, str, int]:
        """
        Locates the header row and the start of the data section using multiple strategies.
        Returns: (metadata_lines, header_line, data_start_index)
        """
        data_line_idx = -1
        metadata_lines = []
        header_line = None
        # Common columns expected in the header to aid detection
        # Added 'axisD[0]' as it's common and less likely in comments
        common_cols = ['gyroadc[0]', 'rccommand[0]', 'motor[0]', 'time', 'axisd[0]']

        # 1. Primary: Look for "loopIteration" (case-insensitive) at the start of a line
        for i, line in enumerate(lines):
            cleaned_line = line.strip().lower().replace('"', '')
            if not cleaned_line or cleaned_line.startswith(('#', 'H ')): # Skip comments/blanks
                 metadata_lines.append(lines[i])
                 continue
            potential_header_start = cleaned_line.split(',')[0].strip()
            if potential_header_start == "loopiteration":
                data_line_idx = i
                header_line = lines[i]
                logger.debug(f"Found header 'loopIteration' at line {i}.")
                break
            metadata_lines.append(lines[i]) # Assume it's metadata until header found

        # 2. Fallback: Look for a line containing several common column names
        if data_line_idx == -1:
            logger.debug("Header 'loopIteration' not found, checking for common columns...")
            # Reset metadata_lines as the previous loop added everything
            metadata_lines = []
            for i, line in enumerate(lines):
                lower_line = line.lower()
                # Check if line contains enough common columns and isn't likely data itself
                # Check if line starts with non-numeric to avoid matching data lines
                if sum(col in lower_line for col in common_cols) >= 3 and \
                   line.strip() and not line.strip()[0].isdigit() and not line.strip().startswith('-'):
                    # Check if the *next* line looks like data (starts with a number, has commas)
                    if i + 1 < len(lines) and lines[i+1].strip() and lines[i+1].strip()[0].isdigit() and ',' in lines[i+1]:
                        data_line_idx = i
                        header_line = lines[i]
                        metadata_lines = lines[:i] # Metadata is everything before this line
                        logger.debug(f"Found potential header by common columns at line {i}.")
                        break
                metadata_lines.append(lines[i]) # Collect potential metadata lines

        # 3. Error if no header found
        if data_line_idx == -1 or header_line is None:
            logger.error("Could not reliably identify the log data header.")
            # Provide more info for debugging
            logger.debug("First 20 lines for header check:")
            for k, l in enumerate(lines[:20]): logger.debug(f"L{k}: {l.strip()}")
            raise ValueError("Could not reliably identify the log data header. Check log format.")

        # Ensure there's data after the header
        data_start_index = data_line_idx + 1
        if data_start_index >= len(lines) or not lines[data_start_index].strip():
            logger.error("No data found after the identified header row.")
            raise ValueError("No data found after the header row.")

        logger.info(f"Identified header at line {data_line_idx}, data starts at line {data_start_index}.")
        return metadata_lines, header_line.strip(), data_start_index # Return stripped header

    def parse_metadata(self, metadata_lines: list) -> dict:
        """Parses key-value pairs and specific settings from metadata lines."""
        logger.debug(f"Parsing metadata from {len(metadata_lines)} lines.")
        metadata = {
            'firmware': {}, 'hardware': {}, 'pids': {}, 'rates': {},
            'filters': {}, 'features': {}, 'other_settings': {},
            'analysis_info': {} # For storing analysis-related info like time units
        }

        # --- Patterns ---
        # Regex patterns grouped by category for better organization

        # Firmware/Board Info
        fw_patterns = {
            'betaflight_version': re.compile(r'Betaflight\s+/\s+\w+\s+(\d+\.\d+\.\d+)', re.IGNORECASE),
            'firmware_target': re.compile(r'Firmware target:\s*(\S+)', re.IGNORECASE),
            'firmware_revision': re.compile(r'Firmware revision:\s*(.+)', re.IGNORECASE),
            'firmware_date': re.compile(r'Firmware date:\s*(.+)', re.IGNORECASE),
            'board_name': re.compile(r'Board information:\s*\w+\s*([\w-]+)', re.IGNORECASE),
            'craft_name': re.compile(r'Craft name:\s*(.*)', re.IGNORECASE),
        }

        # PID Settings (including FF, D_Min etc.) - Expecting comma-separated values in quotes or simple values
        pid_patterns = {
            'rollPID': re.compile(r'"?rollPID"?\s*,\s*"([^"]+)"', re.IGNORECASE),
            'pitchPID': re.compile(r'"?pitchPID"?\s*,\s*"([^"]+)"', re.IGNORECASE),
            'yawPID': re.compile(r'"?yawPID"?\s*,\s*"([^"]+)"', re.IGNORECASE),
            'levelPID': re.compile(r'"?levelPID"?\s*,\s*"([^"]+)"', re.IGNORECASE),
            'd_min_roll': re.compile(r'"?d_min_roll"?\s*,\s*(\d+)', re.IGNORECASE),
            'd_min_pitch': re.compile(r'"?d_min_pitch"?\s*,\s*(\d+)', re.IGNORECASE),
            'd_min_yaw': re.compile(r'"?d_min_yaw"?\s*,\s*(\d+)', re.IGNORECASE), # Usually 0
            'd_min_gain': re.compile(r'"?d_min_gain"?\s*,\s*(\d+)', re.IGNORECASE),
            'd_min_advance': re.compile(r'"?d_min_advance"?\s*,\s*(\d+)', re.IGNORECASE),
            'f_pitch': re.compile(r'"?f_pitch"?\s*,\s*(\d+)', re.IGNORECASE), # Feedforward
            'f_roll': re.compile(r'"?f_roll"?\s*,\s*(\d+)', re.IGNORECASE),
            'f_yaw': re.compile(r'"?f_yaw"?\s*,\s*(\d+)', re.IGNORECASE),
            'feedforward_transition': re.compile(r'"?feedforward_transition"?\s*,\s*(\d+)', re.IGNORECASE),
            'ff_boost': re.compile(r'"?ff_boost"?\s*,\s*(\d+)', re.IGNORECASE),
            'iterm_relax': re.compile(r'"?iterm_relax"?\s*,\s*(\d+)', re.IGNORECASE), # Type (Setpoint/Gyro)
            'iterm_relax_type': re.compile(r'"?iterm_relax_type"?\s*,\s*(\d+)', re.IGNORECASE),
            'iterm_relax_cutoff': re.compile(r'"?iterm_relax_cutoff"?\s*,\s*(\d+)', re.IGNORECASE),
            'anti_gravity_gain': re.compile(r'"?anti_gravity_gain"?\s*,\s*(\d+)', re.IGNORECASE),
            'tpa_rate': re.compile(r'"?tpa_rate"?\s*,\s*(\d+)', re.IGNORECASE),
            'tpa_breakpoint': re.compile(r'"?tpa_breakpoint"?\s*,\s*(\d+)', re.IGNORECASE),
            'tpa_mode': re.compile(r'"?tpa_mode"?\s*,\s*(\d+)', re.IGNORECASE), # 0=D, 1=PD
            'abs_control_gain': re.compile(r'"?abs_control_gain"?\s*,\s*(\d+)', re.IGNORECASE),
        }

        # Rate Settings
        rate_patterns = {
            'rates': re.compile(r'"?rates"?\s*,\s*"([^"]+)"', re.IGNORECASE), # Roll, Pitch, Yaw rates
            'rc_rates': re.compile(r'"?rc_rates"?\s*,\s*"([^"]+)"', re.IGNORECASE),
            'rc_expo': re.compile(r'"?rc_expo"?\s*,\s*"([^"]+)"', re.IGNORECASE),
            'thrMid': re.compile(r'"?thrMid"?\s*,\s*(\d+)', re.IGNORECASE),
            'thrExpo': re.compile(r'"?thrExpo"?\s*,\s*(\d+)', re.IGNORECASE),
            'rates_type': re.compile(r'"?rates_type"?\s*,\s*(\d+)', re.IGNORECASE), # Betaflight, Actual, Quickrates
        }

        # Filter Settings
        filter_patterns = {
            'gyro_lpf': re.compile(r'"?gyro_lpf"?\s*,\s*(\d+)', re.IGNORECASE), # Mode (OFF, ON)
            'gyro_lowpass_type': re.compile(r'"?gyro_soft_type"?\s*,\s*(\d+)', re.IGNORECASE), # Type (PT1, BIQUAD)
            'gyro_lowpass_hz': re.compile(r'"?gyro_lowpass_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'gyro_lowpass2_hz': re.compile(r'"?gyro_lowpass2_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'gyro_lowpass2_type': re.compile(r'"?gyro_soft2_type"?\s*,\s*(\d+)', re.IGNORECASE),
            'gyro_notch_hz': re.compile(r'"?gyro_notch_hz"?\s*,\s*"([^"]+)"', re.IGNORECASE), # List
            'gyro_notch_cutoff': re.compile(r'"?gyro_notch_cutoff"?\s*,\s*"([^"]+)"', re.IGNORECASE), # List
            'dterm_filter_type': re.compile(r'"?dterm_filter_type"?\s*,\s*(\d+)', re.IGNORECASE), # Type (PT1, BIQUAD)
            'dterm_lpf_hz': re.compile(r'"?dterm_lpf_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'dterm_lpf_dyn_hz': re.compile(r'"?dterm_lpf_dyn_hz"?\s*,\s*"([^"]+)"', re.IGNORECASE), # List
            'dterm_filter2_type': re.compile(r'"?dterm_filter2_type"?\s*,\s*(\d+)', re.IGNORECASE),
            'dterm_lpf2_hz': re.compile(r'"?dterm_lpf2_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'dterm_notch_hz': re.compile(r'"?dterm_notch_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'dterm_notch_cutoff': re.compile(r'"?dterm_notch_cutoff"?\s*,\s*(\d+)', re.IGNORECASE),
            'yaw_lpf_hz': re.compile(r'"?yaw_lpf_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'dyn_notch_max_hz': re.compile(r'"?dyn_notch_max_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'dyn_notch_count': re.compile(r'"?dyn_notch_count"?\s*,\s*(\d+)', re.IGNORECASE),
            'dyn_notch_q': re.compile(r'"?dyn_notch_q"?\s*,\s*(\d+)', re.IGNORECASE),
            'dyn_notch_min_hz': re.compile(r'"?dyn_notch_min_hz"?\s*,\s*(\d+)', re.IGNORECASE),
            'gyro_rpm_notch_harmonics': re.compile(r'"?gyro_rpm_notch_harmonics"?\s*,\s*(\d+)', re.IGNORECASE),
            'gyro_rpm_notch_q': re.compile(r'"?gyro_rpm_notch_q"?\s*,\s*(\d+)', re.IGNORECASE),
            'gyro_rpm_notch_min': re.compile(r'"?gyro_rpm_notch_min"?\s*,\s*(\d+)', re.IGNORECASE),
        }

        # Feature Flags / Other Settings
        other_patterns = {
            'features': re.compile(r'"?features"?\s*,\s*(-?\d+)', re.IGNORECASE), # Bitmask
            'looptime': re.compile(r'"?looptime"?\s*,\s*(\d+)', re.IGNORECASE), # Microseconds
            'pid_process_denom': re.compile(r'"?pid_process_denom"?\s*,\s*(\d+)', re.IGNORECASE),
            'acc_1G': re.compile(r'"?acc_1G"?\s*,\s*(\d+)', re.IGNORECASE),
            'motorOutput': re.compile(r'"?motorOutput"?\s*,\s*"([^"]+)"', re.IGNORECASE), # Range
            'debug_mode': re.compile(r'"?debug_mode"?\s*,\s*(\d+)', re.IGNORECASE),
            'vbat_pid_gain': re.compile(r'"?vbat_pid_gain"?\s*,\s*(\d+)', re.IGNORECASE), # ON/OFF (0/1)
            'use_integrated_yaw': re.compile(r'"?use_integrated_yaw"?\s*,\s*(\d+)', re.IGNORECASE),
            'dshot_bidir': re.compile(r'"?dshot_bidir"?\s*,\s*(\d+)', re.IGNORECASE),
            'motor_poles': re.compile(r'"?motor_poles"?\s*,\s*(\d+)', re.IGNORECASE),
        }

        # Combine all pattern dictionaries
        all_patterns = {
            'firmware': fw_patterns,
            'hardware': {}, # Board name is firmware pattern, add others if needed
            'pids': pid_patterns,
            'rates': rate_patterns,
            'filters': filter_patterns,
            'features': {'features': other_patterns.pop('features')}, # Move features here
            'other_settings': other_patterns # Remaining settings
        }

        # Process lines
        for line in metadata_lines:
            line = line.strip()
            if not line or line.startswith(('#', 'H ', 'loopIteration')): continue

            matched = False
            for category, patterns in all_patterns.items():
                for key, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        try:
                            # Determine if it's a list or single value based on pattern structure
                            if '"([^"]+)"' in pattern.pattern: # Pattern expects quoted comma-separated list
                                values_str = match.group(1).split(',')
                                values = []
                                for x in values_str:
                                    x_strip = x.strip()
                                    if not x_strip: continue
                                    try: values.append(float(x_strip))
                                    except ValueError: values.append(x_strip) # Keep as string if not float
                                metadata[category][key] = values[0] if len(values) == 1 else values # Store single value if list has one item
                            elif len(match.groups()) == 1: # Pattern expects single value capture group
                                value_str = match.group(1).strip()
                                try: metadata[category][key] = float(value_str) # Try float first
                                except ValueError: metadata[category][key] = value_str # Fallback to string
                            else: # Should not happen with defined patterns
                                 logger.warning(f"Unexpected match groups for key {key} in line: {line}")

                            matched = True
                            break # Stop checking patterns for this line once matched
                        except Exception as e:
                            logger.warning(f"Error parsing metadata key '{key}' from line '{line}': {e}")
                            matched = True # Mark as matched to avoid logging as unmatched
                            break
                if matched: break # Stop checking categories for this line
            # if not matched: logger.debug(f"Metadata line not matched: {line}") # Optional

        # Add analysis timestamp
        metadata['analysis_info']['analysis_timestamp'] = datetime.now().isoformat()
        logger.info(f"Metadata parsing complete. Found keys in categories: { {k: len(v) for k, v in metadata.items() if isinstance(v, dict)} }")
        return metadata

    # --- Data Parsing and Preparation ---
    def parse_data(self, header_line: str, data_lines: list) -> pd.DataFrame:
        """Parses the data section into a Pandas DataFrame."""
        # Clean header line just in case (remove potential surrounding quotes)
        header_line_cleaned = header_line.strip().replace('"', '')
        csv_content = header_line_cleaned + "\n" + ''.join(data_lines)
        try:
            logger.debug("Parsing data using pandas.read_csv")
            df = pd.read_csv(io.StringIO(csv_content),
                             header=0,
                             skipinitialspace=True, # Handle spaces after commas
                             on_bad_lines='warn', # Warn about bad lines but try to continue
                             quotechar='"',
                             low_memory=False) # May improve parsing robustness for mixed types

            # Remove potential duplicate columns (keeping the first occurrence)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            logger.debug(f"Pandas read_csv successful, shape: {df.shape}")

            # Clean column names robustly
            df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
            logger.debug(f"Cleaned columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error parsing data with pandas: {e}\nHeader: {header_line_cleaned[:100]}...\nFirst data line: {data_lines[0][:100] if data_lines else 'N/A'}...")
            raise ValueError(f"Failed to parse log data into DataFrame: {e}")

    def prepare_data(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Cleans and prepares the DataFrame for analysis."""
        logger.debug(f"Starting data preparation. Initial shape: {df.shape}, Columns: {df.columns.tolist()}")

        # 1. Handle Time Column and Units
        if 'time' not in df.columns:
            if 'loopIteration' in df.columns:
                df.rename(columns={'loopIteration': 'time'}, inplace=True)
                metadata['analysis_info']['time_unit'] = 'us'  # loopIteration is typically microseconds
                logger.info("Using 'loopIteration' as time column (microseconds).")
            else:
                logger.warning("No 'time' or 'loopIteration' column found. Cannot perform time-based analysis.")
                if pd.api.types.is_numeric_dtype(df[col]):
                     df['time'] = df.index
                     metadata['analysis_info']['time_unit'] = 'index' # Indicate it's just the index
                     logger.warning("Created synthetic 'time' column from index. Time unit unknown.")
                else:
                    raise ValueError("Missing time information ('time' or 'loopIteration' column).")
        elif not pd.api.types.is_numeric_dtype(df['time']):
             df['time'] = pd.to_numeric(df['time'], errors='coerce')
             if df['time'].isnull().any():
                 raise ValueError("'time' column contains non-numeric values.")
             metadata['analysis_info'].setdefault('time_unit', 'us') # Assume microseconds if 'time' column exists
             logger.info("Assuming existing 'time' column is in microseconds.")
        else:
             metadata['analysis_info'].setdefault('time_unit', 'us') # Assume microseconds if 'time' column exists
             logger.info("Using existing numeric 'time' column (assumed microseconds).")

        # 2. Convert Data Columns to Numeric
        numeric_cols = []
        non_numeric_problem_cols = []
        for col in df.columns:
            if col != 'time':
                if not pd.api.types.is_numeric_dtype(df[col]):
                    original_nan = df[col].isnull().sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    new_nan = df[col].isnull().sum()
                    if new_nan > original_nan and (new_nan - original_nan) > 0.05 * len(df):
                        non_numeric_problem_cols.append(col)
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
        if non_numeric_problem_cols:
            logger.warning(f"High NaN count after conversion for columns: {non_numeric_problem_cols}.")

        # 3. Handle Missing Values
        essential_gyro_cols = [c for c in df.columns if c.lower().startswith('gyro') and c in numeric_cols]
        if essential_gyro_cols:
             initial_rows = len(df)
             df.dropna(subset=essential_gyro_cols, how='any', inplace=True)
             rows_dropped = initial_rows - len(df)
             if rows_dropped > 0:
                 logger.warning(f"Dropped {rows_dropped} rows due to missing essential gyro data.")

        if numeric_cols:
            original_nan_count = df[numeric_cols].isnull().sum().sum()
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            filled_nan_count = original_nan_count - df[numeric_cols].isnull().sum().sum()
            if filled_nan_count > 0:
                logger.debug(f"Filled {filled_nan_count} missing values in numeric columns using ffill/bfill.")

        # 4. Set Time Index
        if 'time' in df.columns:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['time'], keep='last')
            if len(df) < initial_rows:
                logger.debug(f"Removed {initial_rows - len(df)} duplicate time rows.")

            if pd.api.types.is_numeric_dtype(df['time']) and not df['time'].isnull().all():
                try:
                    if not df['time'].is_unique:
                        logger.warning("Time column not unique after cleaning. Averaging duplicates.")
                        numeric_agg = {col: 'mean' for col in df.select_dtypes(include=np.number).columns if col != 'time'}
                        non_numeric_agg = {col: 'first' for col in df.select_dtypes(exclude=np.number).columns}
                        df = df.groupby(df['time']).agg({**numeric_agg, **non_numeric_agg}).reset_index()

                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)
                    logger.debug("Set 'time' column as index and sorted.")
                except Exception as e:
                    logger.error(f"Could not set 'time' as index: {e}. Proceeding without time index.", exc_info=True)
            else:
                 logger.warning("Time column is not suitable for indexing (non-numeric or all NaN).")
        else:
             logger.warning("No 'time' column available to set as index.")

        if df.empty:
            raise ValueError("DataFrame became empty after preparation.")

        logger.info(f"Data preparation finished. Final shape: {df.shape}")
        return df

    # --- Analysis Methods ---
    def diagnose_data_quality(self, df: pd.DataFrame) -> dict:
        """Assesses the quality and integrity of the log data."""
        logger.debug("Diagnosing data quality...")
        diagnostics = {
            "missing_data": {},
            "unusual_values": [],
            "sampling_issues": {},
            "quality_score": 1.0, # Start with perfect score
            "summary": "Data quality checks passed.", # Default summary
            "diagnosis": [] # List of specific issues found
        }
        total_rows = len(df)
        if total_rows == 0:
             diagnostics["summary"] = "No data rows remaining after preparation."
             diagnostics["quality_score"] = 0.0
             return diagnostics

        # Check for essential columns (assuming 'time' index is set or 'time' column exists)
        essential_columns = [c for c in df.columns if c.lower().startswith('gyro')]
        if not essential_columns:
            diagnostics["missing_data"]["essential_columns"] = "Gyro data (gyroRoll, gyroPitch, gyroYaw)"
            diagnostics["quality_score"] -= 0.5
            diagnostics["diagnosis"].append("Missing essential gyro columns.")

        # Check NaN percentages
        nan_percentages = (df.isnull().sum() / total_rows * 100)
        high_nan_cols = nan_percentages[nan_percentages > 20] # Threshold for high NaN %
        if not high_nan_cols.empty:
            diagnostics["missing_data"]["high_nan_percentages (>20%)"] = high_nan_cols.round(1).to_dict()
            diagnostics["quality_score"] -= 0.2
            diagnostics["diagnosis"].append(f"High percentage of missing data in columns: {list(high_nan_cols.index)}")

        # Check time sampling regularity (if time index exists and is numeric/time-like)
        if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.RangeIndex)) or np.issubdtype(df.index.dtype, np.number):
            time_diffs = pd.Series(df.index).diff().dropna()
            if len(time_diffs) > 1:
                # Convert diffs to seconds for consistent comparison
                if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
                    time_diffs_sec = time_diffs.dt.total_seconds()
                elif np.issubdtype(df.index.dtype, np.number):
                    # Assume microseconds if numeric index
                    time_diffs_sec = time_diffs / 1_000_000.0
                else: time_diffs_sec = time_diffs # Should not happen if RangeIndex

                time_diff_mean_sec = time_diffs_sec.mean()
                time_diff_std_sec = time_diffs_sec.std()

                # Check for irregular sampling intervals
                if time_diff_mean_sec > 1e-9 and time_diff_std_sec > 0.2 * time_diff_mean_sec: # 20% deviation threshold
                    diagnostics["sampling_issues"]["irregular_sampling"] = True
                    diagnostics["sampling_issues"]["time_diff_std_ms"] = round(time_diff_std_sec* 1000, 3)
                    diagnostics["sampling_issues"]["time_diff_mean_ms"] = round(time_diff_mean_sec * 1000, 3)
                    diagnostics["quality_score"] -= 0.15
                    diagnostics["diagnosis"].append("Irregular sampling intervals detected (check log rate).")

                # Check for large gaps (e.g., > 10x mean interval or > 0.1 seconds)
                gap_threshold_sec = max(10 * time_diff_mean_sec, 0.1) if time_diff_mean_sec > 1e-9 else 0.1
                large_gaps = time_diffs_sec[time_diffs_sec > gap_threshold_sec]
                if not large_gaps.empty:
                    diagnostics["sampling_issues"]["data_gaps_count"] = len(large_gaps)
                    diagnostics["sampling_issues"]["max_gap_s"] = round(large_gaps.max(), 3)
                    diagnostics["quality_score"] -= 0.1
                    diagnostics["diagnosis"].append(f"Found {len(large_gaps)} significant gaps in time data (max: {large_gaps.max():.3f}s).")

        # Check for unusual value ranges (Gyro, Motors)
        for col in essential_columns: # Check Gyro columns
            if col in df.columns:
                # Check for extreme values (e.g., > 2500 deg/s)
                extreme_values = df[df[col].abs() > 2500]
                if len(extreme_values) > 0.005 * total_rows: # More than 0.5% extreme values
                    diagnostics["unusual_values"].append({
                        "column": col,
                        "issue": "Extreme values (> +/-2500 deg/s)",
                        "percentage": round(len(extreme_values) / total_rows * 100, 1)
                    })
                    diagnostics["quality_score"] -= 0.05
                    diagnostics["diagnosis"].append(f"Unusually high values (>2500 deg/s) found in {col}.")
                # Check for flatlined data (low standard deviation over rolling window)
                if len(df[col]) > 100:
                    rolling_std = df[col].rolling(window=50, center=True).std()
                    flatline_pct = (rolling_std < 0.1).mean() # Threshold for 'flat'
                    if flatline_pct > 0.1: # More than 10% of time looks flat
                        diagnostics["unusual_values"].append({
                            "column": col,
                            "issue": "Potential flatlined data",
                            "flatline_percentage": round(flatline_pct * 100, 1)
                        })
                        diagnostics["quality_score"] -= 0.1
                        diagnostics["diagnosis"].append(f"Potential flatlined sensor data detected in {col}.")


        motor_columns = [col for col in df.columns if col.lower().startswith('motor') and col[5:].isdigit()]
        if motor_columns:
            motor_data = df[motor_columns]
            motor_min = motor_data.min().min()
            motor_max = motor_data.max().max()
            if pd.isna(motor_min) or pd.isna(motor_max):
                 diagnostics["diagnosis"].append("Motor data could not be evaluated due to missing values.")
                 diagnostics["quality_score"] -= 0.05
            # Check if values are wildly outside typical ranges (0-1, 1000-2000, 0-1000)
            elif motor_min < -50 or motor_max > 2200:
                diagnostics["unusual_values"].append({
                    "type": "motor_out_of_range",
                    "min": round(motor_min,1),
                    "max": round(motor_max,1)
                })
                diagnostics["quality_score"] -= 0.1
                diagnostics["diagnosis"].append(f"Motor values outside expected range ({motor_min:.1f} to {motor_max:.1f}).")

        # Final Score and Summary
        score = max(0, min(1, diagnostics["quality_score"])) # Clamp score between 0 and 1
        diagnostics["quality_score"] = round(score, 2)
        if score < 0.4:
            diagnostics["summary"] = "Poor data quality. Analysis reliability may be low due to significant issues."
        elif score < 0.6:
            diagnostics["summary"] = "Fair data quality. Some analysis results may be affected by identified issues."
        elif score < 0.8:
            diagnostics["summary"] = "Good data quality, but minor issues detected."
        else:
             # Only override default if score isn't perfect but still high
             if score < 1.0 and not diagnostics["diagnosis"]:
                  diagnostics["summary"] = "Excellent data quality overall, potential minor inconsistencies."
             elif not diagnostics["diagnosis"]:
                  diagnostics["summary"] = "Excellent data quality. No significant issues detected."
             # If issues were found but score is high, mention them
             elif diagnostics["diagnosis"]:
                   diagnostics["summary"] = f"Good/Excellent data quality, but note: {diagnostics['diagnosis'][0]}"


        logger.debug(f"Data Quality Score: {diagnostics['quality_score']:.2f}. Diagnosis: {diagnostics['diagnosis']}")
        return diagnostics

    def analyze_pid_performance(self, df: pd.DataFrame, metadata: dict) -> dict:
        """
        Analyzes PID tracking error and step response characteristics.
        
        Now handles multiple gyro column naming formats including:
        - Standard names: gyroRoll, gyroPitch, gyroYaw
        - Array indices: gyroADC[0], gyroADC[1], gyroADC[2]
        
        And setpoint formats:
        - Standard names: setpointRoll, setpointPitch, setpointYaw
        - Array indices: setpoint[0], setpoint[1], setpoint[2]
        """
        logger.debug("Analyzing PID performance...")
        results = {}
        
        # Find relevant columns case-insensitively with multiple naming pattern support
        gyro_cols = {'roll': None, 'pitch': None, 'yaw': None}
        setpoint_cols = {'roll': None, 'pitch': None, 'yaw': None}
        
        # Check for standard names first (gyroRoll, gyroPitch, gyroYaw)
        for axis in ['roll', 'pitch', 'yaw']:
            # Try standard naming format
            std_gyro_name = f'gyro{axis.capitalize()}'
            if std_gyro_name in df.columns:
                gyro_cols[axis] = std_gyro_name
            
            # Try array index format for Betaflight >=4.x (gyroADC[0], gyroADC[1], gyroADC[2])
            array_idx = 0 if axis == 'roll' else 1 if axis == 'pitch' else 2
            array_gyro_name = f'gyroADC[{array_idx}]'
            if array_gyro_name in df.columns:
                gyro_cols[axis] = array_gyro_name
                
            # Try legacy formats (case-insensitive search)
            if gyro_cols[axis] is None:
                for col in df.columns:
                    if col.lower() == f'gyro{axis}' or col.lower() == f'gyro_{axis}':
                        gyro_cols[axis] = col
                        break
        
        # Similar search for setpoint columns
        for axis in ['roll', 'pitch', 'yaw']:
            # Try standard naming format
            std_setpoint_name = f'setpoint{axis.capitalize()}'
            if std_setpoint_name in df.columns:
                setpoint_cols[axis] = std_setpoint_name
            
            # Try array index format (setpoint[0], setpoint[1], setpoint[2])
            array_idx = 0 if axis == 'roll' else 1 if axis == 'pitch' else 2
            array_setpoint_name = f'setpoint[{array_idx}]'
            if array_setpoint_name in df.columns:
                setpoint_cols[axis] = array_setpoint_name
                
            # Try legacy formats (case-insensitive search)
            if setpoint_cols[axis] is None:
                for col in df.columns:
                    if col.lower() == f'setpoint{axis}' or col.lower() == f'setpoint_{axis}':
                        setpoint_cols[axis] = col
                        break
        
        # Check if essential gyro data exists
        has_gyro = any(gyro_cols.values())
        has_setpoint = any(setpoint_cols.values())
        
        if not has_gyro:
            results["error_gyro"] = "No gyro data found. Checked for standard names (gyroRoll) and array format (gyroADC[0])."
            logger.error(results["error_gyro"])
            return {"pid": results}
        
        analysis_performed = False
        for axis in ['roll', 'pitch', 'yaw']:
            gyro_col = gyro_cols[axis]
            setpoint_col = setpoint_cols[axis]

            # Analyze Gyro data if present
            if gyro_col and gyro_col in df.columns and pd.api.types.is_numeric_dtype(df[gyro_col]):
                gyro_data = df[gyro_col].dropna()
                if not gyro_data.empty:
                    results[f"{axis}_gyro_mean"] = gyro_data.mean()
                    results[f"{axis}_gyro_std"] = gyro_data.std()
                    results[f"{axis}_gyro_range"] = gyro_data.max() - gyro_data.min()
                    # Basic noise metric (mean absolute difference)
                    results[f"{axis}_gyro_noise_metric"] = gyro_data.diff().abs().mean()
                    analysis_performed = True

                    # Analyze Tracking Error and Step Response if Setpoint data also exists
                    if setpoint_col and setpoint_col in df.columns and pd.api.types.is_numeric_dtype(df[setpoint_col]):
                        setpoint_data = df[setpoint_col].dropna()
                        # Align gyro and setpoint data on their index (time)
                        aligned_gyro, aligned_setpoint = gyro_data.align(setpoint_data, join='inner')

                        if not aligned_gyro.empty and len(aligned_gyro) > 10: # Need some data points
                            # Calculate Tracking Error
                            error = aligned_setpoint - aligned_gyro
                            results[f"{axis}_tracking_error_mean"] = error.mean()
                            results[f"{axis}_tracking_error_std"] = error.std()
                            results[f"{axis}_tracking_error_mae"] = error.abs().mean() # Mean Absolute Error
                            results[f"{axis}_tracking_error_max_abs"] = error.abs().max()

                            # Calculate Step Response if enough data
                            if len(aligned_setpoint) > 100:
                                logger.debug(f"Calculating step response for {axis}...")
                                # Pass aligned data to step response function
                                results[f"{axis}_step_response"] = self.calculate_step_response(aligned_setpoint, aligned_gyro)
                            else:
                                results[f"{axis}_step_response"] = {"status": "Not enough aligned data for step response."}
                        else:
                            results[f"{axis}_tracking_error_warning"] = "Could not align gyro/setpoint or not enough aligned data."
                            results[f"{axis}_step_response"] = {"status": "Alignment failed or insufficient data."}
                    # If Gyro exists but Setpoint doesn't
                    elif f"{axis}_gyro_mean" in results: # Check if gyro analysis was done
                         results[f"{axis}_tracking_error_warning"] = f"Gyro data found for {axis}, but no corresponding setpoint data for tracking analysis."
                         results[f"{axis}_step_response"] = {"status": "Setpoint data missing."}

        if not analysis_performed:
            results["error_overall"] = "Could not perform PID analysis on any axis."
        elif not has_setpoint:
            results["warning_setpoint"] = "No setpoint data found – performed gyro stability analysis only."

        logger.debug("PID analysis completed.")
        return {"pid": results}

    def calculate_step_response(self, setpoint: pd.Series, actual: pd.Series) -> dict:
        """Calculates step response metrics (rise time, overshoot, settling time)."""
        metrics_list = []
        index_type = setpoint.index.dtype

        # Determine index type for time calculations
        is_time_index = isinstance(setpoint.index, (pd.TimedeltaIndex, pd.DatetimeIndex))
        is_numeric_index = np.issubdtype(index_type, np.number)

        if not (is_time_index or is_numeric_index):
            logger.warning("Step response requires a numeric or time-based index.")
            return {"status": "Time index required", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None}

        # Calculate step detection threshold
        setpoint_diff = setpoint.diff().abs().dropna()
        if len(setpoint_diff) < 5: # Need a few points for robust stats
            return {"status": "Not enough data points for diff stats", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None}
        diff_mean = setpoint_diff.mean()
        diff_std = setpoint_diff.std()
        setpoint_range = setpoint.max() - setpoint.min()
        # Threshold based on standard deviation and percentage of range
        threshold = max(diff_mean + 3 * diff_std, 0.05 * setpoint_range if setpoint_range > 1e-6 else 1.0) if diff_std > 1e-9 else (0.05 * setpoint_range if setpoint_range > 1e-6 else 1.0)
        threshold = max(threshold, 1e-6) # Ensure threshold is positive

        # Find potential step indices
        step_indices_raw = setpoint_diff[setpoint_diff > threshold].index

        # Filter steps that are too close together
        min_step_interval_sec = 0.05 # Minimum time between analyzed steps
        filtered_step_indices = []
        if not step_indices_raw.empty:
            last_step_time = step_indices_raw[0]
            filtered_step_indices.append(last_step_time)
            for current_step_time in step_indices_raw[1:]:
                try:
                    time_diff_val = current_step_time - last_step_time
                    if is_time_index:
                        time_diff_sec = time_diff_val.total_seconds()
                    elif is_numeric_index:
                        time_diff_sec = time_diff_val / 1_000_000.0 # Convert microseconds to seconds
                    else: continue # Should not happen based on initial check

                    if time_diff_sec >= min_step_interval_sec:
                        filtered_step_indices.append(current_step_time)
                        last_step_time = current_step_time
                except Exception as e:
                     logger.error(f"Error filtering step times: {e}")
                     continue

        if not filtered_step_indices:
            return {"status": "No significant steps found after filtering", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None}

        analysis_window_sec = 0.2 # How long after the step to analyze

        # Analyze each filtered step (up to a limit)
        for step_time in filtered_step_indices[:20]: # Limit number of steps analyzed
            try:
                # Define analysis window based on index type
                if is_time_index:
                    end_time = step_time + pd.Timedelta(seconds=analysis_window_sec)
                elif is_numeric_index:
                    end_time = step_time + (analysis_window_sec * 1_000_000) # Add microseconds
                else: continue

                # Select data within the window
                # Use try-except for loc in case start/end times are slightly out of bounds
                try:
                    window_setpoint = setpoint.loc[step_time:end_time]
                    window_actual = actual.loc[step_time:end_time]
                except KeyError:
                    logger.debug(f"Skipping step at {step_time}: window extends beyond data range.")
                    continue

                if len(window_setpoint) < 5 or len(window_actual) < 5: continue # Need enough points in window

                initial_setpoint = window_setpoint.iloc[0]
                # Use mean of last few points for final setpoint to be robust to noise
                final_setpoint = window_setpoint.iloc[-min(5, len(window_setpoint)):].mean()
                initial_actual = window_actual.iloc[0]
                setpoint_change = final_setpoint - initial_setpoint

                # Skip if setpoint change is too small
                if abs(setpoint_change) < threshold * 0.5: continue

                rise_time_ms = None
                overshoot_percent = None
                settling_time_ms = None

                # --- Rise Time (10% to 90%) ---
                try:
                    target_10pct = initial_actual + 0.1 * setpoint_change
                    target_90pct = initial_actual + 0.9 * setpoint_change

                    if setpoint_change >= 0: # Rising step
                        time_10pct_indices = window_actual[window_actual >= target_10pct].index
                        time_90pct_indices = window_actual[window_actual >= target_90pct].index
                    else: # Falling step
                        time_10pct_indices = window_actual[window_actual <= target_10pct].index
                        time_90pct_indices = window_actual[window_actual <= target_90pct].index

                    if not time_10pct_indices.empty and not time_90pct_indices.empty:
                        time_10 = time_10pct_indices[0]
                        time_90 = time_90pct_indices[0]
                        if time_90 >= time_10:
                            rise_time_val = time_90 - time_10
                            # ** CORRECTED CONVERSION **
                            if is_time_index:
                                rise_time_ms = rise_time_val.total_seconds() * 1000
                            elif is_numeric_index:
                                rise_time_ms = rise_time_val / 1000.0 # Convert microsecond diff to ms
                except Exception as e_rise:
                    logger.debug(f"Rise time calculation error: {e_rise}")

                # --- Overshoot ---
                try:
                    if setpoint_change > 0: # Rising step
                        peak_value = window_actual.max()
                        overshoot = peak_value - final_setpoint
                    elif setpoint_change < 0: # Falling step
                        peak_value = window_actual.min()
                        # Overshoot for falling step is how far it went below the final setpoint
                        overshoot = final_setpoint - peak_value
                    else: overshoot = 0

                    if abs(setpoint_change) > 1e-9:
                        overshoot_percent = max(0, (overshoot / abs(setpoint_change)) * 100)
                    else: overshoot_percent = 0
                except Exception as e_over:
                    logger.debug(f"Overshoot calculation error: {e_over}")

                # --- Settling Time (within +/- 5% band) ---
                try:
                    settling_band_half_width = max(0.05 * abs(setpoint_change), 1e-6) # 5% band or minimum value
                    lower_bound = final_setpoint - settling_band_half_width
                    upper_bound = final_setpoint + settling_band_half_width

                    # Find the last time the signal was outside the settling band within the window
                    outside_band_indices = window_actual[(window_actual < lower_bound) | (window_actual > upper_bound)].index
                    if outside_band_indices.empty:
                        # Already settled at the start of the window (or immediately)
                        settling_time_ms = 0.0
                    else:
                        last_unsettled_time = outside_band_indices[-1]
                        # Settling time is the time from the step start to when it last left the band
                        settling_time_val = last_unsettled_time - step_time
                        # ** CORRECTED CONVERSION **
                        if is_time_index:
                            settling_time_ms = settling_time_val.total_seconds() * 1000
                        elif is_numeric_index:
                            settling_time_ms = settling_time_val / 1000.0 # Convert microsecond diff to ms

                        # Ensure settling time isn't negative (shouldn't happen with correct logic)
                        settling_time_ms = max(0, settling_time_ms) if settling_time_ms is not None else None

                except Exception as e_settle:
                    logger.debug(f"Settling time calculation error: {e_settle}")

                # Append calculated metrics for this step
                metrics_list.append({
                    "rise_time_ms": rise_time_ms,
                    "overshoot_percent": overshoot_percent,
                    "settling_time_ms": settling_time_ms
                })
            except Exception as e_outer:
                logger.error(f"Error processing step response at index {step_time}: {e_outer}", exc_info=True)
                continue # Skip to next step if error occurs

        # Calculate median of metrics across all valid steps
        if not metrics_list:
            return {"status": "Could not calculate metrics for any step", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            valid_values = [m[key] for m in metrics_list if m[key] is not None and np.isfinite(m[key])] # Check for finite values
            # Use nanmedian for robustness against outliers
            avg_metrics[key] = round(np.nanmedian(valid_values), 2) if valid_values else None

        avg_metrics["status"] = f"Median from {len(metrics_list)} analyzed steps"
        return avg_metrics

    def analyze_motors(self, df: pd.DataFrame) -> dict:
        """Analyzes motor output, saturation, and balance."""
        logger.debug("Analyzing motors...")
        results = {}

        # Specifically look for motor[0], motor[1], motor[2], motor[3]
        motor_cols = ['motor[0]', 'motor[1]', 'motor[2]', 'motor[3]']
        
        # Verify columns exist in dataframe
        motor_cols = [col for col in motor_cols if col in df.columns]

        if not motor_cols:
            results["error_motors"] = "No motor data columns found."
            return {"motors": results}

        try:
            motor_data = df[motor_cols].dropna(how='all')
            
            if motor_data.empty:
                results["error_motors"] = "Motor columns exist but contain only NaN values."
                return {"motors": results}

            # Detect motor output range and type
            max_value = motor_data.max().max()
            min_value = motor_data.min().min()

            # Determine motor value type based on range
            if min_value >= 1000 and max_value <= 2000:
                motor_range_max = 2000
                motor_range_min = 1000
                results["motor_value_type"] = "PWM/DShot (1000-2000)"
            elif 0 <= min_value and max_value <= 1:
                motor_range_max = 1.0
                motor_range_min = 0.0
                results["motor_value_type"] = "Normalized (0-1)"
            else:
                motor_range_max = max_value
                motor_range_min = min_value
                results["motor_value_type"] = f"Custom ({motor_range_min}-{motor_range_max})"

            # Basic motor statistics
            results["max_motor_output"] = round(max_value, 2)
            results["min_motor_output"] = round(min_value, 2)
            results["motor_range_detected"] = f"{motor_range_min} - {motor_range_max}"

            # Motor saturation calculation
            saturation_threshold = motor_range_max * 0.98
            saturated_points = (motor_data >= saturation_threshold).sum()
            total_points = len(motor_data)

            results["motor_saturation_pct_per_motor"] = (saturated_points / total_points * 100).round(2).to_dict()
            results["motor_saturation_pct_overall"] = round((saturated_points.sum() / motor_data.size) * 100, 2)

            # Motor averages and balance
            avg_motors = motor_data.mean()
            results["motor_averages"] = avg_motors.round(2).to_dict()
            motor_imbalance_std = avg_motors.std()
            results["motor_imbalance_std_dev"] = round(motor_imbalance_std, 2)
            
            # Relative motor balance
            if avg_motors.max() - avg_motors.min() > 0:
                results["motor_imbalance_pct"] = round(motor_imbalance_std / avg_motors.mean() * 100, 2)

            # Throttle information from rcCommand[3]
            throttle_col = 'rcCommand[3]'
            if throttle_col in df.columns and pd.api.types.is_numeric_dtype(df[throttle_col]):
                throttle_data = df[throttle_col].dropna()
                if not throttle_data.empty:
                    results["avg_throttle"] = round(throttle_data.mean(), 2)
                    results["throttle_range"] = [round(throttle_data.min(), 2), round(throttle_data.max(), 2)]

                    # Normalize throttle
                    throttle_min_rc = throttle_data.min()
                    throttle_max_rc = throttle_data.max()

                    if throttle_min_rc >= 1000 and throttle_max_rc <= 2000:
                        norm_min, norm_max = 1000, 2000
                    elif 0 <= throttle_min_rc and throttle_max_rc <= 1:
                        norm_min, norm_max = 0, 1
                    else:
                        norm_min, norm_max = throttle_min_rc, throttle_max_rc

                    if norm_max > norm_min:
                        normalized_throttle = ((throttle_data - norm_min) / (norm_max - norm_min)).clip(0, 1)
                        
                        # Throttle distribution
                        throttle_bins = [0, 0.25, 0.5, 0.75, 1.0]
                        labels = [f"{int(throttle_bins[i]*100)}-{int(throttle_bins[i+1]*100)}%" for i in range(len(throttle_bins)-1)]
                        
                        try:
                            throttle_dist = pd.cut(normalized_throttle, bins=throttle_bins, labels=labels, include_lowest=True).value_counts(normalize=True)
                            results["throttle_distribution_pct"] = (throttle_dist * 100).round(1).to_dict()
                        except Exception as e:
                            results["throttle_distribution_error"] = f"Error calculating distribution: {e}"

            return {"motors": results}

        except Exception as e:
            results["error_motors"] = f"Error analyzing motor data: {str(e)}"
            return {"motors": results}

    def analyze_gyro_accel(self, df: pd.DataFrame) -> dict:
        """
        Analyzes gyroscope and accelerometer data for sensor health and characteristics.
        
        Handles various column naming conventions for gyro and accelerometer data.
        """
        logger.debug("Analyzing Gyro and Accelerometer data...")
        results = {}

        # Identify gyro columns
        gyro_cols = {
            'roll': next((col for col in df.columns if col in ['gyroADC[0]', 'gyroRoll', 'gyro_roll']), None),
            'pitch': next((col for col in df.columns if col in ['gyroADC[1]', 'gyroPitch', 'gyro_pitch']), None),
            'yaw': next((col for col in df.columns if col in ['gyroADC[2]', 'gyroYaw', 'gyro_yaw']), None)
        }

        # Identify accelerometer columns
        acc_cols = {
            'roll': next((col for col in df.columns if col in ['accSmooth[0]', 'accRoll', 'acc_roll']), None),
            'pitch': next((col for col in df.columns if col in ['accSmooth[1]', 'accPitch', 'acc_pitch']), None),
            'yaw': next((col for col in df.columns if col in ['accSmooth[2]', 'accYaw', 'acc_yaw']), None)
        }

        # Analyze gyro data
        for axis, col in gyro_cols.items():
            if col is not None and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                gyro_data = df[col].dropna()
                if not gyro_data.empty:
                    results[f'{axis}_gyro_mean'] = round(gyro_data.mean(), 2)
                    results[f'{axis}_gyro_std'] = round(gyro_data.std(), 2)
                    results[f'{axis}_gyro_min'] = int(gyro_data.min())
                    results[f'{axis}_gyro_max'] = int(gyro_data.max())
                    
                    # Noise and variability metrics
                    diff_data = gyro_data.diff().dropna()
                    results[f'{axis}_gyro_noise_metric'] = round(diff_data.std(), 2)
                    results[f'{axis}_gyro_max_diff'] = int(diff_data.abs().max())

                    # Potential sensor issues detection
                    anomaly_threshold = 3 * results[f'{axis}_gyro_std']
                    anomalies = gyro_data[np.abs(gyro_data - results[f'{axis}_gyro_mean']) > anomaly_threshold]
                    if len(anomalies) > 0:
                        results[f'{axis}_gyro_anomalies'] = {
                            'count': len(anomalies),
                            'percentage': round(len(anomalies) / len(gyro_data) * 100, 2)
                        }

        # Analyze accelerometer data
        for axis, col in acc_cols.items():
            if col is not None and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                acc_data = df[col].dropna()
                if not acc_data.empty:
                    results[f'{axis}_acc_mean'] = round(acc_data.mean(), 2)
                    results[f'{axis}_acc_std'] = round(acc_data.std(), 2)
                    results[f'{axis}_acc_min'] = int(acc_data.min())
                    results[f'{axis}_acc_max'] = int(acc_data.max())

        # Overall sensor health assessment
        sensor_issues = []
        for axis in ['roll', 'pitch', 'yaw']:
            gyro_std_key = f'{axis}_gyro_std'
            if gyro_std_key in results:
                if results[gyro_std_key] == 0:
                    sensor_issues.append(f'Potential gyro flatline detected on {axis} axis')

        if sensor_issues:
            results['sensor_issues'] = sensor_issues

        if not results:
            results['error_gyro'] = "No valid gyro or accelerometer data found."

        logger.debug("Gyro and Accelerometer analysis complete.")
        return results

    def analyze_altitude_power(self, df: pd.DataFrame) -> dict:
        """
        Analyzes altitude and power-related data from the log.
        """
        logger.debug("Analyzing Altitude and Power...")
        results = {}

        # Find altitude column
        alt_col = next((col for col in df.columns if col.lower() in ['altitudebaro', 'baroalt', 'baroAlt']), None)
        
        # Find voltage and current columns
        voltage_col = next((col for col in df.columns if col.lower() in ['vbatlatest', 'voltage', 'vbat']), None)
        current_col = next((col for col in df.columns if col.lower() in ['amperagelatest', 'current', 'amperage']), None)

        # Altitude analysis
        if alt_col and alt_col in df.columns and pd.api.types.is_numeric_dtype(df[alt_col]):
            alt_data = df[alt_col].dropna()
            if not alt_data.empty:
                results['altitude_mean'] = round(alt_data.mean(), 2)
                results['altitude_max'] = int(alt_data.max())
                results['altitude_min'] = int(alt_data.min())
                results['altitude_std'] = round(alt_data.std(), 2)

                # Detect climb/descend rates
                alt_diff = alt_data.diff()
                results['max_climb_rate'] = int(alt_diff[alt_diff > 0].max() or 0)
                results['max_descend_rate'] = int(alt_diff[alt_diff < 0].min() or 0)

        # Power analysis
        if voltage_col and voltage_col in df.columns and pd.api.types.is_numeric_dtype(df[voltage_col]):
            voltage_data = df[voltage_col].dropna()
            results['voltage_mean'] = round(voltage_data.mean(), 2)
            results['voltage_min'] = int(voltage_data.min())
            results['voltage_max'] = int(voltage_data.max())

        if current_col and current_col in df.columns and pd.api.types.is_numeric_dtype(df[current_col]):
            current_data = df[current_col].dropna()
            results['current_mean'] = round(current_data.mean(), 2)
            results['current_max'] = int(current_data.max())

            # Power calculation (if both voltage and current are available)
            if 'voltage_mean' in results:
                results['power_mean'] = round(results['voltage_mean'] * results['current_mean'], 2)
                results['power_max'] = int(results['voltage_max'] * results['current_max'])

        return results

    def analyze_rc_vs_gyro(self, df: pd.DataFrame, metadata: dict) -> dict:
        """
        Analyzes RC command vs Gyro response latency and correlation.
        """
        logger.debug("Analyzing RC Command vs Gyro Latency...")
        results = {}

        # Find RC and Gyro columns
        rc_columns = {
            'roll': next((col for col in df.columns if col in ['rcCommand[0]', 'rcCommands[0]']), None),
            'pitch': next((col for col in df.columns if col in ['rcCommand[1]', 'rcCommands[1]']), None),
            'yaw': next((col for col in df.columns if col in ['rcCommand[2]', 'rcCommands[2]']), None)
        }

        gyro_columns = {
            'roll': next((col for col in df.columns if col in ['gyroADC[0]', 'gyroRoll']), None),
            'pitch': next((col for col in df.columns if col in ['gyroADC[1]', 'gyroPitch']), None),
            'yaw': next((col for col in df.columns if col in ['gyroADC[2]', 'gyroYaw']), None)
        }

        # Time conversion factor
        time_conversion_factor = 1_000_000  # microseconds to seconds

        for axis in ['roll', 'pitch', 'yaw']:
            rc_col = rc_columns[axis]
            gyro_col = gyro_columns[axis]

            if rc_col and gyro_col and rc_col in df.columns and gyro_col in df.columns:
                rc_data = df[rc_col].dropna()
                gyro_data = df[gyro_col].dropna()

                # Align data
                common_index = rc_data.index.intersection(gyro_data.index)
                rc_data = rc_data.loc[common_index]
                gyro_data = gyro_data.loc[common_index]

                # Calculate RC command and Gyro rates
                rc_rate = rc_data.diff()
                gyro_rate = gyro_data.diff()

                # Basic latency and correlation analysis
                if len(rc_rate) > 10 and len(gyro_rate) > 10:
                    # Pearson correlation between RC rate and Gyro rate
                    try:
                        correlation = rc_rate.corr(gyro_rate)
                        results[f'{axis}_rc_gyro_correlation'] = round(correlation, 3)
                    except Exception as e:
                        logger.warning(f"Correlation calculation failed for {axis}: {e}")

                    # Simple response time estimation
                    response_times = []
                    for threshold in [0.1, 0.5, 0.9]:  # Different levels of response
                        try:
                            time_to_threshold = (rc_rate.abs() >= threshold * rc_rate.max()).argmax()
                            response_times.append(time_to_threshold)
                        except Exception:
                            pass

                    if response_times:
                        results[f'{axis}_avg_response_time_idx'] = round(np.mean(response_times), 2)

        logger.debug("RC vs Gyro analysis completed.")
        return results



    def perform_spectral_analysis(self, df: pd.DataFrame, sampling_rate: float = None, metadata: dict = None) -> dict:
        """Performs FFT spectral analysis on gyro data."""
        logger.debug("Starting spectral analysis...")
        results = {'spectra': {}}
        
        # Build a list of potential gyro column names
        potential_gyro_cols = []
        
        # Standard axis naming (gyroRoll, etc.)
        potential_gyro_cols.extend([col for col in df.columns if col.lower() in ['gyroroll', 'gyropitch', 'gyroyaw']])
        
        # Array notation (gyroADC[0], etc.)
        if 'gyroADC[0]' in df.columns: potential_gyro_cols.append('gyroADC[0]')
        if 'gyroADC[1]' in df.columns: potential_gyro_cols.append('gyroADC[1]')
        if 'gyroADC[2]' in df.columns: potential_gyro_cols.append('gyroADC[2]')
        
        # If still no columns found, look for any column starting with 'gyro'
        if not potential_gyro_cols:
            potential_gyro_cols = [col for col in df.columns if col.lower().startswith('gyro')]
        
        if not potential_gyro_cols:
            return {"spectral": {"error": "No gyro data found in log."}}

        # Map to standard names for the output
        gyro_col_map = {}
        for col in potential_gyro_cols:
            if 'roll' in col.lower() or '[0]' in col:
                gyro_col_map[col] = 'gyroRoll'
            elif 'pitch' in col.lower() or '[1]' in col:
                gyro_col_map[col] = 'gyroPitch'
            elif 'yaw' in col.lower() or '[2]' in col:
                gyro_col_map[col] = 'gyroYaw'
            else:
                # If can't categorize, use the original name
                gyro_col_map[col] = col

        # Determine sampling rate
        if sampling_rate is None:
            time_unit = metadata.get('analysis_info', {}).get('time_unit', 'unknown') if metadata else 'unknown'
            if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)) or np.issubdtype(df.index.dtype, np.number):
                time_diffs = pd.Series(df.index).diff().dropna()
                median_diff = np.median(time_diffs) if len(time_diffs) > 0 else 0

                # Convert median diff to seconds based on unit
                if median_diff > 1e-9: # Avoid division by zero
                    if time_unit == 'us' or (time_unit == 'unknown' and np.issubdtype(df.index.dtype, np.number) and df.index.max() > 1e6):
                        median_diff_sec = median_diff / 1_000_000.0
                        logger.debug(f"Detected time unit as microseconds, median diff: {median_diff_sec:.6f} s")
                    elif isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
                        median_diff_sec = median_diff.total_seconds()
                        logger.debug(f"Detected time unit as Timedelta/Datetime, median diff: {median_diff_sec:.6f} s")
                    else: # Assume seconds if unknown numeric or index type
                        median_diff_sec = median_diff
                        logger.debug(f"Assuming time unit as seconds, median diff: {median_diff_sec:.6f} s")

                    sampling_rate = 1.0 / median_diff_sec
                    results['estimated_sampling_rate_hz'] = round(sampling_rate)
                else:
                    sampling_rate = None # Cannot determine rate
                    results['estimated_sampling_rate_hz'] = "N/A (Zero time diff)"
            else: # Fallback if index is not time-like
                sampling_rate = None
                results['estimated_sampling_rate_hz'] = "N/A (Non-time index)"

            # If rate couldn't be estimated, fallback or error
            if sampling_rate is None:
                 # Try getting looptime from metadata as a fallback
                 looptime_us = metadata.get('other_settings', {}).get('looptime')
                 if looptime_us and looptime_us > 0:
                      sampling_rate = 1_000_000.0 / looptime_us
                      results['estimated_sampling_rate_hz'] = f"{round(sampling_rate)} (from looptime)"
                      logger.info(f"Using sampling rate from metadata looptime: {sampling_rate:.1f} Hz")
                 else:
                      logger.error("Could not determine sampling rate from index or metadata.")
                      return {"spectral": {"error": "Could not determine sampling rate."}}

        if not sampling_rate or sampling_rate <= 0:
            return {"spectral": {"error": f"Invalid sampling rate calculated: {sampling_rate}."}}

        results['actual_sampling_rate_used'] = sampling_rate
        logger.info(f"Using sampling rate for FFT: {sampling_rate:.2f} Hz")

        # Perform FFT for each gyro axis
        for col in potential_gyro_cols:
            if col not in df or df[col].isnull().all():
                results['spectra'][gyro_col_map[col]] = {"error": f"Column '{col}' missing or all NaN."}
                continue
            signal_data = df[col].dropna().values
            n = len(signal_data)
            if n < 10: # Need a minimum number of points for meaningful FFT
                results['spectra'][gyro_col_map[col]] = {"error": "Not enough data points (<10)."}
                continue

            try:
                # Apply Hann window to reduce spectral leakage
                windowed_signal = signal_data * signal.windows.hann(n)
                # Compute FFT
                fft_result = fft(windowed_signal)
                # Compute frequencies (only positive half needed)
                freqs = fftfreq(n, 1 / sampling_rate)[:n // 2]
                # Compute magnitude (one-sided spectrum)
                fft_magnitude = np.abs(fft_result[:n // 2])
                # Normalize magnitude (except DC component)
                if n > 0:
                    fft_magnitude[1:] *= (2 / n) # Multiply by 2/N for amplitude
                    fft_magnitude[0] /= n # DC component is just divided by N
                else:
                    fft_magnitude = np.array([]) # Should not happen if n>=10

                axis_results = {}
                # Find dominant peaks
                try:
                    if fft_magnitude.size > 1 and np.max(fft_magnitude[1:]) > 1e-9: # Ignore DC for peak finding
                        # Calculate distance based on frequency resolution (e.g., 5 Hz separation)
                        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1
                        peak_distance = max(1, int(5 / freq_resolution)) if freq_resolution > 0 else 5

                        peaks, _ = signal.find_peaks(fft_magnitude,
                                                      height=np.max(fft_magnitude[1:]) * 0.05, # Relative height threshold (5%)
                                                      distance=peak_distance)
                        if peaks.size > 0:
                            peak_freqs = freqs[peaks]
                            peak_mags = fft_magnitude[peaks]
                            sorted_indices = np.argsort(peak_mags)[::-1] # Sort by magnitude descending
                            top_n = 5
                            axis_results["dominant_peaks_hz_mag"] = list(zip(np.round(peak_freqs[sorted_indices][:top_n], 1),
                                                                              np.round(peak_mags[sorted_indices][:top_n], 3)))
                        else: axis_results["dominant_peaks_hz_mag"] = []
                    else: axis_results["dominant_peaks_hz_mag"] = []
                except Exception as e:
                    logger.warning(f"Peak finding failed for {col}: {e}")
                    axis_results["dominant_peaks_error"] = f"Peak finding failed: {e}"

                # Calculate average energy in frequency bands
                bands = {
                    "prop_wash_band_(0-20Hz)": (0, 20),
                    "mid_freq_band_(20-80Hz)": (20, 80),
                    "high_freq_band_(80-200Hz)": (80, 200),
                    "noise_floor_band_(>200Hz)": (200, sampling_rate / 2)
                }
                band_energy = {}
                if freqs.size > 0:
                    for name, (low, high) in bands.items():
                        # Ensure high freq doesn't exceed Nyquist
                        high = min(high, sampling_rate / 2)
                        if low >= high: continue # Skip invalid bands
                        mask = (freqs >= low) & (freqs < high)
                        # Use mean magnitude in the band
                        band_energy[name] = round(np.mean(fft_magnitude[mask]), 3) if np.any(mask) else 0.0
                axis_results["band_avg_magnitude"] = band_energy

                # Downsample data for storage/plotting if too large
                max_points = 1000
                if len(freqs) > max_points:
                    indices = np.linspace(0, len(freqs) - 1, max_points, dtype=int)
                    axis_results["frequencies_hz"] = np.round(freqs[indices], 2).tolist()
                    axis_results["magnitude"] = np.round(fft_magnitude[indices], 4).tolist()
                elif len(freqs) > 0:
                    axis_results["frequencies_hz"] = np.round(freqs, 2).tolist()
                    axis_results["magnitude"] = np.round(fft_magnitude, 4).tolist()
                else:
                    axis_results["frequencies_hz"] = []
                    axis_results["magnitude"] = []

                results['spectra'][gyro_col_map[col]] = axis_results
            except Exception as fft_err:
                 logger.error(f"Error during FFT processing for {col}: {fft_err}", exc_info=True)
                 results['spectra'][gyro_col_map[col]] = {"error": f"FFT calculation failed: {fft_err}"}

        return {"spectral": results}
    
    def _extract_performance_metrics(self, analysis_results: dict) -> dict:
        """
        Extract key performance metrics from the analysis results.
        
        Args:
            analysis_results (dict): Full analysis results dictionary
        
        Returns:
            dict: Condensed performance metrics
        """
        performance = {}
        # Extract PID tracking error metrics
        pid_results = analysis_results.get('pid', {})
        performance.update({
            'roll_mae': pid_results.get('roll_tracking_error_mae'),
            'pitch_mae': pid_results.get('pitch_tracking_error_mae'),
            'yaw_mae': pid_results.get('yaw_tracking_error_mae')
        })
        # Extract motor saturation and imbalance
        motors_results = analysis_results.get('motors', {})
        performance.update({
            'motor_saturation': motors_results.get('motor_saturation_pct_overall'),
            'motor_imbalance': motors_results.get('motor_imbalance_pct_of_avg')
        })
        # Extract spectral analysis metrics
        spectral_results = analysis_results.get('spectral', {}).get('spectra', {})
        for axis, data in spectral_results.items():
            if isinstance(data, dict) and 'band_avg_magnitude' in data:
                performance[f'{axis.lower()}_noise_level'] = data['band_avg_magnitude'].get('noise_floor_band_(>200Hz)', 0)
        return performance

    def save_log_analysis(self, log_id: str, metadata: dict, analysis_results: dict, recommendations: dict) -> bool:
        """
        Saves log analysis results to the tuning history JSON file.
        
        Args:
            log_id (str): Unique identifier for the log
            metadata (dict): Metadata about the log
            analysis_results (dict): Detailed analysis results
            recommendations (dict): Tuning recommendations
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Read existing history
            history = self.get_tuning_history()
            if isinstance(history, dict) and 'error' in history:
                history = []
            # Prepare entry for saving
            entry = {
                'log_id': log_id,
                'timestamp': datetime.now().isoformat(),
                'filename': metadata.get('filename', 'Unknown'),
                'metadata': metadata,
                'performance': self._extract_performance_metrics(analysis_results),
                'analysis_results': analysis_results,
                'recommendations': recommendations
            }
            # Add new entry to history
            history.append(entry)
            # Limit history to last 100 entries
            history = history[-100:]
            # Write back to file
            with self.tuning_history_path.open('w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)
            logger.info(f"Saved log analysis for {log_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving log analysis: {e}")
            return False       
    def analyze_rc_commands(self, df: pd.DataFrame) -> dict:
        """
        Analyzes RC command characteristics and pilot input style.
        """
        logger.debug("Analyzing RC Commands...")
        results = {}

        rc_columns = {
            'roll': next((col for col in df.columns if col.lower() in ['rccommand[0]', 'rccommands[0]', 'rccommandroll']), None),
            'pitch': next((col for col in df.columns if col.lower() in ['rccommand[1]', 'rccommands[1]', 'rccommandpitch']), None),
            'yaw': next((col for col in df.columns if col.lower() in ['rccommand[2]', 'rccommands[2]', 'rccommandyaw']), None),
            'throttle': next((col for col in df.columns if col.lower() in ['rccommand[3]', 'rccommands[3]', 'rccommandthrottle']), None)
        }

        # Check if RC columns exist and are numeric
        if not any(col is not None for col in rc_columns.values()):
            results['error'] = "No RC command columns found"
            return results

        # Analyze each RC axis
        for axis, col in rc_columns.items():
            if col is not None and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                rc_data = df[col].dropna()
                if not rc_data.empty:
                    # Basic statistics
                    results[f'{axis}_rc_mean'] = rc_data.mean()
                    results[f'{axis}_rc_std'] = rc_data.std()
                    results[f'{axis}_rc_range'] = rc_data.max() - rc_data.min()

                    # Rate of change analysis
                    rc_diff = rc_data.diff().dropna()
                    results[f'{axis}_rc_rate_of_change_mean'] = rc_diff.mean()
                    results[f'{axis}_rc_rate_of_change_std'] = rc_diff.std()
                    results[f'{axis}_rc_rate_of_change_max'] = rc_diff.abs().max()

        # Pilot Style Assessment
        if rc_columns['roll'] and rc_columns['pitch']:
            roll_data = df[rc_columns['roll']].dropna()
            pitch_data = df[rc_columns['pitch']].dropna()

            # Combine roll and pitch rate of change
            combined_diff = pd.concat([roll_data.diff(), pitch_data.diff()]).dropna()

            # Smoothness assessment (lower std of rate of change indicates smoother flying)
            smoothness_std = combined_diff.std()
            if smoothness_std < 0.5:
                results['pilot_smoothness_assessment'] = 'Very Smooth'
            elif smoothness_std < 1.0:
                results['pilot_smoothness_assessment'] = 'Smooth'
            elif smoothness_std < 2.0:
                results['pilot_smoothness_assessment'] = 'Moderate'
            else:
                results['pilot_smoothness_assessment'] = 'Aggressive'

            # Aggression assessment (95th percentile of absolute rate of change)
            aggression_95 = combined_diff.abs().quantile(0.95)
            if aggression_95 < 1.0:
                results['pilot_aggression_assessment'] = 'Low'
            elif aggression_95 < 2.0:
                results['pilot_aggression_assessment'] = 'Moderate'
            else:
                results['pilot_aggression_assessment'] = 'High'

        # Center Focus (time spent near center)
        if rc_columns['roll'] and rc_columns['pitch']:
            roll_center_focus = (df[rc_columns['roll']].abs() < 0.1 * df[rc_columns['roll']].max()).mean() * 100
            pitch_center_focus = (df[rc_columns['pitch']].abs() < 0.1 * df[rc_columns['pitch']].max()).mean() * 100
            results['pilot_center_focus_pct'] = (roll_center_focus + pitch_center_focus) / 2

        return results
    
    def compare_logs(self, log_id1: str, log_id2: str) -> dict:
            """Compares metadata and key performance metrics between two saved logs."""
            logger.debug(f"Comparing logs: {log_id1} vs {log_id2}")
            try:
                logs_db = {}
                if self.logs_db_path.exists():
                    with self.logs_db_path.open('r', encoding='utf-8') as f: # Specify encoding
                        try: logs_db = json.load(f)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON from {self.logs_db_path}")
                            return {"error": "Log database file is corrupted or empty."}
                else:
                    logger.error(f"Log database file not found at {self.logs_db_path}")
                    return {"error": "Log database file not found."}

                if log_id1 not in logs_db or log_id2 not in logs_db:
                    missing = [lid for lid in [log_id1, log_id2] if lid not in logs_db]
                    logger.warning(f"Log ID(s) not found in database: {', '.join(missing)}")
                    return {"error": f"Log ID(s) not found in database: {', '.join(missing)}"}

                log1_data = logs_db[log_id1]
                log2_data = logs_db[log_id2]

                # Ensure timestamps are comparable (handle potential string format)
                ts1 = pd.to_datetime(log1_data.get("timestamp"), errors='coerce')
                ts2 = pd.to_datetime(log2_data.get("timestamp"), errors='coerce')

                # Determine which log is 'older' (log1) and 'newer' (log2) based on timestamp
                # If timestamps are invalid or equal, keep original order
                if ts1 is not None and ts2 is not None and ts1 > ts2:
                    log1_data, log2_data = log2_data, log1_data # Swap if log1 is newer
                    log_id1, log_id2 = log_id2, log_id1
                    ts1, ts2 = ts2, ts1
                    logger.debug(f"Swapped logs based on timestamp: {log_id1} (older) vs {log_id2} (newer)")

                comparison = {
                    "log1_id": log_id1,
                    "log2_id": log_id2,
                    "log1_timestamp": log1_data.get("timestamp"),
                    "log2_timestamp": log2_data.get("timestamp"),
                    "setting_changes": {}, # Combined PID/Filter/Rate changes
                    "performance_changes": {}
                }

                # Helper to compare nested dictionaries (like metadata categories)
                def compare_nested_dicts(dict1, dict2, prefix=""):
                    changes = {}
                    # Ensure inputs are dictionaries
                    d1 = dict1 if isinstance(dict1, dict) else {}
                    d2 = dict2 if isinstance(dict2, dict) else {}
                    all_keys = set(d1.keys()) | set(d2.keys())

                    for key in all_keys:
                        full_key = f"{prefix}{key}" if prefix else key
                        old_val = d1.get(key)
                        new_val = d2.get(key)
                        # Compare serializable forms to handle potential type differences (e.g., list vs tuple)
                        if make_serializable(old_val) != make_serializable(new_val):
                            changes[full_key] = {"old": old_val, "new": new_val}
                    return changes

                # Compare key settings categories (PIDs, Filters, Rates)
                settings1 = {
                    **log1_data.get("metadata", {}).get("pids",{}),
                    **log1_data.get("metadata", {}).get("filters",{}),
                    **log1_data.get("metadata", {}).get("rates",{})
                }
                settings2 = {
                    **log2_data.get("metadata", {}).get("pids",{}),
                    **log2_data.get("metadata", {}).get("filters",{}),
                    **log2_data.get("metadata", {}).get("rates",{})
                }
                comparison["setting_changes"] = compare_nested_dicts(settings1, settings2)


                # Compare key performance metrics
                # Helper function to safely get nested performance metrics and convert to float
                def get_perf_metric(log_data, metric_path, default=None):
                    # Use the saved performance dict if available, otherwise analysis_results
                    perf_source = log_data.get("performance", log_data.get("analysis_results", {}))
                    val = perf_source
                    for key in metric_path:
                        if isinstance(val, dict):
                            val = val.get(key)
                        else:
                            return default # Path doesn't exist
                    # Try converting to float for comparison, return default on failure
                    try:
                        return float(val) if val is not None else default
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert performance metric {metric_path} value '{val}' to float.")
                        return default

                # Define metrics to compare and whether lower values are better
                # NOTE: Ensure these paths match the structure saved in _extract_performance_metrics / save_log_analysis
                metrics_to_compare = {
                    "Roll MAE": (['roll_mae'], True), # Lower is better
                    "Pitch MAE": (['pitch_mae'], True),
                    "Yaw MAE": (['yaw_mae'], True),
                    # Add step response metrics if they are reliably saved in performance dict
                    #"Roll Overshoot (%)": (['roll_overshoot'], True),
                    #"Pitch Overshoot (%)": (['pitch_overshoot'], True),
                    #"Roll Rise Time (ms)": (['roll_rise_time'], True),
                    #"Pitch Rise Time (ms)": (['pitch_rise_time'], True),
                    "Motor Saturation (%)": (['motor_saturation'], True),
                    "Motor Imbalance (%)": (['motor_imbalance'], True),
                    # Add Noise Level comparison if it's reliably saved
                    # "Noise Level": (['noise_level'], True)
                }

                perf_changes_calc = {}
                improvements = 0
                regressions = 0

                for metric_name, (path, lower_is_better) in metrics_to_compare.items():
                    old_val = get_perf_metric(log1_data, path)
                    new_val = get_perf_metric(log2_data, path)

                    # Only compare if both values are valid numbers
                    if old_val is not None and new_val is not None and np.isfinite(old_val) and np.isfinite(new_val):
                        change = new_val - old_val
                        pct_change = None
                        improvement = None

                        # Calculate percentage change
                        if abs(old_val) > 1e-9: # Avoid division by zero
                            pct_change = (change / abs(old_val)) * 100
                        elif new_val != 0: # Handle change from zero
                            pct_change = float('inf') * np.sign(change)

                        # Determine if it's an improvement (only if change is significant)
                        # Using a small relative threshold (e.g., 0.1%) to avoid flagging tiny changes
                        rel_change_threshold = 0.001
                        if abs(change) > 1e-6 and (abs(old_val) < 1e-9 or abs(change / old_val) > rel_change_threshold):
                            if (lower_is_better and change < 0) or (not lower_is_better and change > 0):
                                improvement = True
                                improvements += 1
                            else:
                                improvement = False
                                regressions += 1

                        perf_changes_calc[metric_name] = {
                            "old": round(old_val, 2),
                            "new": round(new_val, 2),
                            "change": round(change, 2),
                            "percent_change": round(pct_change, 1) if pct_change is not None else None,
                            "improvement": improvement
                        }
                    # Handle cases where one or both values are missing/invalid
                    elif old_val is not None or new_val is not None:
                        perf_changes_calc[metric_name] = {
                            "old": round(old_val, 2) if old_val is not None and np.isfinite(old_val) else "N/A",
                            "new": round(new_val, 2) if new_val is not None and np.isfinite(new_val) else "N/A",
                            "change": "N/A",
                            "percent_change": None,
                            "improvement": None
                        }


                comparison["performance_changes"] = perf_changes_calc
                # Determine overall verdict based on counts
                if improvements > regressions: verdict = "Improved"
                elif regressions > improvements: verdict = "Regressed"
                else: verdict = "Mixed/Unchanged"
                comparison["overall_assessment"] = {"improvements": improvements, "regressions": regressions, "verdict": verdict}

                logger.info(f"Comparison complete: {log_id1} vs {log_id2}. Verdict: {verdict}")
                return comparison

            except Exception as e:
                logger.error(f"Error comparing logs {log_id1} and {log_id2}: {e}", exc_info=True)
                return {"error": f"Error comparing logs: {e}"}


    def analyze_altitude_power(self, df: pd.DataFrame) -> dict:
        """
        Analyzes altitude and power-related data from the log.
        """
        logger.debug("Analyzing Altitude and Power...")
        results = {}

        # Find altitude column
        alt_col = next((col for col in df.columns if col.lower() in ['altitudebaro', 'baroalt']), None)
        
        # Find voltage and current columns
        voltage_col = next((col for col in df.columns if col.lower() in ['vbatlatest', 'voltage']), None)
        current_col = next((col for col in df.columns if col.lower() in ['amperagelatest', 'current']), None)

        # Altitude analysis
        if alt_col and alt_col in df.columns and pd.api.types.is_numeric_dtype(df[alt_col]):
            alt_data = df[alt_col].dropna()
            if not alt_data.empty:
                results['altitude_mean'] = alt_data.mean()
                results['altitude_max'] = alt_data.max()
                results['altitude_min'] = alt_data.min()
                results['altitude_std'] = alt_data.std()

                # Detect climb/descend rates
                alt_diff = alt_data.diff()
                results['max_climb_rate'] = alt_diff[alt_diff > 0].max()
                results['max_descend_rate'] = alt_diff[alt_diff < 0].min()

        # Power analysis
        if voltage_col and voltage_col in df.columns and pd.api.types.is_numeric_dtype(df[voltage_col]):
            voltage_data = df[voltage_col].dropna()
            results['voltage_mean'] = voltage_data.mean()
            results['voltage_min'] = voltage_data.min()
            results['voltage_max'] = voltage_data.max()

        if current_col and current_col in df.columns and pd.api.types.is_numeric_dtype(df[current_col]):
            current_data = df[current_col].dropna()
            results['current_mean'] = current_data.mean()
            results['current_max'] = current_data.max()

            # Power calculation (if both voltage and current are available)
            if 'voltage_mean' in results:
                results['power_mean'] = results['voltage_mean'] * results['current_mean']
                results['power_max'] = results['voltage_max'] * results['current_max']

        return results

    def analyze_rc_vs_gyro(self, df: pd.DataFrame, metadata: dict) -> dict:
        """
        Analyzes RC command vs Gyro response latency and correlation.
        """
        logger.debug("Analyzing RC Command vs Gyro Latency...")
        results = {}

        # Find RC and Gyro columns
        rc_columns = {
            'roll': next((col for col in df.columns if col.lower() in ['rccommand[0]', 'rccommands[0]', 'rccommandroll']), None),
            'pitch': next((col for col in df.columns if col.lower() in ['rccommand[1]', 'rccommands[1]', 'rccommandpitch']), None),
            'yaw': next((col for col in df.columns if col.lower() in ['rccommand[2]', 'rccommands[2]', 'rccommandyaw']), None)
        }

        gyro_columns = {
            'roll': next((col for col in df.columns if col.lower() in ['gyroroll', 'gyroadc[0]']), None),
            'pitch': next((col for col in df.columns if col.lower() in ['gyropitch', 'gyroadc[1]']), None),
            'yaw': next((col for col in df.columns if col.lower() in ['gyroyaw', 'gyroadc[2]']), None)
        }

        # Estimate sampling rate for time-based calculations
        time_unit = metadata.get('analysis_info', {}).get('time_unit', 'us')
        sampling_rate = 1000  # Default to 1000 Hz

        # Calculate time conversion factor
        time_conversion = 1_000_000 if time_unit == 'us' else 1

        for axis in ['roll', 'pitch', 'yaw']:
            rc_col = rc_columns[axis]
            gyro_col = gyro_columns[axis]

            if rc_col and gyro_col and rc_col in df.columns and gyro_col in df.columns:
                rc_data = df[rc_col].dropna()
                gyro_data = df[gyro_col].dropna()

                # Align data
                common_index = rc_data.index.intersection(gyro_data.index)
                rc_data = rc_data.loc[common_index]
                gyro_data = gyro_data.loc[common_index]

                # Calculate RC command and Gyro rates
                rc_rate = rc_data.diff()
                gyro_rate = gyro_data.diff()

                # Cross-correlation to estimate response latency
                try:
                    from scipy import signal
                    # Normalize signals
                    rc_rate_norm = (rc_rate - rc_rate.mean()) / rc_rate.std()
                    gyro_rate_norm = (gyro_rate - gyro_rate.mean()) / gyro_rate.std()

                    # Compute cross-correlation
                    correlations = signal.correlate(rc_rate_norm, gyro_rate_norm, mode='full')
                    lags = signal.correlation_lags(len(rc_rate_norm), len(gyro_rate_norm), mode='full')

                    # Find peak correlation and corresponding lag
                    peak_corr_index = np.argmax(np.abs(correlations))
                    peak_lag = lags[peak_corr_index]

                    # Convert lag to milliseconds
                    lag_ms = abs(peak_lag) / sampling_rate * 1000
                    results[f'{axis}_lag_ms'] = round(lag_ms, 2)
                except Exception as e:
                    logger.warning(f"Latency calculation failed for {axis} axis: {e}")
                    results[f'{axis}_lag_ms'] = "N/A"

        return results



    # --- Plotting Functions ---
    # (Plotting functions remain the same as previous versions)
    # ... plot_pid_tracking, plot_motor_output, plot_spectral_analysis, etc. ...
    def plot_pid_tracking(self, df: pd.DataFrame) -> go.Figure:
        """Generates PID tracking plot (Gyro vs Setpoint)."""
        logger.debug("Generating PID tracking plot...")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Roll Axis", "Pitch Axis", "Yaw Axis"))
        time_axis = df.index
        # Convert time axis to seconds if it's numeric (microseconds)
        if np.issubdtype(time_axis.dtype, np.number):
             time_axis_display = time_axis / 1_000_000.0
             xaxis_title = "Time (s)"
        else: # Assumes DatetimeIndex or similar
             time_axis_display = time_axis
             xaxis_title = "Time"

        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            # Find columns case-insensitively
            gyro_col = next((c for c in df.columns if c.lower() == f'gyro{axis.lower()}'), None)
            setpoint_col = next((c for c in df.columns if c.lower() == f'setpoint{axis.lower()}'), None)

            if gyro_col and gyro_col in df.columns:
                fig.add_trace(go.Scatter(x=time_axis_display, y=df[gyro_col],
                                         mode='lines',
                                         name=f'Gyro {axis}', line=dict(width=1)),
                              row=i+1, col=1)
            if setpoint_col and setpoint_col in df.columns:
                fig.add_trace(go.Scatter(x=time_axis_display, y=df[setpoint_col],
                                         mode='lines',
                                         name=f'Setpoint {axis}', line=dict(dash='dash', width=1.5)),
                              row=i+1, col=1)
            fig.update_yaxes(title_text="Rate (°/s)", row=i+1, col=1)

        fig.update_layout(title="PID Tracking: Gyro vs Setpoint", height=600, legend_title_text='Trace')
        fig.update_xaxes(title_text=xaxis_title, row=3, col=1)
        return fig

    def plot_motor_output(self, df: pd.DataFrame) -> go.Figure:
        """Generates motor output plot."""
        logger.debug("Generating motor output plot...")
        motor_cols = sorted([col for col in df.columns if col.lower().startswith('motor') and col[5:].isdigit()])
        if not motor_cols:
            return go.Figure().update_layout(title="Motor Output (No Data)")

        time_axis = df.index
        # Convert time axis to seconds if it's numeric (microseconds)
        if np.issubdtype(time_axis.dtype, np.number):
             time_axis_display = time_axis / 1_000_000.0
             xaxis_title = "Time (s)"
        else:
             time_axis_display = time_axis
             xaxis_title = "Time"

        fig = go.Figure()
        for col in motor_cols:
            fig.add_trace(go.Scatter(x=time_axis_display, y=df[col], mode='lines',
                                     name=col, line=dict(width=1)))
        fig.update_layout(title="Motor Outputs", xaxis_title=xaxis_title,
                          yaxis_title="Motor Output Value", height=400, legend_title_text='Motor')
        return fig

    def plot_motor_saturation(self, motor_results: dict) -> go.Figure:
        """Generates a bar chart showing per-motor saturation percentage."""
        logger.debug("Generating Motor Saturation plot...")

        saturation_per_motor = motor_results.get('motor_saturation_pct_per_motor', {})
        saturation_overall = motor_results.get('motor_saturation_pct_overall')

        if not saturation_per_motor or not isinstance(saturation_per_motor, dict):
            return go.Figure().update_layout(title="Per-Motor Saturation (No Data)")

        motors = list(saturation_per_motor.keys())
        sat_values = list(saturation_per_motor.values())

        fig = go.Figure(data=[go.Bar(x=motors, y=sat_values, name='Saturation %')])

        title_text = "Per-Motor Saturation (%)"
        if saturation_overall is not None:
            title_text += f"<br><sup>Overall Saturation: {saturation_overall:.2f}%</sup>"

        fig.update_layout(
            title=title_text,
            xaxis_title="Motor",
            yaxis_title="Saturation Percentage (%)",
            yaxis_range=[0, 100], # Set y-axis to 0-100%
            height=350,
            bargap=0.2
        )
        return fig

    def plot_throttle_distribution(self, motor_results: dict) -> go.Figure:
        """Generates a pie chart showing throttle distribution."""
        logger.debug("Generating Throttle Distribution plot...")

        distribution = motor_results.get('throttle_distribution_pct', {})

        if not distribution or not isinstance(distribution, dict):
            return go.Figure().update_layout(title="Throttle Distribution (No Data)")

        labels = list(distribution.keys())
        values = list(distribution.values())

        # Ensure values sum reasonably close to 100, handle potential rounding issues
        if not (99 < sum(values) < 101):
            logger.warning(f"Throttle distribution values do not sum near 100%: {values}")
            # Optionally try to normalize if sum is non-zero
            # total = sum(values)
            # if total > 0: values = [(v / total) * 100 for v in values]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, pull=[0.05 if l=='0-25%' else 0 for l in labels])]) # Pull lowest slice slightly

        title_text = "Throttle Distribution (% Time)"
        avg_throttle = motor_results.get('avg_throttle')
        if avg_throttle is not None:
            title_text += f"<br><sup>Average Throttle Command: {avg_throttle:.1f}</sup>"

        fig.update_layout(
            title=title_text,
            height=350,
            showlegend=True
        )
        return fig

    def plot_motor_balance(self, motor_results: dict) -> go.Figure:
        """Generates a bar chart showing average motor outputs and imbalance."""
        logger.debug("Generating Motor Balance plot...")

        averages = motor_results.get('motor_averages', {})
        imbalance_std = motor_results.get('motor_imbalance_std_dev')
        imbalance_pct = motor_results.get('motor_imbalance_pct') # Use the correct key if different

        if not averages or not isinstance(averages, dict):
            return go.Figure().update_layout(title="Motor Average Output (No Data)")

        motors = list(averages.keys())
        avg_values = list(averages.values())

        fig = go.Figure(data=[go.Bar(x=motors, y=avg_values, name='Average Output')])

        title_text = "Motor Average Output"
        if imbalance_std is not None:
            title_text += f"<br><sup>Imbalance StdDev: {imbalance_std:.2f}"
            if imbalance_pct is not None:
                title_text += f" ({imbalance_pct:.1f}%)" # Adjust key if needed
            title_text += "</sup>"


        fig.update_layout(
            title=title_text,
            xaxis_title="Motor",
            yaxis_title="Average Output Value",
            height=350,
            bargap=0.2
        )
        return fig


    def plot_spectral_analysis(self, spectral_results: dict) -> go.Figure:
        """Generates spectral analysis FFT plot."""
        logger.debug("Generating spectral analysis plot...")
        spectra = spectral_results.get('spectra', {})
        if not spectra or all('error' in v for v in spectra.values()):
            return go.Figure().update_layout(title="Spectral Analysis (No Data)")

        valid_axes_data = {k: v for k, v in spectra.items() if isinstance(v,dict) and 'error' not in v and v.get("frequencies_hz")}
        num_axes = len(valid_axes_data)
        if num_axes == 0:
            return go.Figure().update_layout(title="Spectral Analysis (No Valid Data)")

        fig = make_subplots(rows=num_axes, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{col} Spectrum" for col in valid_axes_data])

        plot_row = 1
        max_freq = 0
        for col, axis_data in valid_axes_data.items():
            freqs = axis_data.get("frequencies_hz", [])
            mags = axis_data.get("magnitude", [])
            if freqs and mags:
                 fig.add_trace(go.Scatter(x=freqs, y=mags, mode='lines',
                                          name=col, line=dict(width=1)),
                               row=plot_row, col=1)
                 fig.update_yaxes(title_text="Magnitude", type="log", row=plot_row, col=1) # Use log scale
                 if freqs: max_freq = max(max_freq, freqs[-1])
                 plot_row += 1

        fig.update_layout(title="Gyro Spectral Analysis (FFT)", height=250 * num_axes, showlegend=False)
        # Set reasonable default x-axis range, e.g., up to 500 Hz or Nyquist
        sampling_rate_used = spectral_results.get('actual_sampling_rate_used', 1000) # Default 1kHz if not found
        max_plot_freq = min(500, sampling_rate_used / 2, max_freq if max_freq > 0 else 500)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max_plot_freq], row=num_axes, col=1)
        return fig

    def plot_throttle_freq_heatmap(self, spectral_results: dict) -> go.Figure:
        """Generates throttle vs frequency heatmap (requires pre-computed data)."""
        logger.debug("Generating throttle vs frequency heatmap...")
        heatmap_data = spectral_results.get("throttle_freq_heatmap") # Assuming this key exists if calculated
        if not heatmap_data or not all(k in heatmap_data for k in ["magnitude_matrix", "frequency_bins_hz", "throttle_bins"]):
            logger.warning("Throttle vs Frequency heatmap data not found or incomplete in spectral results.")
            return go.Figure().update_layout(title="Throttle vs Frequency Heatmap (No Data)")
        try:
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data["magnitude_matrix"],
                x=heatmap_data["frequency_bins_hz"],
                y=heatmap_data["throttle_bins"],
                colorscale='Viridis',
                colorbar=dict(title='Magnitude')
            ))
            fig.update_layout(
                title=f"{heatmap_data.get('gyro_axis', 'Gyro')} Frequency vs. Throttle Heatmap",
                xaxis_title="Frequency (Hz)", yaxis_title="Normalized Throttle", height=500
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return go.Figure().update_layout(title=f"Throttle vs Frequency Heatmap (Error: {e})")



    def plot_3d_flight(self, df: pd.DataFrame) -> go.Figure:
        """Generates a comprehensive 3D flight path plot using multiple data sources."""
        logger.debug("Generating comprehensive 3D flight tracking plot...")
        
        # Potential coordinate and altitude sources
        gps_cols = [
            'gpsCartesianCoords[0]', 'gpsCartesianCoords[1]', 'gpsCartesianCoords[2]',
            'gpsDistance', 'gpsHomeAzimuth'
        ]
        altitude_cols = ['baroAlt', 'gpsCartesianCoords[2]']
        heading_cols = ['heading[0]', 'heading[1]', 'heading[2]']

        # Initialize plot data containers
        x, y, z = None, None, None
        color_data, colorbar_title, title, scene = None, None, None, None

        # Priority 1: GPS Cartesian Coordinates
        gps_coord_cols = ['gpsCartesianCoords[0]', 'gpsCartesianCoords[1]', 'gpsCartesianCoords[2]']
        if all(col in df.columns for col in gps_coord_cols):
            x_raw = df[gps_coord_cols[0]].dropna()
            y_raw = df[gps_coord_cols[1]].dropna()
            z_raw = df[gps_coord_cols[2]].dropna()
            
            # Align data
            common_index = x_raw.index.intersection(y_raw.index).intersection(z_raw.index)
            if not common_index.empty:
                x, y, z = x_raw.loc[common_index], y_raw.loc[common_index], z_raw.loc[common_index]
                title = "3D Flight Tracking (GPS Cartesian)"
                scene = dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)")
                color_data = z
                colorbar_title = 'Altitude Z (m)'

        # Fallback: Altitude data
        if x is None:
            alt_col = next((col for col in altitude_cols if col in df.columns), None)
            if alt_col:
                time_axis = df.index
                if np.issubdtype(time_axis.dtype, np.number):
                    time_axis_display = time_axis / 1_000_000.0  # Time in seconds
                    xaxis_title = "Time (s)"
                else:
                    time_axis_display = time_axis
                    xaxis_title = "Time"

                x = time_axis_display
                y = df[alt_col].fillna(0) / 100.0  # Convert cm to m
                z = pd.Series(0, index=df.index)  # Use 0 for the 'Z' axis
                title = "Altitude vs Time"
                scene = dict(xaxis_title=xaxis_title, yaxis_title="Altitude (m)", zaxis_title="")
                color_data = y
                colorbar_title = 'Altitude (m)'

        # Handle cases with no valid data
        if x is None or len(x) == 0:
            logger.warning("No GPS or Altitude data found for 3D plot.")
            return go.Figure().update_layout(title="3D Flight Tracking (No Position Data)")

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, 
            mode='lines+markers',
            marker=dict(
                size=3, 
                color=color_data, 
                colorscale='Viridis',
                colorbar=dict(title=colorbar_title) if color_data is not None else None, 
                opacity=0.8
            ),
            line=dict(width=2)
        )])

        # Additional plot details
        fig.update_layout(
            title=title, 
            scene=scene, 
            height=600, 
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Attempt to add extra information to the plot
        extra_info_text = []
        
        # GPS Distance info
        gps_dist_col = 'gpsDistance'
        if gps_dist_col in df.columns:
            avg_dist = df[gps_dist_col].mean()
            max_dist = df[gps_dist_col].max()
            extra_info_text.append(f"Avg Distance: {avg_dist:.2f} m")
            extra_info_text.append(f"Max Distance: {max_dist:.2f} m")

        # Home Azimuth
        home_azimuth_col = 'gpsHomeAzimuth'
        if home_azimuth_col in df.columns:
            avg_azimuth = df[home_azimuth_col].mean()
            extra_info_text.append(f"Avg Home Azimuth: {avg_azimuth:.2f}°")

        # Heading information
        heading_info = []
        for col in heading_cols:
            if col in df.columns:
                avg_heading = df[col].mean()
                heading_info.append(f"{col}: {avg_heading:.2f}°")
        
        if heading_info:
            extra_info_text.extend(heading_info)

        # Add extra information as annotation
        if extra_info_text:
            extra_text = "<br>".join(extra_info_text)
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.05, y=0.9,
                text=extra_text,
                showarrow=False,
                font=dict(size=10),
                align="left"
            )

        return fig

    def plot_3d_coords_over_time(self, df: pd.DataFrame) -> go.Figure:
        """Generates plots of X, Y, Z coordinates over time."""
        logger.debug("Generating 3D Coordinates over Time plot...")

        # Potential coordinate sources
        coord_cols = {
            'X': 'gpsCartesianCoords[0]',
            'Y': 'gpsCartesianCoords[1]',
            'Z': 'gpsCartesianCoords[2]'
        }

        # Find available columns
        available_cols = {axis: col for axis, col in coord_cols.items() if col in df.columns}

        if not available_cols:
            return go.Figure().update_layout(title="3D Coordinates Over Time (No GPS Cartesian Data)")

        num_axes = len(available_cols)
        fig = make_subplots(rows=num_axes, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{axis} Coordinate" for axis in available_cols.keys()])

        # Prepare time axis
        time_axis = df.index
        if np.issubdtype(time_axis.dtype, np.number):
            time_axis_display = time_axis / 1_000_000.0
            xaxis_title = "Time (s)"
        else:
            time_axis_display = time_axis
            xaxis_title = "Time"

        current_row = 1
        for axis, col_name in available_cols.items():
            coord_data = df[col_name]
            fig.add_trace(
                go.Scatter(
                    x=time_axis_display,
                    y=coord_data,
                    mode='lines',
                    name=f'{axis} Coordinate',
                    line=dict(width=1)
                ),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text=f"{axis} (m)", row=current_row, col=1)
            current_row += 1

        fig.update_layout(title="GPS Cartesian Coordinates Over Time", height=200 * num_axes, showlegend=False)
        fig.update_xaxes(title_text=xaxis_title, row=num_axes, col=1)
        return fig

    
    def plot_gyro_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Generates a comprehensive plot showing gyro data characteristics."""
        logger.debug("Generating Gyro Analysis plot...")

        # Identify gyro columns
        gyro_cols = {
            'roll': next((col for col in df.columns if col in ['gyroADC[0]', 'gyroRoll', 'gyro_roll']), None),
            'pitch': next((col for col in df.columns if col in ['gyroADC[1]', 'gyroPitch', 'gyro_pitch']), None),
            'yaw': next((col for col in df.columns if col in ['gyroADC[2]', 'gyroYaw', 'gyro_yaw']), None)
        }

        # Prepare subplots
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=[f"{axis.capitalize()} Axis Gyro" for axis in ['roll', 'pitch', 'yaw']],
            shared_xaxes=True
        )

        # Time axis conversion
        time_axis = df.index
        if np.issubdtype(time_axis.dtype, np.number):
            time_axis_display = time_axis / 1_000_000.0  # Convert to seconds
            xaxis_title = "Time (s)"
        else:
            time_axis_display = time_axis
            xaxis_title = "Time"

        # Plot for each axis
        for i, (axis, col) in enumerate(gyro_cols.items(), 1):
            if col and col in df.columns:
                gyro_data = df[col]
                
                # Main gyro data line
                fig.add_trace(
                    go.Scatter(
                        x=time_axis_display, 
                        y=gyro_data, 
                        mode='lines', 
                        name=f'{axis.capitalize()} Gyro',
                        line=dict(color='blue', width=1)
                    ),
                    row=i, col=1
                )

                # Highlight anomalies
                anomaly_indices = np.abs(gyro_data - gyro_data.mean()) > (3 * gyro_data.std())
                if anomaly_indices.any():
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_display[anomaly_indices], 
                            y=gyro_data[anomaly_indices], 
                            mode='markers', 
                            name=f'{axis.capitalize()} Anomalies',
                            marker=dict(color='red', size=5),
                        ),
                        row=i, col=1
                    )

        # Update layout
        fig.update_layout(
            title="Gyro Data Analysis with Anomalies",
            height=600,
            showlegend=True
        )
        fig.update_xaxes(title_text=xaxis_title, row=3, col=1)
        fig.update_yaxes(title_text="Rate (°/s)", row=1, col=1)
        fig.update_yaxes(title_text="Rate (°/s)", row=2, col=1)
        fig.update_yaxes(title_text="Rate (°/s)", row=3, col=1)

        # Add summary annotations
        summary_text = []
        for axis, col in gyro_cols.items():
            if col and col in df.columns:
                gyro_data = df[col]
                summary_text.append(
                    f"{axis.capitalize()} Axis: "
                    f"Mean={gyro_data.mean():.2f}, "
                    f"Std={gyro_data.std():.2f}, "
                    f"Anomalies={sum(np.abs(gyro_data - gyro_data.mean()) > (3 * gyro_data.std()))} "
                    f"({sum(np.abs(gyro_data - gyro_data.mean()) > (3 * gyro_data.std())) / len(gyro_data) * 100:.2f}%)"
                )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.05, y=0.5,
            text="<br>".join(summary_text),
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
        return fig

    def plot_gyro_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Generates a heatmap visualization of gyro and RC command data.
        Focuses on intensity and variations across different axes.
        """
        logger.debug("Generating Gyro and RC Command Heatmap...")

        # Identify columns
        gyro_columns = {
            'roll': next((col for col in df.columns if col in ['gyroADC[0]', 'gyroRoll', 'gyro_roll']), None),
            'pitch': next((col for col in df.columns if col in ['gyroADC[1]', 'gyroPitch', 'gyro_pitch']), None),
            'yaw': next((col for col in df.columns if col in ['gyroADC[2]', 'gyroYaw', 'gyro_yaw']), None)
        }

        rc_columns = {
            'roll': next((col for col in df.columns if col in ['rcCommand[0]', 'rcCommands[0]']), None),
            'pitch': next((col for col in df.columns if col in ['rcCommand[1]', 'rcCommands[1]']), None),
            'yaw': next((col for col in df.columns if col in ['rcCommand[2]', 'rcCommands[2]']), None)
        }

        # Prepare data matrix
        data_matrix = []
        axes_labels = []

        # Add Gyro data
        for axis, col in gyro_columns.items():
            if col and col in df.columns:
                gyro_data = df[col]
                # Compute rolling window statistics
                rolling_mean = gyro_data.rolling(window=50, center=True).mean()
                rolling_std = gyro_data.rolling(window=50, center=True).std()
                data_matrix.append(rolling_std.fillna(0))
                axes_labels.append(f"Gyro {axis.capitalize()} Std")

        # Add RC Command data
        for axis, col in rc_columns.items():
            if col and col in df.columns:
                rc_data = df[col]
                # Compute rolling window statistics
                rolling_mean = rc_data.rolling(window=50, center=True).mean()
                rolling_std = rc_data.rolling(window=50, center=True).std()
                data_matrix.append(rolling_std.fillna(0))
                axes_labels.append(f"RC {axis.capitalize()} Std")

        # Convert to numpy for heatmap
        if not data_matrix:
            return go.Figure().update_layout(title="No Data for Heatmap")

        data_matrix = np.array(data_matrix)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=list(range(data_matrix.shape[1])),
            y=axes_labels,
            colorscale='Viridis',
            colorbar=dict(title='Standard Deviation')
        ))

        # Update layout
        fig.update_layout(
            title='Gyro and RC Command Variation Heatmap',
            xaxis_title='Time Steps',
            yaxis_title='Axes',
            height=400
        )

        # Add summary statistics annotation
        summary_text = []
        for label, data in zip(axes_labels, data_matrix):
            summary_text.append(
                f"{label}: Mean={np.mean(data):.2f}, Max={np.max(data):.2f}"
            )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.05, y=0.5,
            text="<br>".join(summary_text),
            showarrow=False,
            font=dict(size=10),
            align="left"
        )

        return fig

    def plot_power_altitude(self, df: pd.DataFrame) -> go.Figure:
            """Generates Power and Altitude plot."""
            logger.debug("Generating Power & Altitude plot...")

            # Find relevant columns using flexible matching
            alt_col = next((col for col in df.columns if col.lower() in ['altitudebaro', 'baroalt']), None)
            voltage_col = next((col for col in df.columns if col.lower() in ['vbatlatest', 'voltage', 'vbat']), None)
            current_col = next((col for col in df.columns if col.lower() in ['amperagelatest', 'current', 'amperage']), None)

            # Determine how many subplots are needed
            plot_cols = [col for col in [alt_col, voltage_col, current_col] if col and col in df.columns]
            plot_names = [name for name, col in zip(['Altitude', 'Voltage', 'Current'], [alt_col, voltage_col, current_col]) if col and col in df.columns]
            
            plot_count = len(plot_cols)
            if plot_count == 0:
                return go.Figure().update_layout(title="Power & Altitude (No Data Found)")

            fig = make_subplots(rows=plot_count, cols=1, shared_xaxes=True,
                                subplot_titles=plot_names)

            time_axis = df.index
            if np.issubdtype(time_axis.dtype, np.number):
                time_axis_display = time_axis / 1_000_000.0
                xaxis_title = "Time (s)"
            else:
                time_axis_display = time_axis
                xaxis_title = "Time"

            current_row = 1
            # Plot Altitude
            if alt_col and alt_col in df.columns:
                # Assuming altitude is in cm if 'baro' in name, convert to m
                alt_data = df[alt_col] / 100.0 if 'baro' in alt_col.lower() else df[alt_col]
                fig.add_trace(go.Scatter(x=time_axis_display, y=alt_data, mode='lines', name='Altitude'), row=current_row, col=1)
                fig.update_yaxes(title_text="Altitude (m)", row=current_row, col=1)
                current_row += 1

            # Plot Voltage
            if voltage_col and voltage_col in df.columns:
                # Assuming voltage is in cV (centivolts) if 'latest' in name, convert to V
                volt_data = df[voltage_col] / 100.0 if 'latest' in voltage_col.lower() else df[voltage_col]
                fig.add_trace(go.Scatter(x=time_axis_display, y=volt_data, mode='lines', name='Voltage'), row=current_row, col=1)
                fig.update_yaxes(title_text="Voltage (V)", row=current_row, col=1)
                current_row += 1

            # Plot Current
            if current_col and current_col in df.columns:
                # Assuming current is in cA (centiamps) if 'latest' in name, convert to A
                curr_data = df[current_col] / 100.0 if 'latest' in current_col.lower() else df[current_col]
                fig.add_trace(go.Scatter(x=time_axis_display, y=curr_data, mode='lines', name='Current'), row=current_row, col=1)
                fig.update_yaxes(title_text="Current (A)", row=current_row, col=1)
                current_row += 1

            fig.update_layout(title="Power & Altitude Data", height=200 * plot_count, showlegend=False)
            fig.update_xaxes(title_text=xaxis_title, row=plot_count, col=1)
            return fig

    def plot_rc_vs_gyro_response(self, df: pd.DataFrame) -> go.Figure:
        """Generates a plot comparing RC commands and gyro response."""
        logger.debug("Generating RC Command vs Gyro Response plot...")

        # Identify RC and Gyro columns
        rc_columns = {
            'roll': next((col for col in df.columns if col in ['rcCommand[0]', 'rcCommands[0]']), None),
            'pitch': next((col for col in df.columns if col in ['rcCommand[1]', 'rcCommands[1]']), None),
            'yaw': next((col for col in df.columns if col in ['rcCommand[2]', 'rcCommands[2]']), None)
        }

        gyro_columns = {
            'roll': next((col for col in df.columns if col in ['gyroADC[0]', 'gyroRoll']), None),
            'pitch': next((col for col in df.columns if col in ['gyroADC[1]', 'gyroPitch']), None),
            'yaw': next((col for col in df.columns if col in ['gyroADC[2]', 'gyroYaw']), None)
        }

        # Prepare subplots
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=[f"{axis.capitalize()} Axis RC vs Gyro" for axis in ['roll', 'pitch', 'yaw']],
            shared_xaxes=True
        )

        # Time axis conversion
        time_axis = df.index
        if np.issubdtype(time_axis.dtype, np.number):
            time_axis_display = time_axis / 1_000_000.0  # Convert to seconds
            xaxis_title = "Time (s)"
        else:
            time_axis_display = time_axis
            xaxis_title = "Time"

        # Plot for each axis
        for i, (axis, (rc_col, gyro_col)) in enumerate(zip(rc_columns.keys(), zip(rc_columns.values(), gyro_columns.values())), 1):
            if rc_col and gyro_col and rc_col in df.columns and gyro_col in df.columns:
                rc_data = df[rc_col]
                gyro_data = df[gyro_col]

                # RC Command line
                fig.add_trace(
                    go.Scatter(
                        x=time_axis_display, 
                        y=rc_data, 
                        mode='lines', 
                        name=f'{axis.capitalize()} RC Command',
                        line=dict(color='blue', width=1)
                    ),
                    row=i, col=1
                )

                # Gyro Response line
                fig.add_trace(
                    go.Scatter(
                        x=time_axis_display, 
                        y=gyro_data, 
                        mode='lines', 
                        name=f'{axis.capitalize()} Gyro Response',
                        line=dict(color='red', width=1, dash='dot')
                    ),
                    row=i, col=1
                )

        # Update layout
        fig.update_layout(
            title="RC Command vs Gyro Response",
            height=600,
            showlegend=True
        )
        fig.update_xaxes(title_text=xaxis_title, row=3, col=1)
        fig.update_yaxes(title_text="RC Command", row=1, col=1)
        fig.update_yaxes(title_text="RC Command", row=2, col=1)
        fig.update_yaxes(title_text="RC Command", row=3, col=1)

        return fig


    
    def identify_problem_patterns(self, analysis_results: dict, metadata: dict) -> list:
            """
            Identifies potential tuning and flight performance issues from analysis results.
            Returns a list of tuples containing (problem_name, details_dict)
            """
            logger.debug("Identifying problem patterns...")
            problem_patterns = []

            # PID Performance Analysis with self-tuning logic
            pid_results = analysis_results.get('pid', {})
            if pid_results:
                for axis in ['roll', 'pitch', 'yaw']:
                    mae_key = f'{axis}_tracking_error_mae'
                    if mae_key in pid_results:
                        mae = pid_results[mae_key]
                        # If Mean Absolute Error is high, attempt to optimize PID settings
                        if mae > 10:  # threshold; adjust as needed
                            # Get current PID gains from metadata (use defaults if not available)
                            current_P = metadata.get("pids", {}).get(f"p_{axis}", 40)  # default value example
                            current_D = metadata.get("pids", {}).get(f"d_{axis}", 20)  # default value example
                            # Run self-tuning optimization
                            rec_P, rec_D, opt_score = optimize_pid_for_axis(current_P, current_D, desired_overshoot=0.05, desired_rise_time=1.0)
                            problem_patterns.append((f"High {axis.capitalize()} Axis Tracking Error", {
                                "recommendation": f"Adjust {axis.capitalize()} PID terms to P={rec_P:.1f}, D={rec_D:.1f}",
                                "explanation": f"High Mean Absolute Error ({mae:.2f}°/s) indicates poor tracking. Your current settings are P={current_P} and D={current_D}. Simulation suggests that P={rec_P:.1f} and D={rec_D:.1f} provide a better response (optimization score: {opt_score:.2f}).",
                                "severity": 7.0,
                                "category": "PID Tuning",
                                "simulated": {
                                    "current": {"P": current_P, "D": current_D},
                                    "recommended": {"P": rec_P, "D": rec_D}
                                },
                                "commands": [
                                    f"set p_{axis} {rec_P:.1f}",
                                    f"set d_{axis} {rec_D:.1f}"
                                ]
                            }))

            # Existing Motor Analysis (unchanged)
            motors_results = analysis_results.get('motors', {})
            if motors_results:
                saturation = motors_results.get('motor_saturation_pct_overall', 0)
                if saturation > 30:
                    problem_patterns.append(("High Motor Saturation", {
                        "recommendation": "Adjust PID and Feed-Forward",
                        "explanation": f"Motors saturated {saturation:.1f}% of the time, indicating aggressive flight or over-tuned PIDs.",
                        "severity": 8.0,
                        "category": "Motor Performance",
                        "commands": [
                            "Reduce P and D terms",
                            "Adjust Feed-Forward gains",
                            "Check mechanical balance"
                        ]
                    }))
                imbalance = motors_results.get('motor_imbalance_pct_of_avg', 0)
                if imbalance > 10:
                    problem_patterns.append(("Motor Imbalance", {
                        "recommendation": "Check motor and prop balance",
                        "explanation": f"Motor output imbalance of {imbalance:.1f}% detected.",
                        "severity": 6.0,
                        "category": "Mechanical",
                        "commands": [
                            "Balance propellers",
                            "Check motor mounts",
                            "Verify motor RPM consistency"
                        ]
                    }))

            # Existing Spectral Analysis (unchanged)
            spectral_results = analysis_results.get('spectral', {}).get('spectra', {})
            for axis, data in spectral_results.items():
                if isinstance(data, dict) and 'band_avg_magnitude' in data:
                    noise_level = data['band_avg_magnitude'].get('noise_floor_band_(>200Hz)', 0)
                    if noise_level > 0.5:
                        problem_patterns.append((f"High {axis} Noise Floor", {
                            "recommendation": "Investigate Vibration Sources",
                            "explanation": f"Elevated high-frequency noise detected in {axis} axis.",
                            "severity": 5.0,
                            "category": "Vibration",
                            "commands": [
                                "Check prop balance",
                                "Inspect motor mounts",
                                "Add soft mounting",
                                "Reduce gyro lowpass filter cutoff"
                            ]
                        }))

            logger.debug(f"Identified {len(problem_patterns)} problem patterns.")
            return problem_patterns

    def generate_tuning_recommendations(self, analysis_results: dict, metadata: dict) -> dict:
        """
        Generates comprehensive tuning recommendations based on log analysis.
        
        Args:
            analysis_results (dict): Detailed analysis results from log processing
            metadata (dict): Metadata about the log and flight
        
        Returns:
            dict: Recommendations and overall flight assessment
        """
        recommendations = {}

        # Identify problem patterns
        problem_patterns = self.identify_problem_patterns(analysis_results, metadata)

        # Overall Flight Quality Assessment
        flight_quality_score = 1.0
        flight_assessment = {
            "flight_quality": flight_quality_score,
            "summary": "Good overall flight performance",
            "strengths": [],
            "weaknesses": []
        }

        # Adjust quality score based on identified problems
        for _, details in problem_patterns:
            severity = details.get('severity', 0)
            flight_quality_score = max(0, flight_quality_score - (severity / 10))

            # Categorize problems into strengths/weaknesses
            weakness = f"{details.get('recommendation', 'Unspecified Issue')} ({details.get('category', 'General')})"
            flight_assessment['weaknesses'].append(weakness)

        # Add some default strengths if no major issues
        if not flight_assessment['weaknesses']:
            flight_assessment['strengths'].append("Stable flight characteristics")
            flight_assessment['strengths'].append("Well-tuned PID controller")
            flight_assessment['strengths'].append("Minimal noise and vibration")

        # Final assessment updates
        flight_assessment['flight_quality'] = round(flight_quality_score, 2)
        if flight_quality_score > 0.8:
            flight_assessment['summary'] = "Excellent flight performance"
        elif flight_quality_score > 0.6:
            flight_assessment['summary'] = "Good flight performance with minor tuning opportunities"
        elif flight_quality_score > 0.4:
            flight_assessment['summary'] = "Moderate flight performance, significant tuning needed"
        else:
            flight_assessment['summary'] = "Poor flight performance, extensive tuning required"

        # Compile final recommendations
        recommendations = {
            "flight_assessment": flight_assessment,
            "problem_patterns": problem_patterns
        }

        return recommendations

    # --- Full Analysis Wrapper ---
    def full_log_analysis(self, file_path: str) -> dict:
        """Performs the complete analysis workflow for a single log file."""
        logger.info(f"--- Starting Full Analysis for {os.path.basename(file_path)} ---")
        analysis_results = {}
        df = None
        metadata = {} # Initialize metadata dict
        try:
            # 1. Read and Parse Header/Metadata
            lines = self._read_log_file(file_path)
            metadata_lines, header_line, data_start_index = self._find_header_and_data(lines)
            metadata = self.parse_metadata(metadata_lines) # This now initializes analysis_info
            metadata['filename'] = os.path.basename(file_path)

            # 2. Parse and Prepare Data
            df_raw = self.parse_data(header_line, lines[data_start_index:])
            df = self.prepare_data(df_raw, metadata) # Pass metadata

            # 3. Data Quality Check
            # Ensure diagnose_data_quality method exists
            if not hasattr(self, 'diagnose_data_quality'):
                 raise AttributeError("'BetaflightLogAnalyzer' object has no attribute 'diagnose_data_quality'")
            data_quality = self.diagnose_data_quality(df)
            analysis_results['data_quality'] = data_quality
            if data_quality["quality_score"] < 0.3:
                logger.warning(f"Analysis aborted: {metadata['filename']}, Quality Score: {data_quality['quality_score']:.2f}")
                return {"error": "Log data quality too poor for reliable analysis.",
                        "diagnostics": data_quality, "metadata": metadata, "df": None}

            # 4. Perform Detailed Analyses (pass metadata where needed)
            logger.debug("Running detailed analyses...")
            analysis_results.update(self.analyze_pid_performance(df, metadata))
            analysis_results.update(self.analyze_motors(df))
            analysis_results.update(self.perform_spectral_analysis(df, metadata=metadata))
            analysis_results['gyro_accel'] = self.analyze_gyro_accel(df)
            analysis_results.update(self.analyze_rc_commands(df))
            analysis_results['alt_power'] = self.analyze_altitude_power(df)
            analysis_results.update(self.analyze_rc_vs_gyro(df, metadata))

            # 5. Generate Recommendations
            # Ensure generate_tuning_recommendations method exists
            if not hasattr(self, 'generate_tuning_recommendations'):
                 raise AttributeError("'BetaflightLogAnalyzer' object has no attribute 'generate_tuning_recommendations'")
            # Ensure identify_problem_patterns method exists (called by generate_tuning_recommendations)
            if not hasattr(self, 'identify_problem_patterns'):
                 raise AttributeError("'BetaflightLogAnalyzer' object has no attribute 'identify_problem_patterns'")

            recommendations = self.generate_tuning_recommendations(analysis_results, metadata)

            # 6. Save Results
            log_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{metadata['filename']}"
            logger.debug(f"Attempting to save analysis results for log ID: {log_id}")
            save_success = self.save_log_analysis(log_id, metadata, analysis_results, recommendations)
            if not save_success:
                logger.warning(f"Failed to save analysis results for log {log_id}.")

            logger.info(f"--- Finished Full Analysis for {metadata['filename']} ---")
            # Return results including the DataFrame for plotting
            return {
                "log_id": log_id,
                "metadata": metadata,
                "analysis_results": analysis_results,
                "recommendations": recommendations,
                "data_quality": data_quality,
                "df": df
            }
        # Handle specific expected errors first
        except FileNotFoundError as e:
             logger.error(f"File not found error during analysis: {e}")
             return {"error": str(e), "metadata": metadata, "df": None}
        except ValueError as e: # Catch header/data parsing errors, empty df errors
             logger.error(f"Value error during analysis (check log format/content): {e}")
             return {"error": str(e), "metadata": metadata, "df": df if df is not None else None}
        except IOError as e:
             logger.error(f"IO error during analysis (check file permissions/readability): {e}")
             return {"error": str(e), "metadata": metadata, "df": None}
        # Catch AttributeError specifically (like missing methods)
        except AttributeError as e:
             logger.error(f"Attribute error during analysis (likely code issue): {e}", exc_info=True)
             error_msg = f"An internal error occurred: {type(e).__name__} - {e}. Please check the code/installation."
             return {"error": error_msg, "metadata": metadata, "df": df if df is not None else None}
        # Catch any other unexpected errors
        except Exception as e:
            logger.error(f"Unexpected error during full_log_analysis for {file_path}: {e}", exc_info=True)
            error_msg = f"An unexpected error occurred during analysis: {type(e).__name__} - {e}. Check logs for details."
            # Return df if it was successfully prepared before the error
            return {"error": error_msg, "metadata": metadata, "df": df if df is not None else None}


# --- Streamlit UI (main function) ---
def main():
    # Session state initialization
    if "current_file_path" not in st.session_state: st.session_state.current_file_path = None
    if "current_file_name" not in st.session_state: st.session_state.current_file_name = None
    if "analysis_output" not in st.session_state: st.session_state.analysis_output = None
    if "comparison_results" not in st.session_state: st.session_state.comparison_results = None
    if "batch_results" not in st.session_state: st.session_state.batch_results = []
    if "last_analyzed_file" not in st.session_state: st.session_state.last_analyzed_file = None
    if "selected_log_ids" not in st.session_state: st.session_state.selected_log_ids = []

    # Instantiate analyzer once (can be used by sidebar and main area)
    analyzer = BetaflightLogAnalyzer()

    # --- Sidebar ---
    st.sidebar.image("https://raw.githubusercontent.com/wiki/betaflight/betaflight/images/betaflight_logo_outline_blue.png", width=100)
    st.sidebar.header("📁 Log File Selection")
    upload_method = st.sidebar.radio("Select files from:", ["Upload Single File", "Local Directory"], key="upload_method", index=0)
    log_files = [] # List to store paths for batch analysis if local dir is used
    selected_file_path_str = None # Path of the currently selected file for single analysis

    if upload_method == "Upload Single File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a Betaflight log (.bbl, .bfl, .csv, .log, .txt)",
            type=["bbl", "bfl", "csv", "log", "txt"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        if uploaded_file is not None:
            temp_dir = pathlib.Path(tempfile.gettempdir()) / "streamlit_uploads_bf"
            temp_dir.mkdir(exist_ok=True)
            selected_file_path = temp_dir / uploaded_file.name
            selected_file_path_str = str(selected_file_path)
            try:
                with open(selected_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")

                if selected_file_path_str != st.session_state.get('current_file_path'):
                    st.session_state.current_file_path = selected_file_path_str
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.analysis_output = None
                    st.session_state.comparison_results = None
                    st.session_state.batch_results = []
                    st.session_state.last_analyzed_file = None
                    logger.debug(f"New file uploaded and selected: {selected_file_path_str}")

            except Exception as e:
                st.sidebar.error(f"Error saving uploaded file: {e}")
                logger.error(f"Error saving uploaded file {uploaded_file.name}", exc_info=True)
                selected_file_path_str = None

    else: # Local Directory
        log_dir_str = st.sidebar.text_input("Log directory path:", ".", key="log_dir")
        log_dir = pathlib.Path(log_dir_str)
        if log_dir.is_dir():
            try:
                found_files = [f for f in os.listdir(log_dir)
                               if (log_dir / f).is_file() and f.lower().endswith(('.bbl', '.bfl', '.csv', '.log', '.txt'))]
                found_files.sort(key=lambda f: (log_dir / f).stat().st_mtime, reverse=True)

                if found_files:
                    selected_file_name = st.sidebar.selectbox(
                        "Select a log file for single analysis:",
                        found_files,
                        key="log_selector",
                        index=None,
                        placeholder="Choose a log..."
                    )
                    if selected_file_name:
                        new_selected_path_str = str(log_dir / selected_file_name)
                        selected_file_path_str = new_selected_path_str
                        if st.session_state.get('current_file_path') != new_selected_path_str:
                            st.session_state.current_file_path = new_selected_path_str
                            st.session_state.current_file_name = selected_file_name
                            st.session_state.analysis_output = None
                            st.session_state.comparison_results = None
                            st.session_state.batch_results = []
                            st.session_state.last_analyzed_file = None
                            logger.debug(f"New file selected from directory: {new_selected_path_str}")
                            st.sidebar.info(f"Selected for single analysis: {selected_file_name}")
                        else:
                            st.session_state.current_file_path = new_selected_path_str
                            st.session_state.current_file_name = selected_file_name

                    log_files = [str(log_dir / f) for f in found_files]
                else:
                    st.sidebar.info("No log files found in the specified directory.")
            except Exception as e:
                st.sidebar.error(f"Error accessing directory '{log_dir}': {e}")
                logger.error(f"Error accessing directory {log_dir}", exc_info=True)
        else:
            st.sidebar.warning("Invalid directory path.")

    # Update session state if a file was selected/uploaded
    if selected_file_path_str is not None:
         st.session_state.current_file_path = selected_file_path_str

    st.sidebar.divider()

    # --- Analysis Actions ---
    st.sidebar.header("⚙️ Analysis Actions")
    analyze_button_disabled = st.session_state.current_file_path is None
    debug_mode = st.sidebar.checkbox("Enable debug mode", value=False)

    if st.sidebar.button("Analyze Selected Log", disabled=analyze_button_disabled, key="analyze_single"):
        if st.session_state.current_file_path:
            logger.info(f"Analyze button clicked for: {st.session_state.current_file_path}")
            if (st.session_state.current_file_path == st.session_state.last_analyzed_file and
                st.session_state.analysis_output and "error" not in st.session_state.analysis_output):
                st.sidebar.info("This log has already been analyzed. Results are displayed.")
                st.rerun()
            else:
                with st.spinner(f"Analyzing {st.session_state.get('current_file_name', 'log file')}..."):
                    # Use the analyzer instance created earlier
                    analysis_output = analyzer.full_log_analysis(st.session_state.current_file_path)
                    st.session_state.analysis_output = analysis_output
                    st.session_state.comparison_results = None
                    st.session_state.batch_results = []

                if analysis_output and "error" not in analysis_output:
                    st.session_state.last_analyzed_file = st.session_state.current_file_path
                    st.sidebar.success("Analysis Complete!")
                    st.rerun()
                else:
                    st.session_state.last_analyzed_file = None
                    error_msg = analysis_output.get('error', 'Unknown error') if isinstance(analysis_output, dict) else "Analysis failed unexpectedly."
                    st.sidebar.error(f"Analysis Failed: {error_msg[:200]}...")
                    logger.error(f"Analysis failed for {st.session_state.current_file_path}: {error_msg}")

        else:
            st.sidebar.warning("Please select or upload a log file first.")

    # Batch Analysis
    if upload_method == "Local Directory" and log_files:
        st.sidebar.subheader("Batch Analysis")
        max_batch = min(50, len(log_files))
        default_batch = min(5, max_batch)
        num_batch = st.sidebar.number_input(
            f"Number of recent logs for batch analysis (max {max_batch}):",
            min_value=1, max_value=max_batch, value=default_batch, key="num_batch"
            )
        if st.sidebar.button(f"Analyze {num_batch} Recent Logs", key="analyze_batch"):
            st.session_state.batch_results = []
            st.session_state.analysis_output = None
            st.session_state.comparison_results = None
            recent_logs_paths = log_files[:num_batch]
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            batch_success_count = 0
            # Use the same analyzer instance for batch
            logger.info(f"Starting batch analysis for {num_batch} logs.")

            for i, file_path in enumerate(recent_logs_paths):
                file_name = os.path.basename(file_path)
                status_text.text(f"Analyzing {i+1}/{num_batch}: {file_name}...")
                try:
                    result = analyzer.full_log_analysis(file_path)
                    st.session_state.batch_results.append(result)
                    if "error" not in result:
                        batch_success_count += 1
                    else:
                        logger.warning(f"Batch analysis failed for {file_name}: {result.get('error')}")
                except Exception as batch_e:
                    logger.error(f"Exception during batch analysis for {file_name}", exc_info=True)
                    st.session_state.batch_results.append({"error": f"Exception: {batch_e}", "metadata": {"filename": file_name}})
                finally:
                    progress_bar.progress((i+1)/len(recent_logs_paths))

            status_text.text(f"Batch analysis complete. Analyzed {batch_success_count}/{len(recent_logs_paths)} logs successfully.")
            st.sidebar.success("Batch Analysis Finished!")
            logger.info(f"Batch analysis finished. Success: {batch_success_count}/{len(recent_logs_paths)}")
            st.rerun()

    st.sidebar.divider()

    # --- History & Comparison ---
    st.sidebar.header("📊 History & Comparison")
    try:
        # Use the same analyzer instance
        if not analyzer.tuning_history_path.exists():
            st.sidebar.info("Tuning history file not found. Analyze logs to create it.")
            history = []
        else:
            history = analyzer.get_tuning_history()

        if isinstance(history, list) and history:
            log_options_map = {}
            for entry in reversed(history):
                log_id = entry.get('log_id')
                timestamp_str = entry.get('timestamp', '1970-01-01T00:00:00')
                filename = entry.get('filename', log_id.split('_', 1)[-1] if log_id and '_' in log_id else log_id or 'Unknown Log')
                try:
                    dt_obj = datetime.fromisoformat(timestamp_str)
                    display_str = f"{dt_obj.strftime('%Y-%m-%d %H:%M')} - {filename}"
                except ValueError:
                    display_str = f"{timestamp_str} - {filename}"

                if log_id:
                    log_options_map[display_str] = log_id

            st.sidebar.info(f"Found {len(history)} historical log analyses.")

            selected_log_display = st.sidebar.multiselect(
                "Select 2 logs from history to compare:",
                list(log_options_map.keys()), max_selections=2, key="log_compare_select"
            )
            st.session_state.selected_log_ids = [log_options_map[disp] for disp in selected_log_display if disp in log_options_map]

            compare_button_disabled = len(st.session_state.selected_log_ids) != 2
            if st.sidebar.button("Compare Selected Logs", disabled=compare_button_disabled, key="compare_logs"):
                if len(st.session_state.selected_log_ids) == 2:
                    log_id1, log_id2 = st.session_state.selected_log_ids
                    logger.info(f"Compare button clicked for: {log_id1} vs {log_id2}")
                    with st.spinner("Comparing logs..."):
                        st.session_state.comparison_results = analyzer.compare_logs(log_id1, log_id2)
                        st.session_state.analysis_output = None
                        st.session_state.batch_results = []

                    if st.session_state.comparison_results and "error" not in st.session_state.comparison_results:
                        st.sidebar.success("Comparison Complete!")
                        st.rerun()
                    else:
                        error_msg = st.session_state.comparison_results.get('error', 'Unknown error') if isinstance(st.session_state.comparison_results, dict) else "Comparison failed."
                        st.sidebar.error(f"Comparison Failed: {error_msg}")
                        logger.error(f"Comparison failed between {log_id1} and {log_id2}: {error_msg}")
                else:
                    st.sidebar.warning("Please select exactly two logs to compare.")
        elif isinstance(history, dict) and 'error' in history:
             st.sidebar.error(f"Error loading history: {history['error']}")
        else:
            if analyzer.tuning_history_path.exists() and not isinstance(history, list):
                 st.sidebar.warning("Tuning history file is empty or corrupted.")
            else:
                 st.sidebar.info("No historical log analyses found to compare.") # Handles empty list and file not found
    except Exception as e:
        st.sidebar.error(f"Error accessing tuning history: {e}")
        logger.error("Error loading tuning history or setting up comparison", exc_info=True)


    # --- Main Content Area ---

    # Display Single Analysis Results
    if (st.session_state.analysis_output and
        isinstance(st.session_state.analysis_output, dict) and
        "error" not in st.session_state.analysis_output):

        results = st.session_state.analysis_output
        st.header(f"🔎 Analysis Results for: `{results.get('metadata', {}).get('filename', 'N/A')}`")

        analysis_data = results.get("analysis_results", {})
        df_current = results.get("df")

        # Dynamically create tabs based on available data
        available_tabs = ["📈 Summary & Recs"]
        if results.get("metadata"): available_tabs.append("📝 Metadata")
        if analysis_data.get("pid"): available_tabs.append("📉 PID Perf.")
        if analysis_data.get("motors"): available_tabs.append("⚙️ Motors")
        if analysis_data.get("spectral"): available_tabs.append("🔊 Spectral")
        if analysis_data.get("gyro_accel"): available_tabs.append("⚖️ Gyro/Accel")
        if analysis_data.get("rc_commands"): available_tabs.append("🎮 RC/Pilot")
        if analysis_data.get("alt_power"): available_tabs.append("⚡ Power/Alt")
        if analysis_data.get("rc_gyro_latency"): available_tabs.append("🛰️ RC Latency")
        if df_current is not None and isinstance(df_current, pd.DataFrame):
             has_gps_coords = all(col in df_current.columns for col in ['gpsCartesianCoords[0]', 'gpsCartesianCoords[1]', 'gpsCartesianCoords[2]'])
             has_baro = next((c for c in df_current.columns if c.lower() == 'altitudebaro'), None) is not None
             if has_gps_coords or has_baro:
                 available_tabs.append("🛩️ 3D Flight")
             available_tabs.append("📄 Raw Data")

        available_tabs.append("💾 History")

        tabs = st.tabs(available_tabs)
        tab_map = {name: tab for name, tab in zip(available_tabs, tabs)}

        # --- Summary & Recs Tab ---
        with tab_map["📈 Summary & Recs"]:
            rec_data = results.get("recommendations", {})
            if rec_data and isinstance(rec_data, dict):
                assessment = rec_data.get("flight_assessment", {})
                if assessment and isinstance(assessment, dict):
                    quality_score = assessment.get("flight_quality", 0.0)
                    st.metric("Overall Flight Quality Score", f"{quality_score:.2f} / 1.0")
                    st.progress(quality_score)
                    st.markdown(f"**Assessment:** {assessment.get('summary', 'N/A')}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**👍 Strengths:**")
                        strengths = assessment.get("strengths", [])
                        st.markdown("\n".join([f"- {s}" for s in strengths]) if strengths else "_None identified_")
                    with col2:
                        st.markdown("**👎 Weaknesses:**")
                        weaknesses = assessment.get("weaknesses", [])
                        st.markdown("\n".join([f"- {w}" for w in weaknesses]) if weaknesses else "_None identified_")
                    st.divider()
                else:
                    st.warning("Flight assessment data missing or invalid.")

                st.subheader("🔧 Tuning Suggestions & Diagnosis")
                # Use the structure defined in generate_tuning_recommendations
                problem_patterns_list = rec_data.get("problem_patterns", [])
                if problem_patterns_list and isinstance(problem_patterns_list, list):
                     for i, pattern_tuple in enumerate(problem_patterns_list):
                         if isinstance(pattern_tuple, (list, tuple)) and len(pattern_tuple) == 2:
                             problem_name, details = pattern_tuple
                             if isinstance(details, dict):
                                 cat = f" ({details.get('category', '')})" if details.get('category') else ""
                                 expander_title = f"{details.get('recommendation', problem_name)}{cat} (Severity: {details.get('severity', 0):.1f}/10)"
                                 with st.expander(expander_title):
                                     st.markdown(f"**Explanation:** {details.get('explanation', 'N/A')}")
                                     commands = details.get('commands')
                                     if commands and isinstance(commands, list):
                                         st.code("\n".join(commands), language="bash")
                             else:
                                  st.warning(f"Invalid format for problem details at index {i}: {details}")
                         else:
                              st.warning(f"Invalid format for problem pattern tuple at index {i}: {pattern_tuple}")

                else:
                    # Check if flight quality was low despite no patterns matched
                    fq_score = assessment.get("flight_quality", 1.0)
                    if fq_score < 0.6:
                        st.info("No specific tuning problem patterns were automatically identified, but the overall flight assessment suggests room for improvement. Review the performance tabs (PID, Spectral, Motors) for potential issues.")
                    else:
                        st.info("No specific problems detected requiring tuning changes based on current thresholds.")
            else:
                st.warning("Could not generate recommendations or recommendations data is invalid.")

            st.divider()
            st.subheader("Data Quality")
            data_quality = results.get("data_quality", {})
            if data_quality and isinstance(data_quality, dict):
                dq_score = data_quality.get('quality_score', 0.0)
                st.metric("Data Quality Score", f"{dq_score:.2f} / 1.0")
                st.progress(dq_score)
                st.markdown(f"**Diagnosis:** {data_quality.get('summary', 'N/A')}")
                diagnosis_list = data_quality.get("diagnosis", [])
                if diagnosis_list and isinstance(diagnosis_list, list):
                    with st.expander("Data Quality Issues Found"):
                        for issue in diagnosis_list:
                            st.warning(issue)
            else:
                 st.warning("Data quality information not available.")

        # --- Metadata Tab ---
        if "📝 Metadata" in tab_map:
            with tab_map["📝 Metadata"]:
                st.subheader("Log File Metadata")
                metadata = results.get("metadata", {})
                if metadata and isinstance(metadata, dict):
                    try:
                        meta_disp = make_serializable(metadata)
                        st.markdown(f"**Filename:** `{meta_disp.get('filename', 'N/A')}`")
                        st.markdown(f"**BF Version:** {meta_disp.get('firmware', {}).get('betaflight_version', 'N/A')}")
                        st.markdown(f"**Target:** {meta_disp.get('firmware', {}).get('firmware_target', 'N/A')}")
                        st.markdown(f"**Board:** {meta_disp.get('firmware', {}).get('board_name', 'N/A')}") # Often under firmware in logs
                        st.markdown(f"**Craft Name:** {meta_disp.get('firmware', {}).get('craft_name', 'N/A')}")
                        st.markdown(f"**Analyzed:** {meta_disp.get('analysis_info', {}).get('analysis_timestamp', 'N/A')}")
                        st.markdown(f"**Time Unit:** `{meta_disp.get('analysis_info', {}).get('time_unit', 'N/A')}`")

                        for cat_key, cat_name in [('pids', 'PID Values'), ('rates', 'Rates'),
                                                ('filters', 'Filters'), ('features', 'Features'),
                                                ('other_settings', 'Other Settings')]:
                            cat_data = meta_disp.get(cat_key)
                            if cat_data and isinstance(cat_data, dict):
                                with st.expander(cat_name):
                                    st.json(cat_data)
                        remaining_keys = {k:v for k,v in meta_disp.items() if k not in ['filename', 'firmware', 'hardware', 'pids', 'rates', 'filters', 'features', 'other_settings', 'analysis_info']}
                        if remaining_keys:
                            with st.expander("Other Metadata"):
                                st.json(remaining_keys)
                    except Exception as meta_e:
                        st.error(f"Error displaying metadata: {meta_e}")
                        st.json(metadata)
                else:
                    st.warning("Metadata not available or invalid.")

        # --- PID Performance Tab ---
        if "📉 PID Perf." in tab_map:
             with tab_map["📉 PID Perf."]:
                st.subheader("PID Tracking Performance")
                pid_results = analysis_data.get('pid', {})
                if isinstance(pid_results, dict) and not pid_results.get("error_gyro") and not pid_results.get("error_overall"):
                    try:
                        if df_current is not None and isinstance(df_current, pd.DataFrame):
                            fig_pid = analyzer.plot_pid_tracking(df_current)
                            st.plotly_chart(fig_pid, use_container_width=True)
                        else:
                            st.warning("Processed DataFrame not available for PID plot.")

                        st.markdown("**Tracking Error Metrics (Mean Absolute Error):**")
                        cols = st.columns(3)
                        has_mae = False
                        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
                            mae = pid_results.get(f"{axis}_tracking_error_mae")
                            cols[i].metric(f"{axis.capitalize()} MAE (°/s)", f"{mae:.2f}" if mae is not None else "N/A")
                            if mae is not None: has_mae = True
                        if not has_mae: st.info("Mean Absolute Error could not be calculated (likely missing setpoint data).")

                        st.markdown("**Step Response Analysis (Median):**")
                        cols_step = st.columns(3)
                        has_step_response = False
                        for i, (metric_key, name) in enumerate(zip(["rise_time_ms", "overshoot_percent", "settling_time_ms"],
                                                                   ["Rise Time (ms)", "Overshoot (%)", "Settling Time (ms)"])):
                            with cols_step[i]:
                                st.markdown(f"**{name}**")
                                axis_has_data = False
                                for axis in ['roll', 'pitch', 'yaw']:
                                    step_res = pid_results.get(f"{axis}_step_response", {})
                                    value = step_res.get(metric_key) if isinstance(step_res, dict) else None
                                    st.markdown(f"- {axis.capitalize()}: {value:.1f}" if isinstance(value, (int, float)) else 'N/A')
                                    if value is not None: axis_has_data = True ; has_step_response = True
                                if not axis_has_data: st.caption("_N/A for all axes_")
                        if not has_step_response: st.info("Step response metrics could not be calculated (requires sufficient stick movements and setpoint data).")

                        with st.expander("Raw PID Analysis Data"):
                            st.json(make_serializable(pid_results))
                    except Exception as e:
                        st.error(f"Error displaying PID results: {e}")
                        st.json(make_serializable(pid_results))
                else:
                    error_msg = pid_results.get('error_gyro') or pid_results.get('error_overall') or 'No data or invalid format'
                    st.warning(f"Could not display PID performance. Error: {error_msg}")
                    if isinstance(pid_results, dict): st.json(make_serializable(pid_results))

        # --- Motors Tab ---
        if "⚙️ Motors" in tab_map:
              with tab_map["⚙️ Motors"]:
                st.subheader("Motor Output Analysis")
                motor_results = analysis_data.get('motors', {})
                if isinstance(motor_results, dict) and not motor_results.get("error_motors"):
                    try:
                        if df_current is not None and isinstance(df_current, pd.DataFrame):
                            # Motor Time Series
                            st.markdown("#### Motor Output Over Time")
                            fig_motor = analyzer.plot_motor_output(df_current)
                            st.plotly_chart(fig_motor, use_container_width=True)

                            # Balance Plot
                            st.markdown("---")
                            st.markdown("#### Motor Balance (Average Output)")
                            fig_motor_balance = analyzer.plot_motor_balance(motor_results)
                            if "No Data" not in fig_motor_balance.layout.title.text:
                                st.plotly_chart(fig_motor_balance, use_container_width=True)
                                imb_pct = motor_results.get('motor_imbalance_pct') # Adjust key if needed
                                if imb_pct is not None and imb_pct > 15:
                                    st.warning(f"High motor imbalance detected ({imb_pct:.1f}%). Check motor/prop condition, especially motors with higher average output.")
                            else:
                                st.info("Motor average data not available for balance plot.")

                             # Saturation Plot
                            st.markdown("---")
                            st.markdown("#### Per-Motor Saturation")
                            fig_motor_sat = analyzer.plot_motor_saturation(motor_results)
                            if "No Data" not in fig_motor_sat.layout.title.text:
                                st.plotly_chart(fig_motor_sat, use_container_width=True)
                                sat_overall = motor_results.get('motor_saturation_pct_overall')
                                if sat_overall is not None and sat_overall > 10:
                                    sat_per_motor = motor_results.get('motor_saturation_pct_per_motor', {})
                                    if sat_per_motor:
                                        try: # Handle potential non-numeric values if errors occur
                                           max_sat_motor = max(sat_per_motor, key=lambda k: sat_per_motor.get(k, 0) or 0)
                                           max_sat_val = sat_per_motor.get(max_sat_motor, 0)
                                           st.warning(f"Overall motor saturation ({sat_overall:.1f}%) is high. {max_sat_motor} shows particularly high saturation ({max_sat_val:.1f}%). Consider reducing overall gains.")
                                        except Exception as e_sat:
                                            logger.warning(f"Could not determine max saturating motor: {e_sat}")
                                            st.warning(f"Overall motor saturation ({sat_overall:.1f}%) is high. Check per-motor values. Consider reducing overall gains.")
                            else:
                                st.info("Per-motor saturation data not available.")

                            # Throttle Distribution Plot
                            st.markdown("---")
                            st.markdown("#### Throttle Distribution")
                            fig_throttle_dist = analyzer.plot_throttle_distribution(motor_results)
                            if "No Data" not in fig_throttle_dist.layout.title.text:
                                st.plotly_chart(fig_throttle_dist, use_container_width=True)
                                distribution = motor_results.get('throttle_distribution_pct', {})
                                if distribution.get('0-25%', 0) > 60:
                                     st.info("Flight spent considerable time (>60%) at low throttle (<25%). Step response and tuning analysis might be less representative of high-throttle behavior.")
                            else:
                                 st.info("Throttle distribution data not available.")

                        else:
                            st.warning("Processed DataFrame not available for motor plots.")

                        # Display raw metrics
                        with st.expander("Detailed Motor Metrics"):
                            st.json(make_serializable(motor_results))

                    except Exception as e:
                        st.error(f"Error displaying Motor results: {e}")
                        logger.error("Error in Motors tab display", exc_info=True)
                        st.json(make_serializable(motor_results))
                else:
                    error_msg = motor_results.get('error_motors', 'No data or invalid format')
                    st.warning(f"Could not display motor analysis. Error: {error_msg}")
                    if isinstance(motor_results, dict): st.json(make_serializable(motor_results))

        # --- Spectral Tab ---
        if "🔊 Spectral" in tab_map:
            with tab_map["🔊 Spectral"]:
                st.subheader("Gyro Spectral Analysis (FFT)")
                spectral_results = analysis_data.get('spectral', {})
                if isinstance(spectral_results, dict) and not spectral_results.get("error"):
                    try:
                        fig_spec = analyzer.plot_spectral_analysis(spectral_results)
                        st.plotly_chart(fig_spec, use_container_width=True)

                        st.markdown("---")
                        st.subheader("Throttle vs Frequency Heatmap")
                        st.info("Note: Heatmap requires specific pre-calculation. The `perform_spectral_analysis` function must be modified to generate data segmented by throttle.")
                        fig_heatmap = analyzer.plot_throttle_freq_heatmap(spectral_results)
                        if "No Data" not in fig_heatmap.layout.title.text and "Error" not in fig_heatmap.layout.title.text:
                             st.plotly_chart(fig_heatmap, use_container_width=True)
                        else:
                             st.warning(fig_heatmap.layout.title.text)

                        with st.expander("Raw Spectral Analysis Data"):
                            display_spec = make_serializable(spectral_results)
                            st.json(display_spec)
                    except Exception as e:
                        st.error(f"Error displaying Spectral results: {e}")
                        logger.error("Error displaying spectral plots", exc_info=True)
                        st.json(make_serializable(spectral_results))
                else:
                    error_msg = spectral_results.get('error', 'No data or invalid format')
                    st.warning(f"Could not display spectral analysis. Error: {error_msg}")
                    if isinstance(spectral_results, dict): st.json(make_serializable(spectral_results))

        # --- Gyro/Accel Tab ---
        if "⚖️ Gyro/Accel" in tab_map:
             with tab_map["⚖️ Gyro/Accel"]:
                st.subheader("Gyro & Accelerometer Details")
                gyro_accel_results = analysis_data.get('gyro_accel', {})
                if isinstance(gyro_accel_results, dict) and not gyro_accel_results.get("error_gyro"):
                    try:
                        if df_current is not None and isinstance(df_current, pd.DataFrame):
                            st.markdown("#### Gyro Data Over Time")
                            fig_gyro = analyzer.plot_gyro_analysis(df_current)
                            st.plotly_chart(fig_gyro, use_container_width=True)
                            # Optional: Add Accel plot here
                        else:
                            st.warning("Processed DataFrame not available for Gyro/Accel plots.")

                        with st.expander("Raw Gyro/Accel Metrics"):
                            st.json(make_serializable(gyro_accel_results))
                    except Exception as e:
                        st.error(f"Error displaying Gyro/Accel plots or data: {e}")
                        logger.error("Error in Gyro/Accel tab display", exc_info=True)
                        st.json(make_serializable(gyro_accel_results))
                else:
                    error_msg = gyro_accel_results.get('error_gyro', 'No data or invalid format')
                    st.warning(f"Could not display Gyro/Accel analysis. Error: {error_msg}")
                    if isinstance(gyro_accel_results, dict): st.json(make_serializable(gyro_accel_results))

        # --- RC/Pilot Tab ---
        if "🎮 RC/Pilot" in tab_map:
            with tab_map["🎮 RC/Pilot"]:
                st.subheader("RC Command & Pilot Analysis")
                rc_results = analysis_data.get('rc_commands', {})
                if isinstance(rc_results, dict) and not rc_results.get("error"):
                    st.subheader("Pilot Style Assessment")
                    col1, col2, col3 = st.columns(3)
                    smoothness = rc_results.get("pilot_smoothness_assessment", "N/A")
                    aggression = rc_results.get("pilot_aggression_assessment", "N/A")
                    center_focus = rc_results.get("pilot_center_focus_pct_avg") # Use average if available
                    col1.metric("Smoothness", smoothness, help="Based on standard deviation of stick rate of change.")
                    col2.metric("Aggression", aggression, help="Based on 95th percentile of stick rate of change.")
                    col3.metric("Center Focus (%)", f"{center_focus:.1f}%" if center_focus is not None else "N/A", help="Average percentage of time sticks are near center.")

                    if df_current is not None and isinstance(df_current, pd.DataFrame):
                         try:
                             st.markdown("---")
                             st.markdown("#### RC Commands Over Time")
                             fig_rc = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Roll Cmd", "Pitch Cmd", "Yaw Cmd", "Throttle Cmd"))
                             time_axis_display = df_current.index / 1_000_000.0 if np.issubdtype(df_current.index.dtype, np.number) else df_current.index
                             xaxis_title = "Time (s)" if np.issubdtype(df_current.index.dtype, np.number) else "Time"
                             rc_cols_plot = {'Roll': 'rcCommand[0]', 'Pitch': 'rcCommand[1]', 'Yaw': 'rcCommand[2]', 'Throttle': 'rcCommand[3]'} # Assuming standard names
                             for i, (axis, col) in enumerate(rc_cols_plot.items(), 1):
                                 # Check if column exists before trying to plot
                                 plot_col_name = next((c for c in df_current.columns if c.lower() == col.lower()), None)
                                 if plot_col_name:
                                      fig_rc.add_trace(go.Scatter(x=time_axis_display, y=df_current[plot_col_name], mode='lines', name=f'{axis} Cmd', line=dict(width=1)), row=i, col=1)
                                 fig_rc.update_yaxes(title_text="Command", row=i, col=1) # Update y-axis even if no data, for consistency
                             fig_rc.update_layout(height=600, showlegend=False)
                             fig_rc.update_xaxes(title_text=xaxis_title, row=4, col=1)
                             st.plotly_chart(fig_rc, use_container_width=True)
                         except Exception as e:
                             st.error(f"Error plotting RC commands: {e}")
                             logger.error("Error plotting RC Commands", exc_info=True)
                    else:
                        st.warning("Processed DataFrame not available for RC Command plots.")

                    with st.expander("Detailed RC Command Metrics"):
                        st.json(make_serializable(rc_results))
                else:
                    error_msg = rc_results.get('error', 'No data or invalid format')
                    st.warning(f"Could not display RC Command analysis. Error: {error_msg}")
                    if isinstance(rc_results, dict): st.json(make_serializable(rc_results))

        # --- Power/Alt Tab ---
        if "⚡ Power/Alt" in tab_map:
             with tab_map["⚡ Power/Alt"]:
                st.subheader("Power & Altitude Analysis")
                alt_power_results = analysis_data.get('alt_power', {})
                if alt_power_results and isinstance(alt_power_results, dict):
                    try:
                        if df_current is not None and isinstance(df_current, pd.DataFrame):
                            st.markdown("#### Data Over Time")
                            fig_power_alt = analyzer.plot_power_altitude(df_current)
                            st.plotly_chart(fig_power_alt, use_container_width=True)
                        else:
                            st.warning("Processed DataFrame not available for Power/Altitude plot.")

                        with st.expander("Raw Power/Altitude Metrics"):
                             st.json(make_serializable(alt_power_results))
                    except Exception as e:
                        st.error(f"Error displaying Power/Altitude plot or data: {e}")
                        logger.error("Error in Power/Alt tab display", exc_info=True)
                        st.json(make_serializable(alt_power_results))
                else:
                    st.info("No Power or Altitude data available for analysis.")

        # --- RC Latency Tab ---
        if "🛰️ RC Latency" in tab_map:
            with tab_map["🛰️ RC Latency"]:
                st.subheader("RC Command vs Gyro Response Latency")
                latency_results = analysis_data.get('rc_gyro_latency', {})
                if isinstance(latency_results, dict) and not latency_results.get("error"):
                    cols = st.columns(3)
                    has_latency_data = False
                    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
                        lag = latency_results.get(f"{axis.lower()}_lag_ms")
                        display_lag = f"{lag:.1f}" if isinstance(lag, (int, float)) else "N/A"
                        if isinstance(lag, (int, float)): has_latency_data = True
                        cols[i].metric(f"{axis.capitalize()} Lag (ms)", display_lag)
                    if has_latency_data:
                        st.markdown("_Note: Estimated latency based on peak correlation between RC command rate and Gyro rate. May be inaccurate with noisy data or insufficient stick movements._")
                    else:
                        st.info("Latency could not be calculated for any axis (requires sufficient stick movements).")
                else:
                    error_msg = latency_results.get('error', 'No data or missing columns')
                    st.warning(f"Could not display RC Latency analysis. Error: {error_msg}")

        # --- 3D Flight Tab ---
        if "🛩️ 3D Flight" in tab_map:
             with tab_map["🛩️ 3D Flight"]:
                st.subheader("3D Flight Path & Coordinates")
                if df_current is not None and isinstance(df_current, pd.DataFrame):
                    try:
                        st.markdown("#### Flight Path (3D View)")
                        fig_3d_path = analyzer.plot_3d_flight(df_current)
                        st.plotly_chart(fig_3d_path, use_container_width=True)

                        st.markdown("---")

                        st.markdown("#### Coordinates vs. Time")
                        fig_3d_time = analyzer.plot_3d_coords_over_time(df_current)
                        if "No GPS Cartesian Data" not in fig_3d_time.layout.title.text:
                            st.plotly_chart(fig_3d_time, use_container_width=True)
                        else:
                            st.info("GPS Cartesian coordinate data (gpsCartesianCoords[0/1/2]) not found for time-series plot.")

                    except Exception as e:
                        st.error(f"Error displaying 3D Flight Tracking plots: {e}")
                        logger.error("Error generating 3D plots", exc_info=True)
                else:
                     st.warning("Processed DataFrame not available for 3D plots.")


        # --- Raw Data Tab ---
        if "📄 Raw Data" in tab_map:
            with tab_map["📄 Raw Data"]:
                 if df_current is not None and isinstance(df_current, pd.DataFrame):
                    st.subheader("Processed Log Data")
                    st.markdown(f"Displaying the first 1000 rows of the processed data used for analysis. Total rows: {len(df_current)}")
                    st.dataframe(df_current.head(1000))

                    @st.cache_data
                    def convert_df_to_csv(df_to_convert: pd.DataFrame):
                        return df_to_convert.to_csv(index=True).encode('utf-8')

                    try:
                        csv_data = convert_df_to_csv(df_current)
                        download_filename = f"processed_{st.session_state.current_file_name or 'log'}.csv"
                        st.download_button(label="Download Processed Data as CSV",
                                           data=csv_data,
                                           file_name=download_filename,
                                           mime='text/csv')
                    except Exception as e:
                        st.error(f"Error preparing data for download: {e}")
                        logger.error("Error converting DataFrame to CSV for download", exc_info=True)
                 else:
                     st.warning("Processed DataFrame not available.")

        # --- History Tab ---
        if "💾 History" in tab_map:
            with tab_map["💾 History"]:
                st.header("📊 Tuning History")
                history = []
                try:
                    if analyzer.tuning_history_path.exists():
                        history = analyzer.get_tuning_history()

                    if isinstance(history, list) and len(history) > 1:
                        history_data = []
                        for entry in history:
                            perf_data = entry.get("performance", {})
                            if isinstance(perf_data, dict):
                                try:
                                    roll_err = pd.to_numeric(perf_data.get("roll_mae"), errors='coerce')
                                    pitch_err = pd.to_numeric(perf_data.get("pitch_mae"), errors='coerce')
                                    motor_sat = pd.to_numeric(perf_data.get("motor_saturation"), errors='coerce')
                                    motor_imb = pd.to_numeric(perf_data.get("motor_imbalance"), errors='coerce') # Use updated key if changed in save_to_history
                                    noise = pd.to_numeric(perf_data.get("noise_level"), errors='coerce')
                                    track_err_combined = np.nansum([abs(roll_err), abs(pitch_err)]) if pd.notna(roll_err) or pd.notna(pitch_err) else np.nan

                                    history_data.append({
                                        "Date": entry.get("timestamp", "Unknown"),
                                        "Log ID": entry.get("log_id", "Unknown"),
                                        "Filename": entry.get("filename", entry.get("log_id", "Unknown")), # Show filename
                                        "Tracking Error (Roll+Pitch Abs)": track_err_combined,
                                        "Motor Saturation %": motor_sat,
                                        "Motor Imbalance %": motor_imb,
                                        "Noise Level": noise
                                    })
                                except Exception as e:
                                    logger.warning(f"Skipping history entry {entry.get('log_id')} due to error processing performance data: {e}")

                        if history_data:
                            df_history = pd.DataFrame(history_data)
                            df_history["Date"] = pd.to_datetime(df_history["Date"], errors='coerce')
                            df_history.dropna(subset=["Date"], inplace=True)
                            df_history.sort_values("Date", inplace=True)

                            metrics_to_plot = ["Tracking Error (Roll+Pitch Abs)", "Motor Saturation %", "Motor Imbalance %", "Noise Level"]
                            valid_metrics = [m for m in metrics_to_plot if m in df_history.columns and df_history[m].notna().any()]

                            if valid_metrics:
                                fig_hist = make_subplots(rows=len(valid_metrics), cols=1, shared_xaxes=True,
                                                       subplot_titles=valid_metrics)
                                for i, metric in enumerate(valid_metrics):
                                    fig_hist.add_trace(go.Scatter(x=df_history["Date"], y=df_history[metric],
                                                                   mode="lines+markers", name=metric,
                                                                   text=df_history["Filename"], # Add filename to hover
                                                                   hoverinfo='x+y+text'), row=i+1, col=1)
                                    fig_hist.update_yaxes(title_text=metric, row=i+1, col=1)

                                fig_hist.update_layout(height=250*len(valid_metrics), title_text="Key Performance Metrics Over Time", showlegend=False)
                                fig_hist.update_xaxes(title_text="Date", row=len(valid_metrics), col=1)
                                st.plotly_chart(fig_hist, use_container_width=True)

                                with st.expander("Raw History Data Table"):
                                    df_history_display = df_history.copy()
                                    df_history_display["Date"] = df_history_display["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
                                    # Display relevant columns
                                    display_cols = ["Date", "Filename", "Tracking Error (Roll+Pitch Abs)", "Motor Saturation %", "Motor Imbalance %", "Noise Level", "Log ID"]
                                    st.dataframe(df_history_display[[col for col in display_cols if col in df_history_display.columns]])
                            else:
                                st.info("Not enough comparable numeric data in history to plot trends.")
                        else:
                            st.info("Could not extract valid performance data from history.")
                    elif isinstance(history, list) and len(history) <= 1:
                        st.info("Need at least 2 historical log analyses to plot trends.")
                    elif isinstance(history, dict) and 'error' in history:
                         st.error(f"Error loading history data: {history['error']}")
                    else: # Handles empty list or other invalid history states
                        if analyzer.tuning_history_path.exists() and not isinstance(history, list):
                             st.info("No history data found or history file is invalid.")
                        # else: file not found case handled earlier

                except Exception as hist_e:
                    st.error(f"An error occurred while processing tuning history: {hist_e}")
                    logger.error("Error processing tuning history tab", exc_info=True)


    # Display Comparison Results
    elif st.session_state.comparison_results:
        comp = st.session_state.comparison_results
        st.header("📊 Log Comparison Results")
        if "error" in comp:
            st.error(f"Comparison Failed: {comp['error']}")
        else:
            log_id_1 = comp.get("log1_id", "Log 1")
            log_id_2 = comp.get("log2_id", "Log 2")
            st.subheader(f"Comparing:")
            st.markdown(f"- **Log 1 (Older):** `{log_id_1}` (Analyzed: {comp.get('log1_timestamp', 'N/A')})")
            st.markdown(f"- **Log 2 (Newer):** `{log_id_2}` (Analyzed: {comp.get('log2_timestamp', 'N/A')})")

            assessment = comp.get("overall_assessment", {})
            st.metric("Overall Change", assessment.get('verdict', 'N/A'),
                      f"{assessment.get('improvements', 0)} Improvements / {assessment.get('regressions', 0)} Regressions")

            st.markdown("---")
            st.subheader("Detailed Changes:")

            for category, title in [("setting_changes", "Setting Changes"),
                                    ("performance_changes", "Performance Changes")]:
                changes = comp.get(category)
                if changes and isinstance(changes, dict):
                    with st.expander(f"{title} ({len(changes)} changed)", expanded=(category=="performance_changes")):
                        change_data = []
                        for key, values in changes.items():
                             if isinstance(values, dict):
                                pct_change_val = values.get('percent_change')
                                pct_change_str = f"{pct_change_val:.1f}%" if isinstance(pct_change_val, (int, float)) else "N/A"
                                improvement_indicator = "-"
                                if category == "performance_changes":
                                    improvement_flag = values.get('improvement')
                                    if improvement_flag is True: improvement_indicator = "✅ Improvement"
                                    elif improvement_flag is False: improvement_indicator = "❌ Regression"

                                change_data.append({
                                    "Parameter": key,
                                    "Old Value (Log 1)": values.get('old', 'N/A'),
                                    "New Value (Log 2)": values.get('new', 'N/A'),
                                    "% Change": pct_change_str,
                                    "Note": improvement_indicator
                                })
                             else:
                                 logger.warning(f"Invalid format for change item '{key}' in category '{category}'")
                        if change_data:
                            st.dataframe(change_data, use_container_width=True)
                        else:
                            st.info(f"No valid change data processed for {title.lower()}.")
                else:
                    st.info(f"No changes detected in {title.lower()}.")

    # Display Batch Results
    elif st.session_state.batch_results:
        st.header(f"📋 Batch Analysis Results ({len(st.session_state.batch_results)} Logs)")
        batch_summary = []
        valid_batch_results = []

        for i, result in enumerate(st.session_state.batch_results):
            metadata_info = result.get("metadata", {})
            analysis_info = metadata_info.get("analysis_info", {})
            filename = metadata_info.get("filename", f"Log {i+1}")
            timestamp = analysis_info.get("analysis_timestamp", "")
            log_id = result.get("log_id", None)

            if "error" in result:
                batch_summary.append({
                    "Log": filename,
                    "Status": "Error",
                    "Quality Score": result.get("diagnostics", {}).get("quality_score"),
                    "Assessment": result.get("error", "Unknown Error")[:100] + "...",
                    "Timestamp": timestamp
                })
            else:
                assessment = result.get("recommendations", {}).get("flight_assessment", {})
                batch_summary.append({
                    "Log": filename,
                    "Status": "Success",
                    "Quality Score": result.get("data_quality", {}).get("quality_score"),
                    "Assessment": assessment.get("summary", "N/A"),
                    "Timestamp": timestamp
                })
                valid_batch_results.append({"index": i, "display": f"{i+1}: {filename}", "log_id": log_id})

        if batch_summary:
            df_batch = pd.DataFrame(batch_summary)
            if "Quality Score" in df_batch.columns:
                 df_batch["Quality Score"] = df_batch["Quality Score"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A")
            if "Timestamp" in df_batch.columns:
                 df_batch["Timestamp"] = pd.to_datetime(df_batch["Timestamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(df_batch, use_container_width=True)
        else:
             st.info("No batch results to display.")

        if valid_batch_results:
            log_options = {res["display"]: res["index"] for res in valid_batch_results}
            selected_log_display = st.selectbox(
                 "View details for successful batch log:",
                 log_options.keys(),
                 index=None,
                 placeholder="Select a log from the batch..."
                 )

            if selected_log_display is not None:
                selected_log_index = log_options[selected_log_display]
                st.session_state.analysis_output = st.session_state.batch_results[selected_log_index]
                st.session_state.current_file_name = st.session_state.analysis_output.get("metadata", {}).get("filename", f"Log {selected_log_index+1}")
                st.session_state.current_file_path = st.session_state.analysis_output.get("log_id", st.session_state.current_file_name)
                st.session_state.last_analyzed_file = None
                st.session_state.comparison_results = None
                st.session_state.batch_results = []
                logger.info(f"Switching to detailed view for batch log index {selected_log_index}")
                st.rerun()
        else:
             st.info("No logs in the batch were analyzed successfully.")

    # Display Error from Single Analysis Attempt
    elif st.session_state.analysis_output and isinstance(st.session_state.analysis_output, dict) and "error" in st.session_state.analysis_output:
         error_msg = st.session_state.analysis_output.get('error', 'Unknown error')
         st.error(f"Analysis Failed:")
         st.error(error_msg)
         diagnostics = st.session_state.analysis_output.get('diagnostics')
         if diagnostics and isinstance(diagnostics, dict):
             st.subheader("Data Quality Diagnostics (on Error)")
             st.json(diagnostics)
         df_on_error = st.session_state.analysis_output.get('df')
         if debug_mode and isinstance(df_on_error, pd.DataFrame):
             st.subheader("DataFrame Head (on Error)")
             st.dataframe(df_on_error.head())

    # Initial state or after clearing results
    else:
        st.markdown("## Welcome to the Advanced Betaflight Log Analyzer!")
        st.info("👈 **Getting Started:**\n\n1.  Use the sidebar to **upload a single log file** or select one from a **local directory**.\n2.  Click **'Analyze Selected Log'**.\n3.  Explore the results in the tabs that appear.\n4.  Analyze multiple logs to view **'History'** trends or **'Compare'** two specific logs.")

# --- Run the App ---
if __name__ == "__main__":
    main()