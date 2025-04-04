import streamlit as st
st.set_page_config(page_title="Advanced Betaflight Log Analyzer", layout="wide")
import base64
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
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO) # Set to DEBUG for more verbose output
logger = logging.getLogger(__name__)

# --- Default SVG (Quadcopter Silhouette) as Data URI ---
default_svg = '''
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <g fill="none" stroke="black" stroke-width="5">
    <circle cx="100" cy="100" r="20" fill="black" />
    <line x1="100" y1="50" x2="100" y2="150" />
    <line x1="50"  y1="100" x2="150" y2="100" />
    <circle cx="100" cy="50"  r="5" fill="black" />
    <circle cx="100" cy="150" r="5" fill="black" />
    <circle cx="50"  cy="100" r="5" fill="black" />
    <circle cx="150" cy="100" r="5" fill="black" />
  </g>
</svg>
'''
try:
    encoded_svg = base64.b64encode(default_svg.encode('utf-8')).decode('utf-8')
    default_svg_data_uri = "data:image/svg+xml;base64," + encoded_svg
except Exception as e:
    logger.error(f"Error encoding default SVG: {e}")
    default_svg_data_uri = None # Fallback if encoding fails

# --- Simulation Functions ---
def simulate_pid_response(P: float, D: float, wn: float = 5, zeta: float = 0.7):
    """
    Simulate a simplified closed-loop step response for a system controlled by a PD controller.
    This uses a second order model for the plant.
    - wn: natural frequency of the plant (rad/s).
    - zeta: damping ratio of the plant.

    The plant is assumed to be: G(s) = 1 / (s^2 + 2*zeta*wn*s + wn^2)
    The controller is: C(s) = P + D*s. (Ignores I and F terms)

    Returns:
      t: time array.
      y: step response of the closed-loop system.
      overshoot: peak overshoot relative to the setpoint (assumed 1.0).
      rise_time: time taken for the response to rise from 10% to 90% of the setpoint.
    """
    try:
        # Define plant transfer function
        num_plant = [1.0]
        den_plant = [1.0, 2 * zeta * wn, wn**2]

        # Define PD controller
        num_controller = [D, P]
        den_controller = [1.0]

        # Open-loop: L(s) = C(s)*G(s)
        num_open = np.polymul(num_controller, num_plant)
        den_open = np.polymul(den_controller, den_plant)

        # Closed-loop: T(s) = L(s) / (1 + L(s))
        num_closed = num_open
        den_closed = np.polyadd(den_open, num_open)

        # Ensure stability and proper order before creating TransferFunction
        if len(num_closed) >= len(den_closed):
             logger.warning(f"Simulation unstable: Numerator order >= Denominator order for P={P}, D={D}")
             return None, None, None, None # Indicate instability or improper system

        system = signal.TransferFunction(num_closed, den_closed)

        # Simulate step response
        t_max = 10.0 # Define a max simulation time
        t_eval = np.linspace(0, t_max, 500) # Evaluate at specific points for consistency
        t, y = signal.step(system, T=t_eval)

        # Check if simulation produced valid output
        if t is None or y is None or len(t) == 0 or len(y) == 0:
             logger.warning(f"Simulation failed for P={P}, D={D}")
             return None, None, None, None

        # Calculate overshoot (if any)
        max_y = np.max(y) if len(y) > 0 else 0
        overshoot = max(max_y - 1.0, 0.0) if max_y > 1.0 else 0.0

        # Calculate rise time between 10% and 90% of setpoint
        try:
            indices_10 = np.where(y >= 0.1)[0]
            indices_90 = np.where(y >= 0.9)[0]
            if len(indices_10) > 0 and len(indices_90) > 0:
                t10 = t[indices_10[0]]
                t90 = t[indices_90[0]]
                rise_time = t90 - t10
                if rise_time < 0: # Should not happen in a normal step response
                    rise_time = None
            else:
                 rise_time = None # Response didn't reach 10% or 90%
        except IndexError:
            rise_time = None
        except Exception as e_rt:
            logger.warning(f"Error calculating rise time for P={P}, D={D}: {e_rt}")
            rise_time = None

        return t, y, overshoot, rise_time

    except ValueError as ve:
        # Catch potential issues like non-matching array sizes in polyadd/polymul
        logger.error(f"Simulation ValueError for P={P}, D={D}: {ve}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error in simulate_pid_response for P={P}, D={D}: {e}", exc_info=True)
        return None, None, None, None

def optimize_pid_for_axis(current_P: float, current_D: float,
                          desired_overshoot: float = 0.05, # Target 5%
                          desired_rise_time: float = 0.1, # Target 100ms (Adjust as needed)
                          p_range: float = 0.2, # Search +/- 20%
                          d_range: float = 0.2, # Search +/- 20%
                          steps: int = 10) -> Tuple[float, float, float]:
    """
    Optimizes the P and D gains by sweeping a range of values around the current settings
    using the simplified simulation model.
    Returns the recommended P, recommended D, and the best optimization score (lower is better).
    """
    best_score = float('inf')
    best_P = current_P
    best_D = current_D

    p_min = max(0.1, current_P * (1 - p_range)) # Ensure P > 0
    p_max = current_P * (1 + p_range)
    d_min = max(0.0, current_D * (1 - d_range)) # Allow D >= 0
    d_max = current_D * (1 + d_range)

    logger.debug(f"Optimizing PID: Current P={current_P:.2f}, D={current_D:.2f}. Searching P=[{p_min:.2f}-{p_max:.2f}], D=[{d_min:.2f}-{d_max:.2f}]")

    # Generate linearly spaced values, ensuring current values are included if steps > 1
    p_values = np.linspace(p_min, p_max, steps)
    d_values = np.linspace(d_min, d_max, steps)

    for P in p_values:
        for D in d_values:
            # Use the simulation function
            t, y, overshoot, rise_time = simulate_pid_response(P, D)

            # Skip if simulation fails or doesn't provide valid metrics
            if rise_time is None or overshoot is None or t is None:
                continue

            # Define a simple score combining overshoot and rise time error:
            # Weighting factors can be added if one metric is more important
            overshoot_error = abs(overshoot - desired_overshoot)
            rise_time_error = abs(rise_time - desired_rise_time)
            score = overshoot_error + rise_time_error # Simple sum, lower is better

            # Add a penalty for excessive overshoot to prefer safer tunes
            if overshoot > (desired_overshoot + 0.1): # e.g., penalize > 15% if target is 5%
                score *= (1 + (overshoot - (desired_overshoot + 0.1)) * 5)

            if score < best_score:
                best_score = score
                best_P = P
                best_D = D
                logger.debug(f"  New best: P={P:.2f}, D={D:.2f}, Score={score:.4f} (OS={overshoot:.3f}, RT={rise_time:.3f})")

    if best_score == float('inf'):
         logger.warning("PID optimization failed to find any valid simulation results.")
         return current_P, current_D, best_score # Return original values if no solution found

    logger.debug(f"Optimization result: Best P={best_P:.2f}, Best D={best_D:.2f}, Score={best_score:.4f}")
    return best_P, best_D, best_score

# --- Helper Function for JSON Serialization ---
def make_serializable(obj: Any) -> Any:
    """Converts numpy types, datetimes, etc., to JSON-serializable formats."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                          np.float64)):
        if np.isnan(obj):
            return None # Represent NaN as null in JSON
        if np.isinf(obj):
             # Represent infinity as a string or a large number marker
             # Using None might be safer for some downstream processing
            return None # Or str(obj) if you prefer "+inf" / "-inf" strings
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        # Recursively make elements serializable
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()} # Ensure keys are strings
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    elif pd.isna(obj): # Handle pandas NA types
        return None
    elif isinstance(obj, (go.Figure)): # Avoid trying to serialize complex objects like plots
        return "Plotly Figure Object"
    # Add checks for other non-serializable types if needed
    try:
        # Attempt to serialize if it's a basic type
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # Fallback to string representation for unknown types
        logger.debug(f"Could not serialize object of type {type(obj)}, converting to string.")
        return str(obj)

    # --- Helper Function to Find a Column by Acceptable Names ---
    # This helper is used by prepare_data, ensure it's defined correctly in your class or globally
def find_column(self, df: pd.DataFrame, alternatives: List[str]) -> str | None:
    """Finds the first matching column name (case-insensitive, ignoring spaces)."""
    if df is None or df.empty: # Add check for empty DataFrame
        return None
    df_cols_cleaned = {col.strip().lower(): col for col in df.columns}
    for alt in alternatives:
        cleaned_alt = alt.strip().lower()
        if cleaned_alt in df_cols_cleaned:
            return df_cols_cleaned[cleaned_alt] # Return original column name
    return None

# --- BetaflightLogAnalyzer Class ---
class BetaflightLogAnalyzer:
    """
    Analyzes Betaflight blackbox log files.
    Provides methods for parsing logs, performing detailed analysis,
    generating tuning recommendations, comparing logs, plotting results, and 3D flight tracking.
    """
    # (HEADER_DEFINITIONS remains the same as provided)
    HEADER_DEFINITIONS = [
        ("loopIteration", "Counter for each flight controller loop iteration."),
        ("time", "Timestamp in microseconds since flight start."),
        ("axisP[0]", "Proportional PID term for Roll axis."),
        ("axisP[1]", "Proportional PID term for Pitch axis."),
        ("axisP[2]", "Proportional PID term for Yaw axis."),
        ("axisI[0]", "Integral PID term for Roll axis."),
        ("axisI[1]", "Integral PID term for Pitch axis."),
        ("axisI[2]", "Integral PID term for Yaw axis."),
        ("axisD[0]", "Derivative PID term for Roll axis."),
        ("axisD[1]", "Derivative PID term for Pitch axis."),
        ("axisF[0]", "Feedforward PID term for Roll axis."),
        ("axisF[1]", "Feedforward PID term for Pitch axis."),
        ("axisF[2]", "Feedforward PID term for Yaw axis."),
        ("rcCommand[0]", "RC input for Roll from the transmitter."),
        ("rcCommand[1]", "RC input for Pitch from the transmitter."),
        ("rcCommand[2]", "RC input for Yaw from the transmitter."),
        ("rcCommand[3]", "RC input for Throttle from the transmitter."),
        ("setpoint[0]", "Target rotation rate for Roll."),
        ("setpoint[1]", "Target rotation rate for Pitch."),
        ("setpoint[2]", "Target rotation rate for Yaw."),
        ("setpoint[3]", "Throttle setpoint (target throttle)."),
        ("vbatLatest", "Battery voltage reading."),
        ("amperageLatest", "Current draw (amperes * 100)."),
        ("baroAlt", "Altitude from barometer (if available)."),
        ("gyroADC[0]", "Raw gyro sensor data - Roll axis."),
        ("gyroADC[1]", "Raw gyro sensor data - Pitch axis."),
        ("gyroADC[2]", "Raw gyro sensor data - Yaw axis."),
        ("accSmooth[0]", "Filtered accelerometer data - Roll axis."),
        ("accSmooth[1]", "Filtered accelerometer data - Pitch axis."),
        ("accSmooth[2]", "Filtered accelerometer data - Yaw axis."),
        ("debug[0]", "Debug value 0 (depends on debug mode)."),
        ("debug[1]", "Debug value 1 (depends on debug mode)."),
        ("debug[2]", "Debug value 2 (depends on debug mode)."),
        ("debug[3]", "Debug value 3 (depends on debug mode)."),
        ("motor[0]", "Motor 0 throttle command."),
        ("motor[1]", "Motor 1 throttle command."),
        ("motor[2]", "Motor 2 throttle command."),
        ("motor[3]", "Motor 3 throttle command."),
        ("flightModeFlags", "Flags indicating active flight modes."),
        ("stateFlags", "Flags for system state (e.g., arming, failsafe)."),
        ("failsafePhase", "Current failsafe phase (if triggered)."),
        ("rxSignalReceived", "1 if RX signal is currently received."),
        ("rxFlightChannelsValid", "1 if RX channels are valid for flight."),
        ("heading[0]", "Roll angle (heading estimation)."),
        ("heading[1]", "Pitch angle (heading estimation)."),
        ("heading[2]", "Yaw angle (heading estimation)."),
        ("axisSum[0]", "Combined control signal for Roll."),
        ("axisSum[1]", "Combined control signal for Pitch."),
        ("axisSum[2]", "Combined control signal for Yaw."),
        # ("rcCommands[0]", "Normalized RC Roll command."), - Often same as rcCommand[0]
        # ("rcCommands[1]", "Normalized RC Pitch command."),
        # ("rcCommands[2]", "Normalized RC Yaw command."),
        # ("rcCommands[3]", "Normalized RC Throttle command."),
        ("axisError[0]", "Roll axis error (Setpoint - Gyro)."),
        ("axisError[1]", "Pitch axis error."),
        ("axisError[2]", "Yaw axis error."),
        ("gpsCartesianCoords[0]", "X coordinate from GPS in local Cartesian frame."),
        ("gpsCartesianCoords[1]", "Y coordinate from GPS in local Cartesian frame."),
        ("gpsCartesianCoords[2]", "Z coordinate (altitude)."),
        ("gpsDistance", "Distance from home position (meters)."),
        ("gpsHomeAzimuth", "Direction to home in degrees."),
    ]

    def __init__(self):
        self.logs_db_path = "logs_database.json"
        self.tuning_history_path = "tuning_history.json"
        self._ensure_db_files_exist()

    def find_column(self, df: pd.DataFrame, alternatives: List[str]) -> str | None:
            """Finds the first matching column name (case-insensitive, ignoring spaces)."""
            if df is None or df.empty: # Add check for empty DataFrame
                return None
            df_cols_cleaned = {col.strip().lower(): col for col in df.columns}
            for alt in alternatives:
                cleaned_alt = alt.strip().lower()
                if cleaned_alt in df_cols_cleaned:
                    return df_cols_cleaned[cleaned_alt] # Return original column name
            return None

    def _ensure_db_files_exist(self):
        for db_path in [self.logs_db_path, self.tuning_history_path]:
            if not os.path.exists(db_path):
                try:
                    with open(db_path, 'w') as f:
                        # Initialize logs_db as a dictionary, tuning_history as a list
                        json.dump({} if db_path == self.logs_db_path else [], f)
                    logger.info(f"Initialized empty database file: {db_path}")
                except Exception as e:
                    logger.error(f"Error initializing database file {db_path}: {e}")

    def _read_log_file(self, file_path: str) -> List[str]:
        logger.debug(f"Attempting to read file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            logger.warning(f"File is empty: {file_path}")
            raise ValueError("File is empty.")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            if not lines:
                logger.warning(f"File read successfully but contains no lines: {file_path}")
                raise ValueError("File contains no lines.")
            # Basic sanity check for content structure (e.g., minimum number of lines)
            # Increased minimum lines check
            if len(lines) < 3:
                logger.warning(f"File has very few lines ({len(lines)}): {file_path}")
                # raise ValueError("File has too few lines to be a valid log.")
            logger.debug(f"Successfully read {len(lines)} lines from {file_path}.")
            return lines
        except Exception as e:
             logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
             raise IOError(f"Could not read file: {e}")


    def _find_header_and_data(self, lines: List[str]) -> Tuple[List[str], str, int]:
        data_line_idx = -1
        metadata_lines = []
        header_line = None
        header_found = False

        # Common header starts
        header_indicators = ["loopiteration", "time,", "roll,", "pitch,"]

        for i, line in enumerate(lines):
            cleaned_line = line.strip().lower().replace('"', '')
            if not cleaned_line:
                metadata_lines.append(lines[i]) # Keep blank lines in metadata if before header
                continue

            potential_headers = [h.strip() for h in cleaned_line.split(',')]

            # Check if the first few columns look like standard headers
            if len(potential_headers) > 3 and any(indicator in potential_headers[0] for indicator in header_indicators):
                 # More robust check: look for multiple common fields
                 common_fields_present = sum(1 for field in ["time", "gyroadc", "setpoint", "motor"] if any(field in hdr for hdr in potential_headers))
                 if common_fields_present >= 2: # Require at least two common field types
                    # Check if the *next* line looks like data (starts with a digit, common case)
                    if i + 1 < len(lines) and lines[i+1].strip() and lines[i+1].strip()[0].isdigit():
                        data_line_idx = i
                        header_line = lines[i]
                        metadata_lines = lines[:i] # Metadata is everything before the header
                        header_found = True
                        logger.debug(f"Found header matching standard indicators at line {i}.")
                        break
                    # Fallback if next line doesn't start with digit (maybe corrupted first data line)
                    elif i > 0: # Check previous line as potential end of metadata
                         logger.debug(f"Found potential header at line {i}, next line non-digit. Using fallback.")
                         data_line_idx = i
                         header_line = lines[i]
                         metadata_lines = lines[:i]
                         header_found = True
                         break

            if not header_found:
                metadata_lines.append(lines[i]) # Add to metadata if not identified as header yet

        # Fallback if specific keywords failed: look for a line with many commas followed by numeric data
        if not header_found:
            logger.debug("Header indicators not found, trying generic fallback detection...")
            for i in range(len(lines) - 1, 0, -1): # Search backwards
                line = lines[i].strip()
                prev_line = lines[i-1].strip()
                if ',' in prev_line and len(prev_line.split(',')) > 5: # Likely header?
                     if line and (line[0].isdigit() or line[0] == '-'): # Current line looks like data
                          # Check if header candidate actually contains alpha characters
                          if any(c.isalpha() for c in prev_line):
                            data_line_idx = i -1 # The line *before* the first data line is the header
                            header_line = lines[i-1]
                            metadata_lines = lines[:i-1]
                            header_found = True
                            logger.debug(f"Found potential header at line {i-1} using generic fallback.")
                            break

        if not header_found or header_line is None:
            logger.error("Could not reliably identify the log data header.")
            raise ValueError("Could not reliably identify the log data header. Check log file format.")

        data_start_index = data_line_idx + 1
        if data_start_index >= len(lines):
            logger.error("No data found after the identified header row.")
            raise ValueError("No data found after the header row.")

        logger.debug(f"Header found at index {data_line_idx}, data starts at index {data_start_index}.")
        return metadata_lines, header_line, data_start_index

    def parse_metadata(self, metadata_lines: List[str]) -> dict:
        logger.debug(f"Parsing metadata from {len(metadata_lines)} lines.")
        metadata = {'pid_values': {}, 'rates': {}, 'filters': {}, 'rc_rates': {}, 'other_settings': {}}

        # Precompile regex patterns for efficiency
        patterns = {
            'betaflight_version': re.compile(r'Betaflight\s+/\s+(\w+)\s+(\d+\.\d+\.\d+)', re.IGNORECASE),
            'firmware_target': re.compile(r'Firmware target:\s*(\S+)', re.IGNORECASE), # Use \S+ for target name
            'board_name': re.compile(r'Board information:\s*\w+\s*([\w-]+)', re.IGNORECASE),
            'log_start_datetime': re.compile(r'Log\s+Start\s+DateTime:\s*"?([^",]+)', re.IGNORECASE),
            'craft_name': re.compile(r'Craft\s+name:\s*"?([^",]+)', re.IGNORECASE),
            # Generic Key-Value using common separators (:, =, comma after quoted key)
             'generic_kv': re.compile(r'H\s+([^:]+):\s*(.+)', re.IGNORECASE), # Simple H Key: Value
             'set_command': re.compile(r'(?:set\s+)?([\w_]+)\s*=\s*([\w.-]+)', re.IGNORECASE) # Matches "set <key> = <value>" or "<key> = <value>"
        }

        pid_axis_map = {'roll': 0, 'pitch': 1, 'yaw': 2}

        for line in metadata_lines:
            line = line.strip()
            if not line:
                continue

            # Check specific patterns first
            matched = False
            if line.startswith('H'): # Process header lines first
                # Try generic Key: Value pattern
                generic_match = patterns['generic_kv'].match(line)
                if generic_match:
                    key = generic_match.group(1).strip().lower().replace(' ', '_')
                    value = generic_match.group(2).strip().strip('"')
                    # Special handling for known keys
                    if key == 'log_start_datetime':
                         metadata["log_start_datetime"] = value
                    elif key == 'firmware:': # Example specific key handling
                         parts = value.split('/')
                         if len(parts) > 1:
                             fw_parts = parts[1].split()
                             if len(fw_parts) > 1:
                                 metadata['firmware_target'] = fw_parts[0].strip()
                                 metadata['betaflight_version'] = fw_parts[1].strip()
                    elif key == 'craft_name:':
                         metadata['craft_name'] = value
                    else:
                         # Store other H lines in other_settings
                         metadata['other_settings'][key.rstrip(':')] = value # Remove trailing colon if present
                    matched = True
                    continue # Move to next line once matched

            # If not a standard H Key: Value, check other regexes
            for key_name, pattern in patterns.items():
                 if key_name == 'generic_kv': continue # Already tried

                 match = pattern.search(line)
                 if match:
                     if key_name == 'betaflight_version':
                         # Sometimes target/version are on the same line
                         metadata['firmware_target'] = match.group(1)
                         metadata['betaflight_version'] = match.group(2)
                     elif key_name == 'firmware_target':
                          # Only set if not already found by betaflight_version pattern
                         if 'firmware_target' not in metadata:
                             metadata['firmware_target'] = match.group(1)
                     elif key_name == 'board_name':
                         metadata['board'] = match.group(1)
                     elif key_name == 'log_start_datetime':
                          if 'log_start_datetime' not in metadata:
                              metadata['log_start_datetime'] = match.group(1).strip('"')
                     elif key_name == 'craft_name':
                         if 'craft_name' not in metadata:
                             metadata['craft_name'] = match.group(1).strip('"')
                     elif key_name == 'set_command':
                        set_key = match.group(1).lower()
                        set_value = match.group(2)
                        # Attempt to convert value to float/int if possible
                        try:
                            if '.' in set_value:
                                num_value = float(set_value)
                            else:
                                num_value = int(set_value)
                            set_value = num_value
                        except ValueError:
                            pass # Keep as string if conversion fails

                        # Categorize common settings
                        pid_match = re.match(r'([pidf])_?(roll|pitch|yaw)', set_key)
                        rate_match = re.match(r'(roll|pitch|yaw)_rate', set_key)
                        filter_match = re.match(r'(\w+)_lpf_hz', set_key) # e.g., gyro_lpf_hz, dterm_lpf_hz

                        if pid_match:
                             term, axis = pid_match.groups()
                             if isinstance(set_value, (int, float)):
                                 metadata['pid_values'][f"{term}_{axis}"] = set_value
                             else:
                                 logger.warning(f"Non-numeric PID value found for {set_key}: {set_value}")
                        elif rate_match:
                             axis = rate_match.group(1)
                             metadata['rates'][f"{axis}_rate"] = set_value
                        elif filter_match:
                             filter_name = filter_match.group(1)
                             metadata['filters'][f"{filter_name}_lpf_hz"] = set_value
                        elif 'rc_rate' in set_key or 'expo' in set_key or 'super_rate' in set_key:
                             metadata['rc_rates'][set_key] = set_value
                        else:
                             metadata['other_settings'][set_key] = set_value

                     matched = True
                     break # Stop checking patterns for this line

            if not matched and not line.startswith('H') and '=' in line:
                # Fallback for lines like "key = value" not caught by 'set_command' regex
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    try:
                         if '.' in value: num_value = float(value)
                         else: num_value = int(value)
                         value = num_value
                    except ValueError: pass # Keep as string
                    if key: # Ensure key is not empty
                        metadata['other_settings'][key] = value
                        matched = True


        metadata['analysis_timestamp'] = datetime.now().isoformat() # Use ISO format
        logger.debug("Metadata parsing finished.")
        # Log missing crucial metadata
        for essential in ['betaflight_version', 'firmware_target']:
             if essential not in metadata:
                 logger.warning(f"Essential metadata '{essential}' not found.")
        return metadata

    def parse_data(self, header_line: str, data_lines: List[str]) -> pd.DataFrame:
        """Parses the data section into a pandas DataFrame."""
        # Clean header: remove quotes, strip whitespace
        header = [h.strip().replace('"', '') for h in header_line.strip().split(',')]
        num_columns = len(header)
        logger.debug(f"Parsing data with header: {header}")

        # Pre-process data lines: remove leading/trailing whitespace
        # Using io.StringIO is generally efficient for pandas
        csv_content = header_line + "".join(data_lines) # Assume lines already have newlines
        csv_file = io.StringIO(csv_content)

        try:
            logger.debug(f"Attempting to read CSV data with {num_columns} columns.")
            df = pd.read_csv(
                csv_file,
                header=0, # Header is the first line we provided
                # names=header, # Explicitly setting names can sometimes help with malformed lines
                skipinitialspace=True,
                on_bad_lines='warn', # Log bad lines
                # Use a specific na_values list if needed, e.g., na_values=['NA', 'N/A', '']
                low_memory=False # Often necessary for mixed types or large files
            )
            logger.debug(f"Initial DataFrame shape: {df.shape}")

            # Post-read cleaning and validation
            df.columns = df.columns.str.strip().str.replace('"', '') # Ensure columns are clean
            logger.debug(f"Cleaned columns: {df.columns.tolist()}")

             # Check if column count matches header count
            if df.shape[1] != num_columns:
                 logger.warning(f"Column count mismatch: Header has {num_columns}, DataFrame has {df.shape[1]}. Data might be misaligned.")
                 # Attempt to fix by re-reading with expected columns if mismatch is severe
                 if abs(df.shape[1] - num_columns) > 1:
                      logger.info("Attempting re-read with usecols...")
                      csv_file.seek(0) # Reset StringIO
                      df = pd.read_csv(
                         csv_file, header=0, usecols=range(num_columns), names=header, # Force columns
                         skipinitialspace=True, on_bad_lines='warn', low_memory=False
                      )
                      logger.info(f"Re-read DataFrame shape: {df.shape}")
                      if df.shape[1] != num_columns:
                          logger.error("Column count mismatch persists after re-read.")
                          # Potentially drop extra unnamed columns if they appear
                          df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


            # Remove duplicated columns (keeping the first occurrence)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            logger.debug(f"Shape after removing duplicate columns: {df.shape}")

            return df

        except pd.errors.ParserError as pe:
             logger.error(f"Pandas ParserError: {pe}. Check for malformed lines near the error indication.")
             raise ValueError(f"Failed to parse log data due to CSV format issues: {pe}")
        except Exception as e:
            logger.error(f"Unexpected error parsing data with pandas: {e}", exc_info=True)
            raise ValueError(f"Failed to parse log data into DataFrame: {e}")


# --- Corrected prepare_data function ---
    def prepare_data(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Cleans, prepares, and validates the raw DataFrame data.
        Handles missing values, numeric conversion, and sets the time index.
        """
        logger.debug(f"Starting data preparation. Initial shape: {df.shape}, Columns: {df.columns.tolist()}")

        if df.empty:
            logger.warning("Input DataFrame is empty before preparation.")
            return df

        # 1. Clean Column Names (redundant check, but safe)
        df.columns = df.columns.str.strip().str.replace('"', '')
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # 2. Drop Columns that are ALL NaN
        initial_cols = set(df.columns)
        df = df.dropna(axis=1, how='all')
        dropped_cols = initial_cols - set(df.columns)
        if dropped_cols:
            logger.debug(f"Dropped fully NaN columns: {dropped_cols}")
        if df.empty:
            logger.warning("DataFrame became empty after dropping all-NaN columns.")
            return df

        # 3. Convert to Numeric, Coercing Errors
        numeric_cols = []
        non_numeric_issues = {}
        for col in df.columns:
            is_numeric_before = pd.api.types.is_numeric_dtype(df[col])
            if not is_numeric_before:
                original_nan_count = df[col].isnull().sum()
                converted_col = pd.to_numeric(df[col], errors='coerce')
                new_nan_count = converted_col.isnull().sum()
                increase_in_nans = new_nan_count - original_nan_count
                # Log significant NaN increases but don't revert automatically
                if increase_in_nans > 0.1 * len(df) and increase_in_nans > 100:
                    logger.warning(f"Column '{col}' had significant NaN increase ({increase_in_nans}) after numeric conversion.")
                    non_numeric_issues[col] = increase_in_nans
                df[col] = converted_col
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        if non_numeric_issues:
            logger.warning(f"Columns with significant NaN increase after coercion: {non_numeric_issues}")

        # 4. Handle Missing Values (Forward Fill then Backward Fill)
        if numeric_cols:
            original_nan_total = df[numeric_cols].isnull().sum().sum()
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            filled_nan_count = original_nan_total - df[numeric_cols].isnull().sum().sum()
            if filled_nan_count > 0:
                logger.debug(f"Filled {filled_nan_count} missing values in numeric columns using ffill/bfill.")
            remaining_nan_cols = df[numeric_cols].columns[df[numeric_cols].isnull().any()].tolist()
            if remaining_nan_cols:
                logger.warning(f"NaN values remain in numeric columns after fill: {remaining_nan_cols}.")

        # 5. Identify and Set Time Index
        # Use self.find_column since this is a class method
        time_col = self.find_column(df, ["time", "Time"])
        logger.debug(f"Found time column candidate: {time_col}")
        time_index_set = False # Flag to track if index was successfully set
        if time_col and time_col in df.columns:
            logger.debug(f"Time column '{time_col}' found in DataFrame columns.")
            if pd.api.types.is_numeric_dtype(df[time_col]):
                logger.debug(f"Time column '{time_col}' is numeric.")
                try:
                    logger.debug(f"Converting time column '{time_col}' (microseconds) to seconds.")
                    df[time_col] = df[time_col] / 1_000_000.0
                    if df[time_col].duplicated().any():
                        duplicates = df[time_col].duplicated().sum()
                        logger.warning(f"Found {duplicates} duplicate timestamps in '{time_col}'. Averaging values.")
                        numeric_cols_in_df = df.select_dtypes(include=np.number).columns.tolist()
                        if time_col not in numeric_cols_in_df: numeric_cols_in_df.append(time_col)
                        agg_funcs = {col: 'mean' for col in numeric_cols_in_df if col != time_col}
                        non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                        agg_funcs.update({col: 'first' for col in non_numeric_cols})
                        logger.debug(f"Aggregation functions for groupby: {agg_funcs}")
                        df = df.groupby(time_col).agg(agg_funcs).reset_index()
                        logger.debug(f"Shape after groupby/agg/reset_index: {df.shape}")

                    logger.debug(f"Columns before set_index: {df.columns.tolist()}")
                    logger.debug(f"Index before set_index: {df.index}")
                    df.set_index(time_col, inplace=True)
                    logger.debug(f"Index name after set_index: {df.index.name}")
                    if not df.index.is_monotonic_increasing:
                        logger.warning(f"Time index '{time_col}' is not monotonic increasing. Sorting index.")
                        df.sort_index(inplace=True)
                    logger.debug(f"Set '{time_col}' as DataFrame index (seconds).")
                    metadata['time_unit'] = 'seconds'
                    metadata['time_column'] = time_col
                    time_index_set = True # Mark as successful
                except Exception as e:
                    logger.error(f"Could not set '{time_col}' as index: {e}. Proceeding without time index.", exc_info=True)
            else:
                logger.warning(f"Time column '{time_col}' is not numeric. Cannot set as index.")
        else:
            logger.warning("No standard 'time' column found or not in DataFrame columns.")

        # 6. Drop Rows with NaNs in Essential Columns (after fill attempts)
        # Define essential columns for basic flight analysis
        essential_cols = [col for col in df.columns if 'gyro' in col.lower() or 'motor' in col.lower()]
        logger.debug(f"Initial essential_cols: {essential_cols}")
        logger.debug(f"Value of time_col variable: '{time_col}'")
        logger.debug(f"Current df.index.name: '{df.index.name}'")

        # ****** FIX APPLIED HERE ******
        # Add index name to essential columns ONLY if it was successfully set and matches time_col
        if time_col is not None and df.index.name == time_col: # Check time_col is not None AND index name matches
            logger.debug(f"Condition TRUE: Appending index '{df.index.name}' to essential_cols.")
            essential_cols.append(df.index.name)
        else:
            logger.debug(f"Condition FALSE: index name ('{df.index.name}') != time_col ('{time_col}') or time_col is None.")
        # ****** END OF FIX ******

        logger.debug(f"Essential columns before subset construction: {essential_cols}")
        # Construct the subset for dropna
        subset_cols = [col for col in essential_cols if col in df.columns or col == df.index.name]
        logger.debug(f"Constructed subset for dropna: {subset_cols}")

        initial_rows = len(df)
        if subset_cols: # Only drop if there's a valid subset
            try:
                # This is the line that was failing
                df.dropna(subset=subset_cols, how='any', inplace=True)
                rows_dropped = initial_rows - len(df)
                if rows_dropped > 0: logger.warning(f"Dropped {rows_dropped} rows due to NaN in essential columns: {subset_cols}")
            except KeyError as ke:
                logger.error(f"KeyError during dropna with subset {subset_cols}. Columns: {df.columns}, Index: {df.index.name}. Error: {ke}", exc_info=True)
                raise ke # Re-raise after logging details
            except Exception as e_drop:
                logger.error(f"Error during dropna: {e_drop}", exc_info=True)
                raise e_drop # Re-raise other errors
        else:
            logger.warning("No essential columns found to check for NaNs.")

        if df.empty:
            logger.error("DataFrame became empty after final NaN drop. Cannot proceed.")
            raise ValueError("No valid data rows remaining after preparation.")

        logger.debug(f"Data preparation finished. Final shape: {df.shape}")
        return df

    # --- Analysis Methods ---
    def analyze_pid_performance(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing PID performance...")
        results = {'step_response': {}, 'tracking_error': {}}
        pid_found = False

        # --- Tracking Error Calculation ---
        for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
            gyro_col = find_column(df, [f"gyro{axis_name.capitalize()}", f"gyroADC[{axis_idx}]"])
            setpoint_col = find_column(df, [f"setpoint{axis_name.capitalize()}", f"setpoint[{axis_idx}]"])
            error_col = find_column(df, [f"axisError[{axis_idx}]"])

            if gyro_col and setpoint_col and gyro_col in df and setpoint_col in df:
                pid_found = True
                gyro_data = df[gyro_col].dropna()
                setpoint_data = df[setpoint_col].dropna()

                # Align data using the index (time)
                aligned_gyro, aligned_setpoint = gyro_data.align(setpoint_data, join='inner')

                if not aligned_gyro.empty:
                    # Use pre-calculated error if available, otherwise calculate
                    if error_col and error_col in df:
                        error_data = df[error_col].dropna()
                        aligned_error, _ = error_data.align(aligned_gyro, join='inner') # Align error with gyro/setpoint
                        if aligned_error.empty:
                             logger.warning(f"Pre-calculated error column '{error_col}' exists but has no overlap with gyro/setpoint.")
                             tracking_error = aligned_setpoint - aligned_gyro
                        else:
                             tracking_error = aligned_error
                    else:
                        tracking_error = aligned_setpoint - aligned_gyro

                    results['tracking_error'][f"{axis_name}_mae"] = float(tracking_error.abs().mean())
                    results['tracking_error'][f"{axis_name}_rmse"] = float(np.sqrt((tracking_error**2).mean()))
                    results['tracking_error'][f"{axis_name}_mean"] = float(tracking_error.mean())
                    results['tracking_error'][f"{axis_name}_std"] = float(tracking_error.std())
                else:
                    logger.warning(f"No overlapping gyro/setpoint data for {axis_name} axis.")
                    results['tracking_error'][f"{axis_name}_error"] = "No overlapping data"
            else:
                results['tracking_error'][f"{axis_name}_error"] = "Gyro or Setpoint column missing"

        # --- Step Response Analysis (Simplified) ---
        # This is complex to do robustly from noisy flight logs.
        # We look for sharp changes in setpoint and analyze the gyro response shortly after.
        time_step = None
        if isinstance(df.index, pd.TimedeltaIndex) or pd.api.types.is_numeric_dtype(df.index.dtype):
             time_diffs = np.diff(df.index)
             if len(time_diffs) > 0:
                 time_step = np.median(time_diffs)

        if time_step is not None and time_step > 1e-9: # Ensure valid time step
            for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
                setpoint_col = find_column(df, [f"setpoint{axis_name.capitalize()}", f"setpoint[{axis_idx}]"])
                gyro_col = find_column(df, [f"gyro{axis_name.capitalize()}", f"gyroADC[{axis_idx}]"])

                if setpoint_col and gyro_col and setpoint_col in df and gyro_col in df:
                    setpoint = df[setpoint_col]
                    gyro = df[gyro_col]

                    # Detect step changes in setpoint (derivative threshold)
                    setpoint_diff = setpoint.diff().abs()
                    step_threshold = setpoint_diff.quantile(0.95) # Look for changes in top 5% magnitude
                    step_indices = np.where(setpoint_diff > max(step_threshold, 50))[0] # Min threshold 50deg/s change

                    rise_times = []
                    overshoots = []
                    settling_times = []

                    # Analyze response after each detected step
                    window_sec = 0.2 # Analyze 200ms window after step
                    window_size = int(window_sec / time_step) if time_step else 50 # Default window size

                    for idx in step_indices:
                         if idx + window_size >= len(df) or idx < 1: continue # Ensure window is within bounds

                         t_step = df.index[idx]
                         setpoint_pre_step = setpoint.iloc[idx-1]
                         setpoint_post_step = setpoint.iloc[idx] # Target value is the new setpoint level
                         step_magnitude = setpoint_post_step - setpoint_pre_step

                         if abs(step_magnitude) < 20: continue # Ignore small steps

                         gyro_window = gyro.iloc[idx : idx + window_size]
                         time_window = df.index[idx : idx + window_size]

                         # Normalize gyro response relative to the step start
                         gyro_relative = gyro_window - gyro.iloc[idx-1]
                         target_relative = setpoint_post_step - gyro.iloc[idx-1]

                         if abs(target_relative) < 1e-6: continue # Avoid division by zero

                         # Calculate metrics if target is non-zero
                         try:
                             # Rise Time (10% to 90% of *relative* target)
                             indices_10 = np.where(gyro_relative >= 0.1 * target_relative)[0]
                             indices_90 = np.where(gyro_relative >= 0.9 * target_relative)[0]
                             if len(indices_10) > 0 and len(indices_90) > 0:
                                 t10 = time_window[indices_10[0]]
                                 t90 = time_window[indices_90[0]]
                                 rt = (t90 - t10) * 1000 # Rise time in ms
                                 if rt > 0 and rt < window_sec * 1000: # Basic validity check
                                     rise_times.append(rt)

                             # Overshoot (peak relative to target)
                             peak_relative = gyro_relative.max() if target_relative > 0 else gyro_relative.min()
                             overshoot_val = (peak_relative - target_relative) / abs(target_relative)
                             if overshoot_val > 0 and overshoot_val < 2.0: # Check if overshoot is positive and reasonable (less than 200%)
                                 overshoots.append(overshoot_val * 100) # Overshoot in percent

                             # Settling Time (within 5% of target) - Find last time it exits band
                             settled_band_low = target_relative * 0.95
                             settled_band_high = target_relative * 1.05
                             outside_band_indices = np.where((gyro_relative < settled_band_low) | (gyro_relative > settled_band_high))[0]
                             if len(outside_band_indices) > 0:
                                 last_outside_idx = outside_band_indices[-1]
                                 if last_outside_idx + 1 < len(time_window):
                                      settling_time_val = (time_window[last_outside_idx + 1] - t_step) * 1000 # ms
                                      settling_times.append(settling_time_val)
                                 else: # Settled at the very end or not at all in window
                                      settling_times.append(window_sec * 1000)
                             else: # Settled immediately (or within first sample)
                                 settling_times.append((time_window[0] - t_step) * 1000 if len(time_window) > 0 else 0)

                         except Exception as e_step:
                             logger.warning(f"Error calculating step response metrics for {axis_name} at index {idx}: {e_step}")


                    # Store median results if enough steps were found
                    if len(rise_times) > 3:
                         results['step_response'][f"{axis_name}_rise_time_ms"] = round(np.nanmedian(rise_times), 1)
                    if len(overshoots) > 3:
                         results['step_response'][f"{axis_name}_overshoot_percent"] = round(np.nanmedian(overshoots), 1)
                    if len(settling_times) > 3:
                         results['step_response'][f"{axis_name}_settling_time_ms"] = round(np.nanmedian(settling_times), 1)


        if not pid_found:
             results["error_overall"] = "No suitable PID/Gyro/Setpoint columns found for analysis."
             logger.error(results["error_overall"])

        logger.debug("PID performance analysis finished.")
        return results

    def analyze_setpoint_vs_rc(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing Setpoint vs RC Command relationship...")
        results = {}
        rc_mapping_found = False

        for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
             rc_col = find_column(df, [f"rcCommand{axis_name.capitalize()}", f"rcCommand[{axis_idx}]"])
             setpoint_col = find_column(df, [f"setpoint{axis_name.capitalize()}", f"setpoint[{axis_idx}]"])

             if rc_col and setpoint_col and rc_col in df and setpoint_col in df:
                 rc_data = df[rc_col].dropna()
                 sp_data = df[setpoint_col].dropna()

                 # Filter for significant RC input and non-zero setpoint to find relationship
                 # Assuming RC commands are typically 1000-2000 or similar range, center is ~1500
                 rc_center = rc_data.median() # Estimate center
                 rc_significant = rc_data[abs(rc_data - rc_center) > 50] # Look where stick is moved significantly
                 sp_active = sp_data[abs(sp_data) > 10] # Look where setpoint is non-trivial

                 # Align the filtered data
                 aligned_rc, aligned_sp = rc_significant.align(sp_active, join='inner')

                 if len(aligned_rc) > 50: # Need sufficient points
                     rc_mapping_found = True
                     # Calculate correlation
                     correlation = aligned_rc.corr(aligned_sp)
                     results[f"{axis_name}_rc_sp_correlation"] = float(correlation) if pd.notna(correlation) else None

                     # Estimate rate sensitivity (deg/s per RC unit change from center) - simplified linear fit
                     try:
                         # Use only data where RC is away from center
                         rc_relative = aligned_rc - rc_center
                         # Simple linear regression: sp = slope * rc_relative
                         # slope = cov(rc_relative, aligned_sp) / var(rc_relative)
                         covariance = np.cov(rc_relative, aligned_sp)[0, 1]
                         variance = np.var(rc_relative)
                         if variance > 1e-6:
                             slope = covariance / variance
                             results[f"{axis_name}_rc_sensitivity_deg_s_per_unit"] = float(slope)
                         else:
                             results[f"{axis_name}_rc_sensitivity_error"] = "RC variance too low"

                     except Exception as e_sens:
                         logger.warning(f"Could not calculate RC sensitivity for {axis_name}: {e_sens}")
                         results[f"{axis_name}_rc_sensitivity_error"] = str(e_sens)

                 else:
                     results[f"{axis_name}_rc_sp_error"] = "Insufficient active RC/Setpoint data"
             else:
                 results[f"{axis_name}_rc_sp_error"] = "RC Command or Setpoint column missing"

        if not rc_mapping_found:
             results["error_overall"] = "Could not find sufficient data to analyze RC vs Setpoint relationship."

        logger.debug("Setpoint vs RC analysis finished.")
        return {"setpoint_vs_rc": results} # Nest results for clarity

    def analyze_motors(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing motors...")
        results = {}

        # Find motor columns (e.g., motor[0], motor[1], ..., or Motor1, Motor2, ...)
        motor_cols_numeric = sorted([col for col in df.columns if col.lower().startswith('motor[') and col.endswith(']')])
        motor_cols_named = sorted([col for col in df.columns if col.lower().startswith('motor') and col[5:].isdigit()])
        motor_cols = motor_cols_numeric if motor_cols_numeric else motor_cols_named

        if not motor_cols:
            # Fallback: look for any column containing 'motor' if primary patterns fail
            motor_cols = sorted([col for col in df.columns if 'motor' in col.lower()])
            if not motor_cols:
                 results["error_motors"] = "No motor data columns found."
                 logger.error(results["error_motors"])
                 return {"motors": results} # Return nested structure
            else:
                 logger.warning(f"Using fallback motor column detection: {motor_cols}")

        logger.debug(f"Found motor columns: {motor_cols}")
        motor_data = df[motor_cols].copy() # Work on a copy

        # Ensure motor data is numeric
        all_numeric = True
        for col in motor_cols:
            if not pd.api.types.is_numeric_dtype(motor_data[col]):
                logger.warning(f"Motor column '{col}' is not numeric. Attempting conversion.")
                motor_data[col] = pd.to_numeric(motor_data[col], errors='coerce')
                if motor_data[col].isnull().all():
                    logger.error(f"Motor column '{col}' became all NaN after conversion. Cannot analyze.")
                    results["error_motors"] = f"Motor column '{col}' conversion failed."
                    return {"motors": results}
            if not pd.api.types.is_numeric_dtype(motor_data[col]):
                 all_numeric = False # Mark if any column failed conversion

        if not all_numeric:
             logger.error("Not all identified motor columns could be converted to numeric.")
             # Decide whether to proceed with only numeric columns or fail
             # Let's try proceeding with numeric ones
             numeric_motor_cols = motor_data.select_dtypes(include=np.number).columns.tolist()
             if not numeric_motor_cols:
                   results["error_motors"] = "No numeric motor data available after conversion attempts."
                   return {"motors": results}
             logger.warning(f"Proceeding with only numeric motor columns: {numeric_motor_cols}")
             motor_cols = numeric_motor_cols
             motor_data = motor_data[motor_cols]


        if motor_data.isnull().all().all():
             results["error_motors"] = "Motor data columns found but contain only NaN values."
             logger.error(results["error_motors"])
             return {"motors": results}

        # Determine motor output range (e.g., 0-1, 1000-2000, 0-1000)
        # Use quantiles to be robust against outliers
        try:
            q_low = motor_data.quantile(0.01).min() # 1st percentile min across motors
            q_high = motor_data.quantile(0.99).max() # 99th percentile max across motors
            overall_min = motor_data.min().min() # Absolute min
            overall_max = motor_data.max().max() # Absolute max
        except Exception as e_stat:
            logger.error(f"Error calculating motor stats: {e_stat}")
            results["error_motors"] = f"Could not calculate motor value statistics: {e_stat}"
            return {"motors": results}


        motor_range_min = 0
        motor_range_max = 2000 # Default guess

        # Refined range detection based on typical Betaflight/ESC protocols
        if q_high <= 1.1 and q_low >= -0.1:
            motor_range_min, motor_range_max = 0.0, 1.0
            results["motor_value_type"] = "Percentage (0-1)"
        elif q_high > 1000 and q_low >= 950: # DShot/PWM range often starts near 1000
            motor_range_min, motor_range_max = 1000, 2000
            results["motor_value_type"] = "Scaled (e.g., 1000-2000)"
        elif q_high > 100 and q_low >= -10: # Could be 0-1000 or percentage * 10
             # Disambiguate based on max value
             if q_high > 900:
                  motor_range_min, motor_range_max = 0, 1000
                  results["motor_value_type"] = "Scaled (0-1000)"
             else: # Likely 0-100 or similar - less common now
                  motor_range_min, motor_range_max = 0, max(100, q_high) # Use 100 or detected high
                  results["motor_value_type"] = f"Scaled (0-{motor_range_max})"

        else: # Unknown range
            motor_range_min = max(0, round(q_low))
            motor_range_max = round(q_high)
            results["motor_value_type"] = f"Unknown/Custom ({motor_range_min}-{motor_range_max})"
            logger.warning(f"Could not reliably determine motor range. Assuming {motor_range_min}-{motor_range_max} based on quantiles.")

        results["motor_range_min_detected"] = motor_range_min
        results["motor_range_max_detected"] = motor_range_max
        results["motor_min_output_overall"] = round(float(overall_min), 2)
        results["motor_max_output_overall"] = round(float(overall_max), 2)


        # --- Motor Saturation ---
        saturation_threshold_upper = motor_range_max * 0.98 # 98% of max range
        saturation_threshold_lower = motor_range_min + (motor_range_max - motor_range_min) * 0.02 # 2% above min range (e.g., for bidirectional)

        saturated_high = (motor_data >= saturation_threshold_upper).sum()
        saturated_low = (motor_data <= saturation_threshold_lower).sum() # Check lower saturation too

        total_points_per_motor = motor_data.count() # Count non-NaN points per motor
        total_points_overall = total_points_per_motor.sum()

        if total_points_overall > 0:
            # Saturation percentage per motor
            results["motor_saturation_pct_per_motor_high"] = ((saturated_high / total_points_per_motor) * 100).round(2).to_dict()
            results["motor_saturation_pct_per_motor_low"] = ((saturated_low / total_points_per_motor) * 100).round(2).to_dict()
            # Overall saturation percentage
            results["motor_saturation_pct_overall_high"] = round(float(saturated_high.sum() / total_points_overall) * 100, 2)
            results["motor_saturation_pct_overall_low"] = round(float(saturated_low.sum() / total_points_overall) * 100, 2)
            # Combine high/low saturation for a total metric if desired
            results["motor_saturation_pct_overall_any"] = round(float((saturated_high.sum() + saturated_low.sum()) / total_points_overall) * 100, 2)
        else:
            results["motor_saturation_error"] = "No valid motor data points to calculate saturation."


        # --- Motor Balance/Imbalance ---
        try:
            avg_motor_output = motor_data.mean(axis=0) # Average per motor
            results["motor_averages"] = avg_motor_output.round(2).to_dict()

            if len(motor_cols) > 1:
                overall_avg = avg_motor_output.mean()
                # Calculate imbalance as std dev of average motor outputs, relative to overall average
                imbalance_std = avg_motor_output.std()
                if overall_avg > 1e-6: # Avoid division by zero
                     results["motor_imbalance_std_dev"] = round(float(imbalance_std), 2)
                     results["motor_imbalance_pct_of_avg"] = round(float(imbalance_std / overall_avg) * 100, 2)
                else:
                     results["motor_imbalance_error"] = "Overall average motor output too low for relative imbalance."

                # Also calculate max difference between motor averages
                max_diff = avg_motor_output.max() - avg_motor_output.min()
                results["motor_max_avg_difference"] = round(float(max_diff), 2)
                if overall_avg > 1e-6:
                    results["motor_max_avg_diff_pct"] = round(float(max_diff / overall_avg) * 100, 2)

        except Exception as e_bal:
            logger.error(f"Error calculating motor balance: {e_bal}")
            results["error_motor_balance"] = f"Error calculating motor balance: {e_bal}"


        # --- Average Throttle ---
        # Requires identifying the throttle command column
        throttle_col = find_column(df, ["rcCommand[3]", "rcCommandThrottle", "throttle"])
        if throttle_col and throttle_col in df and pd.api.types.is_numeric_dtype(df[throttle_col]):
             avg_throttle = df[throttle_col].mean()
             results["avg_throttle_rc_input"] = round(float(avg_throttle), 2)
             # Estimate motor output at average throttle? Complex, depends on mix.
             # For simplicity, use the overall average motor output as a proxy.
             if "motor_averages" in results and results["motor_averages"]:
                 overall_motor_avg = np.mean(list(results["motor_averages"].values()))
                 results["avg_motor_output_overall"] = round(float(overall_motor_avg), 2)


        logger.debug("Motor analysis finished.")
        return {"motors": results} # Return nested

    def perform_spectral_analysis(self, df: pd.DataFrame, sampling_rate: float | None = None) -> dict:
        logger.debug("Starting spectral analysis...")
        results = {'spectra': {}, 'settings': {}}

        # --- Determine Sampling Rate ---
        if sampling_rate is None:
            if isinstance(df.index, pd.TimedeltaIndex):
                 time_diffs = np.diff(df.index.total_seconds())
            elif pd.api.types.is_numeric_dtype(df.index.dtype):
                 time_diffs = np.diff(df.index)
            else:
                 time_diffs = []
                 logger.warning("Cannot determine sampling rate from non-numeric/non-timedelta index.")

            if len(time_diffs) > 10: # Need enough points to estimate
                median_diff = np.nanmedian(time_diffs)
                std_diff = np.nanstd(time_diffs)
                if median_diff > 1e-9: # Check for valid time difference
                     # Check for irregular sampling
                     if std_diff > 0.1 * median_diff: # If std dev is > 10% of median diff
                          logger.warning(f"Irregular sampling detected. Median diff: {median_diff*1000:.3f} ms, Std dev: {std_diff*1000:.3f} ms. FFT results may be less reliable.")
                          results['settings']['sampling_warning'] = f"Irregular sampling (median dt={median_diff*1000:.3f}ms, std={std_diff*1000:.3f}ms)"
                     # Calculate sampling rate from median difference
                     sampling_rate = 1.0 / median_diff
                     results['settings']['estimated_sampling_rate_hz'] = round(sampling_rate)
                else:
                     sampling_rate = None # Invalid median difference
            else: # Not enough data points or couldn't determine diffs
                 sampling_rate = None

            # Fallback if calculation failed
            if sampling_rate is None:
                sampling_rate = 1000 # Default fallback (e.g., 1kHz loop rate)
                results['settings']['estimated_sampling_rate_hz'] = f"{sampling_rate} (Fallback)"
                logger.warning("Could not determine sampling rate from index. Using fallback: 1000 Hz.")
        else: # Sampling rate provided externally
             results['settings']['provided_sampling_rate_hz'] = sampling_rate

        results['settings']['actual_sampling_rate_used'] = sampling_rate
        if sampling_rate <= 0:
            logger.error("Invalid sampling rate <= 0.")
            return {"spectral": {"error": "Invalid sampling rate.", "spectra": {}, "settings": results['settings']}}

        logger.debug(f"Using sampling rate: {sampling_rate:.2f} Hz")
        nyquist_freq = sampling_rate / 2.0


        # --- Select Signals for FFT ---
        # Prioritize filtered gyro if available, fallback to ADC
        signals_to_analyze = {}
        for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
            gyro_smooth_col = find_column(df, [f"gyro{axis_name.capitalize()}"])
            gyro_adc_col = find_column(df, [f"gyroADC[{axis_idx}]"])

            if gyro_smooth_col and gyro_smooth_col in df:
                 signals_to_analyze[f"gyro{axis_name.capitalize()}"] = df[gyro_smooth_col]
            elif gyro_adc_col and gyro_adc_col in df:
                 signals_to_analyze[f"gyroADC[{axis_idx}]"] = df[gyro_adc_col]
            else:
                 logger.warning(f"No suitable gyro column found for {axis_name} axis.")

        if not signals_to_analyze:
             logger.error("No gyro columns found for spectral analysis.")
             return {"spectral": {"error": "No gyro data found.", "spectra": {}, "settings": results['settings']}}


        # --- Perform FFT for each signal ---
        max_points_for_json = 1000 # Limit data points saved in JSON/DB

        for name, signal_data_pd in signals_to_analyze.items():
            axis_results = {}
            signal_data = signal_data_pd.dropna().values # Get numpy array, drop NaNs

            n = len(signal_data)
            if n < 10: # Need a minimum number of points for meaningful FFT
                logger.warning(f"Not enough data points ({n}) for FFT on '{name}'.")
                axis_results["error"] = f"Insufficient data points ({n})."
                results['spectra'][name] = axis_results
                continue

            try:
                # Apply a window function (Hann) to reduce spectral leakage
                windowed_signal = signal_data * signal.windows.hann(n)

                # Perform FFT
                fft_result = fft(windowed_signal)
                # Calculate frequencies corresponding to the FFT output
                freqs = fftfreq(n, 1 / sampling_rate)

                # Keep only positive frequencies (and zero)
                positive_freq_mask = (freqs >= 0) & (freqs <= nyquist_freq)
                freqs_pos = freqs[positive_freq_mask]
                fft_magnitude = np.abs(fft_result[positive_freq_mask])

                # Normalize magnitude (scaling factor depends on convention, often 2/N for amplitude)
                if n > 0:
                    fft_magnitude = fft_magnitude * (2 / n)
                    fft_magnitude[0] = fft_magnitude[0] / 2 # DC component (0 Hz) is not doubled
                else:
                     fft_magnitude = np.array([])


                # --- Peak Detection ---
                if len(freqs_pos) > 1 and len(fft_magnitude) > 1:
                     # Dynamic height threshold based on median magnitude (robust to spikes)
                     median_mag = np.median(fft_magnitude[freqs_pos > 5]) # Ignore low freq noise for threshold
                     height_threshold = max(median_mag * 3, 0.05) # At least 3x median or 0.05 abs

                     # Dynamic distance based on frequency resolution
                     freq_resolution = freqs_pos[1] - freqs_pos[0] if len(freqs_pos) > 1 else 1.0
                     distance_hz = 5 # Minimum separation of 5 Hz
                     distance_samples = max(1, int(distance_hz / freq_resolution))

                     try:
                         peaks_indices, _ = signal.find_peaks(
                             fft_magnitude,
                             height=height_threshold,
                             distance=distance_samples
                         )

                         peak_freqs = freqs_pos[peaks_indices]
                         peak_mags = fft_magnitude[peaks_indices]

                         # Sort peaks by magnitude (descending) and take top N
                         sorted_indices = np.argsort(peak_mags)[::-1]
                         top_n = 5
                         axis_results["dominant_peaks_hz_mag"] = list(zip(
                             np.round(peak_freqs[sorted_indices][:top_n], 1),
                             np.round(peak_mags[sorted_indices][:top_n], 3)
                         ))
                     except Exception as e_peak:
                         logger.error(f"Peak finding failed for {name}: {e_peak}")
                         axis_results["dominant_peaks_error"] = f"Peak finding failed: {e_peak}"
                         axis_results["dominant_peaks_hz_mag"] = []

                else: # Not enough points for peak finding
                     axis_results["dominant_peaks_hz_mag"] = []

                # --- Frequency Band Analysis ---
                # Define bands relevant to drone tuning
                bands = {
                    "low_(<20Hz)": (0, 20),          # Prop wash, slow oscillations
                    "mid_(20-80Hz)": (20, 80),      # P/D term tuning issues, minor vibrations
                    "high_(80-200Hz)": (80, 200),    # D term noise, motor/frame vibrations
                    "vhigh_(>200Hz)": (200, nyquist_freq) # Higher frequency noise floor
                }
                band_avg_magnitude = {}
                if len(freqs_pos) > 0:
                    for band_name, (low, high) in bands.items():
                        mask = (freqs_pos >= low) & (freqs_pos < high)
                        if np.any(mask):
                            band_avg_magnitude[band_name] = round(float(np.mean(fft_magnitude[mask])), 4)
                        else:
                            band_avg_magnitude[band_name] = 0.0
                axis_results["band_avg_magnitude"] = band_avg_magnitude


                # --- Store Subset for Plotting ---
                if len(freqs_pos) > max_points_for_json:
                    # Downsample for storage/plotting: take evenly spaced points
                    indices = np.linspace(0, len(freqs_pos) - 1, max_points_for_json, dtype=int)
                    axis_results["frequencies_hz"] = freqs_pos[indices].tolist()
                    axis_results["magnitude"] = fft_magnitude[indices].tolist()
                elif len(freqs_pos) > 0:
                    axis_results["frequencies_hz"] = freqs_pos.tolist()
                    axis_results["magnitude"] = fft_magnitude.tolist()
                else:
                     axis_results["frequencies_hz"] = []
                     axis_results["magnitude"] = []

                # --- Store results for this axis ---
                results['spectra'][name] = axis_results

            except Exception as e_fft:
                 logger.error(f"FFT analysis failed for '{name}': {e_fft}", exc_info=True)
                 results['spectra'][name] = {"error": f"FFT analysis failed: {e_fft}"}


        # --- Throttle vs Frequency Heatmap (Optional, requires throttle data) ---
        throttle_col = find_column(df, ["rcCommand[3]", "rcCommandThrottle", "throttle"])
        if throttle_col and throttle_col in df and signals_to_analyze:
             logger.debug("Attempting Throttle vs Frequency heatmap generation...")
             try:
                 throttle_data = df[throttle_col].dropna()
                 # Choose one gyro axis for the heatmap (e.g., Roll)
                 primary_gyro_key = next(iter(signals_to_analyze.keys())) # Get the first analyzed gyro key
                 gyro_data_pd = signals_to_analyze[primary_gyro_key]

                 # Align throttle and gyro data
                 aligned_gyro, aligned_throttle = gyro_data_pd.align(throttle_data, join='inner')

                 if len(aligned_gyro) > 100: # Need sufficient aligned data
                     n_fft = 256 # FFT window size
                     n_overlap = int(n_fft * 0.75) # Overlap (e.g., 75%)
                     n_throttle_bins = 10 # Number of throttle bins

                     # Calculate Spectrogram (Frequency vs Time)
                     frequencies, times, Sxx = signal.spectrogram(
                         aligned_gyro.values,
                         fs=sampling_rate,
                         window='hann',
                         nperseg=n_fft,
                         noverlap=n_overlap,
                         scaling='density' # Power Spectral Density
                     )

                     # Interpolate throttle values to match spectrogram time points
                     spectrogram_time_indices = df.index.searchsorted(times + aligned_gyro.index[0]) # Map times back to original index
                     spectrogram_time_indices = np.clip(spectrogram_time_indices, 0, len(df) - 1)
                     throttle_at_times = df[throttle_col].iloc[spectrogram_time_indices].values

                     # Create throttle bins
                     min_thr, max_thr = np.min(throttle_at_times), np.max(throttle_at_times)
                     throttle_bins = np.linspace(min_thr, max_thr, n_throttle_bins + 1)
                     throttle_bin_centers = (throttle_bins[:-1] + throttle_bins[1:]) / 2

                     # Assign each time segment to a throttle bin
                     throttle_indices = np.digitize(throttle_at_times, throttle_bins[1:-1], right=True) # bin index for each time segment

                     # Average magnitude within each throttle bin and frequency bin
                     heatmap_matrix = np.zeros((n_throttle_bins, len(frequencies)))
                     counts = np.zeros(n_throttle_bins)

                     for t_idx, bin_idx in enumerate(throttle_indices):
                         if bin_idx < n_throttle_bins: # Ensure index is within bounds
                             heatmap_matrix[bin_idx, :] += Sxx[:, t_idx] # Sum PSD for this time segment
                             counts[bin_idx] += 1

                     # Average the PSD in bins with data
                     valid_bins = counts > 0
                     heatmap_matrix[valid_bins, :] /= counts[valid_bins, np.newaxis]
                     heatmap_matrix[~valid_bins, :] = np.nan # Mark bins with no data as NaN

                     # Store heatmap data (limit frequency range and size for JSON)
                     freq_mask = frequencies <= 500 # Limit to 500 Hz
                     results["throttle_freq_heatmap"] = {
                         "gyro_axis": primary_gyro_key,
                         "frequency_bins_hz": np.round(frequencies[freq_mask], 1).tolist(),
                         "throttle_bins": np.round(throttle_bin_centers, 2).tolist(),
                         # Convert to magnitude (sqrt) and handle NaNs before serialization
                         "magnitude_matrix": make_serializable(np.sqrt(heatmap_matrix[:, freq_mask]))
                     }
                     logger.debug("Throttle vs Frequency heatmap generated.")
                 else:
                     results["throttle_freq_heatmap_warning"] = "Insufficient aligned gyro/throttle data for heatmap."
                     logger.warning(results["throttle_freq_heatmap_warning"])

             except Exception as e_heatmap:
                 logger.error(f"Error generating throttle vs frequency heatmap: {e_heatmap}", exc_info=True)
                 results["throttle_freq_heatmap_error"] = f"Error generating heatmap: {e_heatmap}"


        logger.debug("Spectral analysis finished.")
        return {"spectral": results} # Return nested

    def analyze_gyro_accel(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing gyro & accelerometer data...")
        results = {'gyro': {}, 'accel': {}}
        gyro_found = False
        accel_found = False

        # --- Gyro Analysis ---
        for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
            # Prefer filtered gyro, fallback to ADC
            gyro_col = find_column(df, [f"gyro{axis_name.capitalize()}", f"gyroADC[{axis_idx}]"])
            if gyro_col and gyro_col in df and pd.api.types.is_numeric_dtype(df[gyro_col]):
                 gyro_found = True
                 data = df[gyro_col].dropna()
                 if not data.empty:
                     results['gyro'][f"{axis_name}_mean"] = float(data.mean())
                     results['gyro'][f"{axis_name}_std"] = float(data.std())
                     results['gyro'][f"{axis_name}_min"] = float(data.min())
                     results['gyro'][f"{axis_name}_max"] = float(data.max())
                     # Estimate noise using high-frequency standard deviation or diff mean
                     diff_data = data.diff().dropna()
                     if not diff_data.empty:
                         # Use MAD (Median Absolute Deviation) for robustness
                         results['gyro'][f"{axis_name}_noise_mad"] = float(diff_data.mad())
                         results['gyro'][f"{axis_name}_noise_std"] = float(diff_data.std())

                 else:
                     results['gyro'][f"{axis_name}_error"] = "No data"
            else:
                 results['gyro'][f"{axis_name}_error"] = "Column missing or not numeric"


        # --- Accelerometer Analysis ---
        for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
             # Prefer smoothed accel, fallback to raw if available (though less common in std logs)
             accel_col = find_column(df, [f"accSmooth[{axis_idx}]", f"accel{axis_name.capitalize()}"])
             if accel_col and accel_col in df and pd.api.types.is_numeric_dtype(df[accel_col]):
                 accel_found = True
                 data = df[accel_col].dropna()
                 if not data.empty:
                     results['accel'][f"{axis_name}_mean"] = float(data.mean())
                     results['accel'][f"{axis_name}_std"] = float(data.std())
                     results['accel'][f"{axis_name}_min"] = float(data.min())
                     results['accel'][f"{axis_name}_max"] = float(data.max())
                     diff_data = data.diff().dropna()
                     if not diff_data.empty:
                         results['accel'][f"{axis_name}_noise_mad"] = float(diff_data.mad())
                         results['accel'][f"{axis_name}_noise_std"] = float(diff_data.std())
                 else:
                     results['accel'][f"{axis_name}_error"] = "No data"
             else:
                 results['accel'][f"{axis_name}_error"] = "Column missing or not numeric"

        # Calculate overall vector magnitude if possible
        acc_cols_found = [find_column(df, [f"accSmooth[{i}]", f"accel{ax.capitalize()}"]) for i, ax in enumerate(['roll', 'pitch', 'yaw'])]
        if all(c and c in df for c in acc_cols_found):
             accel_found = True
             try:
                 acc_data = df[acc_cols_found].dropna(how='any')
                 if not acc_data.empty:
                      acc_vector_mag = np.sqrt(np.sum(np.square(acc_data), axis=1))
                      results['accel']["vector_mag_mean"] = float(acc_vector_mag.mean())
                      results['accel']["vector_mag_std"] = float(acc_vector_mag.std())
                      results['accel']["vector_mag_min"] = float(acc_vector_mag.min())
                      results['accel']["vector_mag_max"] = float(acc_vector_mag.max())
                      # Jerk (derivative of acceleration magnitude)
                      time_step = None
                      if isinstance(df.index, pd.TimedeltaIndex) or pd.api.types.is_numeric_dtype(df.index.dtype):
                           time_diffs = np.diff(df.index)
                           if len(time_diffs) > 0: time_step = np.median(time_diffs)
                      if time_step and time_step > 1e-9:
                           jerk = acc_vector_mag.diff() / time_step
                           results['accel']["jerk_mean_abs"] = float(jerk.abs().mean())
                           results['accel']["jerk_std"] = float(jerk.std())
                 else:
                      results['accel']["vector_error"] = "No overlapping accel data"
             except Exception as e_vec:
                  logger.error(f"Error calculating accel vector magnitude: {e_vec}")
                  results['accel']["vector_error"] = f"Calculation error: {e_vec}"
        else:
             results['accel']["vector_error"] = "Missing one or more accel axis columns"


        if not gyro_found: results['gyro']["error_overall"] = "No gyro data found."
        if not accel_found: results['accel']["error_overall"] = "No accelerometer data found."

        logger.debug("Gyro/Accel analysis finished.")
        return results # Return combined dict

    def analyze_rc_commands(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing RC commands...")
        results = {'stats': {}, 'pilot_style': {}}
        rc_found = False

        rc_cols_map = {}
        for i, axis_name in enumerate(['Roll', 'Pitch', 'Yaw', 'Throttle']):
             col = find_column(df, [f"rcCommand{axis_name}", f"rcCommand[{i}]"])
             if col and col in df and pd.api.types.is_numeric_dtype(df[col]):
                 rc_cols_map[axis_name] = col
                 rc_found = True
             else:
                 results['stats'][f"{axis_name}_error"] = "Column missing or not numeric"

        if not rc_found:
            results["error_overall"] = "No valid RC command columns found."
            logger.error(results["error_overall"])
            return {"rc_commands": results} # Return nested

        time_step = None
        if isinstance(df.index, pd.TimedeltaIndex) or pd.api.types.is_numeric_dtype(df.index.dtype):
            time_diffs = np.diff(df.index)
            if len(time_diffs) > 0:
                time_step = np.median(time_diffs)

        for axis_name, col in rc_cols_map.items():
            data = df[col].dropna()
            if not data.empty:
                results['stats'][f"{axis_name}_mean"] = float(data.mean())
                results['stats'][f"{axis_name}_std"] = float(data.std())
                results['stats'][f"{axis_name}_min"] = float(data.min())
                results['stats'][f"{axis_name}_max"] = float(data.max())

                # --- Pilot Style Analysis ---
                if axis_name != 'Throttle' and time_step and time_step > 1e-9:
                     # Stick movement rate (derivative)
                     rate = data.diff().abs() / time_step
                     rate = rate.dropna()
                     if not rate.empty:
                         results['pilot_style'][f"{axis_name}_stick_rate_mean_abs"] = float(rate.mean())
                         results['pilot_style'][f"{axis_name}_stick_rate_max"] = float(rate.max())
                         results['pilot_style'][f"{axis_name}_stick_rate_95p"] = float(rate.quantile(0.95)) # 95th percentile

                     # Stick Reversals (sign changes in derivative)
                     direction = np.sign(data.diff().dropna())
                     reversals = direction.diff().abs() / 2.0 # Change from +1 to -1 or vice versa is diff 2
                     if not reversals.empty:
                          # Rate of reversals per second
                          reversal_freq = reversals.sum() / (df.index[-1] - df.index[0]) if (df.index[-1] - df.index[0]) > 0 else 0
                          results['pilot_style'][f"{axis_name}_reversal_freq_hz"] = float(reversal_freq)


                # Center Focus (for Roll, Pitch, Yaw)
                if axis_name != 'Throttle':
                     center_val = data.median() # Estimate center
                     # Define a 'center zone' (e.g., +/- 5% of typical range, assuming ~500 units range -> +/- 25)
                     center_range = 25 # Adjust as needed based on typical RC range
                     in_center = data[abs(data - center_val) < center_range]
                     results['pilot_style'][f"{axis_name}_center_focus_pct"] = float(len(in_center) / len(data) * 100) if len(data)>0 else 0.0


        # --- Aggregate Pilot Style Assessment ---
        # Simple assessment based on stick rates and reversals
        avg_stick_rate = np.mean([results['pilot_style'].get(f"{ax}_stick_rate_mean_abs", 0) for ax in ['Roll', 'Pitch', 'Yaw']])
        avg_reversal_freq = np.mean([results['pilot_style'].get(f"{ax}_reversal_freq_hz", 0) for ax in ['Roll', 'Pitch', 'Yaw']])

        if avg_stick_rate > 0: # Check if calculated
             # Smoothness (lower rate, fewer reversals = smoother)
             if avg_stick_rate < 1000 and avg_reversal_freq < 2: results['pilot_style']["smoothness_assessment"] = "Smooth"
             elif avg_stick_rate < 2500 and avg_reversal_freq < 5: results['pilot_style']["smoothness_assessment"] = "Moderate"
             else: results['pilot_style']["smoothness_assessment"] = "Jerky/Aggressive"

             # Aggression (higher rate = more aggressive)
             if avg_stick_rate > 3000: results['pilot_style']["aggression_assessment"] = "High"
             elif avg_stick_rate > 1500: results['pilot_style']["aggression_assessment"] = "Moderate"
             else: results['pilot_style']["aggression_assessment"] = "Low"


        logger.debug("RC command analysis finished.")
        return {"rc_commands": results} # Return nested

    def analyze_altitude_power(self, df: pd.DataFrame) -> dict:
        # Combined analysis for battery and altitude, as they are often related
        logger.debug("Analyzing altitude and power...")
        results = {'altitude': {}, 'power': {}}
        alt_found = False
        power_found = False

        # --- Altitude Analysis ---
        # Prefer Baro, fallback to GPS Z coord if available
        alt_col = find_column(df, ["baroAlt", "baroAltitude"]) # Barometer altitude
        gps_z_col = find_column(df, ["gpsCartesianCoords[2]", "gpsAltitude"]) # GPS altitude (can be noisy)

        primary_alt_col = None
        if alt_col and alt_col in df and pd.api.types.is_numeric_dtype(df[alt_col]):
            primary_alt_col = alt_col
            results['altitude']['source'] = 'Barometer'
        elif gps_z_col and gps_z_col in df and pd.api.types.is_numeric_dtype(df[gps_z_col]):
            primary_alt_col = gps_z_col
            results['altitude']['source'] = 'GPS'
            logger.info("Using GPS altitude for analysis as Barometer data is unavailable.")

        if primary_alt_col:
             alt_found = True
             alt_data = df[primary_alt_col].dropna()
             if not alt_data.empty:
                 results['altitude']["altitude_min"] = float(alt_data.min())
                 results['altitude']["altitude_max"] = float(alt_data.max())
                 results['altitude']["altitude_mean"] = float(alt_data.mean())
                 results['altitude']["altitude_range"] = float(alt_data.max() - alt_data.min())

                 # Vertical Speed (derivative of altitude)
                 time_step = None
                 if isinstance(df.index, pd.TimedeltaIndex) or pd.api.types.is_numeric_dtype(df.index.dtype):
                     time_diffs = np.diff(df.index)
                     if len(time_diffs) > 0: time_step = np.median(time_diffs)

                 if time_step and time_step > 1e-9:
                     vertical_speed = alt_data.diff() / time_step
                     vertical_speed = vertical_speed.dropna()
                     if not vertical_speed.empty:
                          results['altitude']["vertical_speed_mean_abs_mps"] = float(vertical_speed.abs().mean())
                          results['altitude']["vertical_speed_max_mps"] = float(vertical_speed.max())
                          results['altitude']["vertical_speed_min_mps"] = float(vertical_speed.min())

             else:
                 results['altitude']["error"] = "Altitude data is empty"
        else:
            results['altitude']["error"] = "No suitable altitude column found (Baro or GPS)."


        # --- Power Analysis ---
        volt_col = find_column(df, ['vbatLatest', 'vbat', 'voltage'])
        curr_col = find_column(df, ['amperageLatest', 'current', 'currentSensor']) # 'amperageLatest' is often 100*A

        voltage_data = None
        current_data = None # Amps
        power_watts = None

        if volt_col and volt_col in df and pd.api.types.is_numeric_dtype(df[volt_col]):
             power_found = True # Need at least voltage
             voltage_data = df[volt_col].dropna()
             if not voltage_data.empty:
                 results['power']["voltage_mean"] = float(voltage_data.mean())
                 results['power']["voltage_min"] = float(voltage_data.min())
                 results['power']["voltage_max"] = float(voltage_data.max())
                 if results['power']["voltage_max"] > 0.1: # Avoid division by zero/small numbers
                     sag_abs = results['power']["voltage_max"] - results['power']["voltage_min"]
                     sag_pct = (sag_abs / results['power']["voltage_max"]) * 100
                     results['power']["voltage_sag_absolute"] = round(float(sag_abs), 2)
                     results['power']["voltage_sag_percent"] = round(float(sag_pct), 1)
             else:
                  results['power']["voltage_error"] = "Voltage data empty"
        else:
            results['power']["voltage_error"] = "Voltage column missing or not numeric"


        if curr_col and curr_col in df and pd.api.types.is_numeric_dtype(df[curr_col]):
             current_raw = df[curr_col].dropna()
             if not current_raw.empty:
                 # Check if current looks like 100*A (common in older BF)
                 # If max current > 10000, assume it's amperageLatest format
                 if current_raw.max() > 10000 and 'amperagelatest' in curr_col.lower():
                      current_data = current_raw / 100.0 # Convert cA to A
                      results['power']['current_unit'] = 'Amps (converted from cA)'
                 else:
                      current_data = current_raw
                      results['power']['current_unit'] = 'Amps (assumed)'

                 power_found = True
                 results['power']["current_mean"] = float(current_data.mean())
                 results['power']["current_max"] = float(current_data.max())
                 results['power']["current_min"] = float(current_data.min()) # Useful for checking sensor floor noise

             else:
                 results['power']["current_error"] = "Current data empty"
        else:
            results['power']["current_error"] = "Current column missing or not numeric"


        # Calculate Power (Watts) if both V and I are available
        if voltage_data is not None and current_data is not None:
             # Align voltage and current before calculating power
             aligned_v, aligned_i = voltage_data.align(current_data, join='inner')
             if not aligned_v.empty:
                 power_watts = aligned_v * aligned_i
                 results['power']["power_mean_watts"] = float(power_watts.mean())
                 results['power']["power_max_watts"] = float(power_watts.max())
                 power_found = True
             else:
                 results['power']["power_error"] = "No overlapping Voltage/Current data."

        if not alt_found and not power_found:
             results["error_overall"] = "No Altitude or Power data found."

        logger.debug("Altitude/Power analysis finished.")
        return results # Return combined dict


    def analyze_flight_trajectory(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing flight trajectory (from GPS)...")
        results = {}
        # Check for necessary GPS Cartesian coordinate columns
        x_col = find_column(df, ['gpsCartesianCoords[0]', 'gpsCoordX'])
        y_col = find_column(df, ['gpsCartesianCoords[1]', 'gpsCoordY'])
        z_col = find_column(df, ['gpsCartesianCoords[2]', 'gpsCoordZ', 'gpsAltitude']) # Use Z or altitude

        if x_col and y_col and z_col and x_col in df and y_col in df and z_col in df:
            # Extract and clean data
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            z_data = df[z_col].dropna()

            # Align all three axes based on time index
            aligned_x, aligned_y = x_data.align(y_data, join='inner')
            aligned_x, aligned_z = aligned_x.align(z_data, join='inner')
            # Now align y and z with the result of x vs z alignment
            aligned_y, aligned_z = y_data.align(aligned_z, join='inner')
            aligned_x, aligned_y = aligned_x.align(aligned_y, join='inner')


            if len(aligned_x) > 10: # Need sufficient points for meaningful analysis
                 x = aligned_x.to_numpy()
                 y = aligned_y.to_numpy()
                 z = aligned_z.to_numpy()
                 time_index = aligned_x.index # Get the common time index

                 # Calculate distances between consecutive points
                 distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
                 total_distance = np.sum(distances)
                 results["total_distance_m"] = round(float(total_distance), 2)

                 # Calculate time differences and speed
                 if isinstance(time_index, pd.TimedeltaIndex):
                     time_secs = time_index.total_seconds()
                 elif pd.api.types.is_numeric_dtype(time_index.dtype):
                      time_secs = time_index.to_numpy() # Assume already in seconds based on prepare_data
                 else:
                      results["speed_error"] = "Cannot calculate speed due to non-numeric time index."
                      time_secs = None

                 if time_secs is not None:
                     time_diffs = np.diff(time_secs)
                     valid_time_diffs = time_diffs[time_diffs > 1e-9] # Avoid zero/negative time steps

                     if len(valid_time_diffs) > 0 and len(valid_time_diffs) == len(distances):
                         speeds = distances / valid_time_diffs # Speed in m/s
                         results["average_speed_mps"] = round(float(np.mean(speeds)), 2)
                         results["max_speed_mps"] = round(float(np.max(speeds)), 2)
                         results["speed_std_dev"] = round(float(np.std(speeds)), 2)
                         total_time = time_secs[-1] - time_secs[0]
                         results["flight_duration_s"] = round(float(total_time), 2)
                         # Calculate speed excluding potential outliers (e.g., 95th percentile)
                         results["speed_95percentile_mps"] = round(float(np.percentile(speeds, 95)), 2)
                     else:
                         results["speed_error"] = "Mismatch between distance and time diff array lengths or invalid time steps."

                 # Bounding box
                 results["x_range_m"] = [round(float(x.min()), 1), round(float(x.max()), 1)]
                 results["y_range_m"] = [round(float(y.min()), 1), round(float(y.max()), 1)]
                 results["z_range_m"] = [round(float(z.min()), 1), round(float(z.max()), 1)]
                 results["max_displacement_from_origin_m"] = round(float(np.max(np.sqrt(x**2 + y**2 + z**2))), 1)

            else:
                 results["error"] = "Insufficient overlapping GPS coordinate data points."
        else:
            results["error"] = "GPS Cartesian coordinate columns (e.g., gpsCartesianCoords[0/1/2]) not found."
            logger.info(results["error"])

        logger.debug("Flight trajectory analysis finished.")
        return {"trajectory": results} # Return nested

    def analyze_acceleration(self, df: pd.DataFrame) -> dict:
        # This is the first (more detailed) definition - removing the duplicate.
        logger.debug("Analyzing accelerometer data...")
        results = {}
        # Use find_column for robustness
        acc_cols = [find_column(df, [f"accSmooth[{i}]", f"accel{ax.capitalize()}"]) for i, ax in enumerate(['Roll', 'Pitch', 'Yaw'])]
        available_acc = [col for col in acc_cols if col is not None and col in df and pd.api.types.is_numeric_dtype(df[col])]

        if len(available_acc) == 3: # Require all 3 axes
            acc_data = df[available_acc].dropna(how='all')
            if not acc_data.empty:
                 # Align data just in case of NaNs
                 acc_data = acc_data.dropna(how='any') # Drop rows where any axis is NaN for vector calc
                 if not acc_data.empty:
                     acc_vector = np.sqrt(np.sum(np.square(acc_data), axis=1))
                     results["vector_mag_mean"] = round(float(np.mean(acc_vector)), 3)
                     results["vector_mag_max"] = round(float(np.max(acc_vector)), 3)
                     results["vector_mag_min"] = round(float(np.min(acc_vector)), 3)
                     results["vector_mag_std"] = round(float(np.std(acc_vector)), 3)

                     # Jerk Calculation (needs time index)
                     time_step = None
                     if isinstance(df.index, pd.TimedeltaIndex) or pd.api.types.is_numeric_dtype(df.index.dtype):
                          time_diffs = np.diff(df.index)
                          if len(time_diffs) > 0: time_step = np.median(time_diffs)

                     if time_step and time_step > 1e-9:
                          # Align vector mag with time index if needed (should be aligned if acc_data was)
                          acc_vector_s = pd.Series(acc_vector, index=acc_data.index)
                          jerk = acc_vector_s.diff().abs() / time_step
                          jerk = jerk.dropna()
                          if not jerk.empty:
                              results["jerk_mean_abs"] = round(float(np.mean(jerk)), 3)
                              results["jerk_max"] = round(float(np.max(jerk)), 3)
                              results["jerk_std"] = round(float(np.std(jerk)), 3)
                          else: results["jerk_error"] = "No valid jerk data"
                     else: results["jerk_error"] = "Cannot calculate jerk without valid time step"
                 else: results["error"] = "No overlapping accelerometer data points."
            else: results["error"] = "Accelerometer data columns are empty."
        elif available_acc:
             results["error"] = f"Missing some accelerometer axes. Found: {available_acc}. Need all 3 for vector analysis."
             # Optionally analyze individual axes here if needed
        else:
            results["error"] = "No valid accelerometer columns (accSmooth[0/1/2]) found or data not numeric."
            logger.info(results["error"])

        logger.debug("Acceleration analysis finished.")
        return {"acceleration": results} # Return nested


    def analyze_control_effort(self, df: pd.DataFrame) -> dict:
         # This is the first (more detailed) definition - removing the duplicate.
        logger.debug("Analyzing control effort (PID terms)...")
        results = {}
        pid_output_cols_found = []
        effort_metrics = {}

        # Find PID output columns
        for term in ['P', 'I', 'D', 'F']:
             for axis_idx, axis_name in enumerate(['roll', 'pitch', 'yaw']):
                 # Common naming conventions
                 col = find_column(df, [f"axis{term}[{axis_idx}]", f"pid{term}{axis_name.capitalize()}"])
                 if col and col in df and pd.api.types.is_numeric_dtype(df[col]):
                     pid_output_cols_found.append(col)
                     # Store individual term stats if needed
                     data = df[col].dropna()
                     if not data.empty:
                          effort_metrics[f"{col}_mean_abs"] = float(data.abs().mean())
                          effort_metrics[f"{col}_max_abs"] = float(data.abs().max())
                          effort_metrics[f"{col}_std"] = float(data.std())


        if pid_output_cols_found:
            pid_data = df[pid_output_cols_found].dropna(how='all')
            if not pid_data.empty:
                 # Calculate total absolute effort (sum of absolute values of all terms)
                 total_abs_effort = pid_data.abs().sum(axis=1) # Sum across columns for each time step
                 results["total_abs_effort_mean"] = round(float(total_abs_effort.mean()), 3)
                 results["total_abs_effort_max"] = round(float(total_abs_effort.max()), 3)
                 results["total_abs_effort_std"] = round(float(total_abs_effort.std()), 3)

                 # Include individual term stats
                 results["individual_term_stats"] = {k: round(v, 3) for k, v in effort_metrics.items()}

                 # Optionally, calculate contribution percentage of each term?
                 # mean_abs_p = np.mean([v for k,v in effort_metrics.items() if k.startswith('axisP') and k.endswith('_mean_abs')]) ... etc.
                 # total_mean_abs = sum of mean_abs_p, i, d, f
                 # p_contrib_pct = (mean_abs_p / total_mean_abs) * 100 if total_mean_abs > 0 else 0 ...

            else:
                results["error"] = "PID output columns found but contain only NaN values."
        else:
            results["error"] = "No valid PID output columns (axisP/I/D/F[0/1/2]) found or data not numeric."
            logger.info(results["error"])

        logger.debug("Control effort analysis finished.")
        return {"control_effort": results} # Return nested


    def analyze_rc_vs_gyro(self, df: pd.DataFrame) -> dict:
        logger.debug("Analyzing RC vs Gyro latency...")
        results = {}

        # Determine time step
        time_step = None
        if isinstance(df.index, pd.TimedeltaIndex) or pd.api.types.is_numeric_dtype(df.index.dtype):
            time_diffs = np.diff(df.index)
            if len(time_diffs) > 0:
                time_step = np.nanmedian(time_diffs)
                if time_step <= 1e-9:
                    time_step = None  # Invalid step
        # Fallback if index didn't work
        if not time_step:
            time_col = find_column(df, ["time"])  # Assuming 'time' column exists and is in seconds
            if time_col and time_col in df and pd.api.types.is_numeric_dtype(df[time_col]):
                time_diffs = df[time_col].diff().dropna()
                if not time_diffs.empty:
                    time_step = time_diffs.median()
                    if time_step <= 1e-9:
                        time_step = None

        if not time_step:
            logger.warning("Cannot determine time step for RC vs Gyro lag analysis.")
            return {"rc_gyro_latency": {"error": "Cannot determine time step."}}

        results['time_step_used_s'] = time_step
        lag_found = False

        for axis_idx, axis_name in enumerate(['Roll', 'Pitch', 'Yaw']):
            rc_col = find_column(df, [f"rcCommand{axis_name}", f"rcCommand[{axis_idx}]"])
            gyro_col = find_column(df, [f"gyro{axis_name}", f"gyroADC[{axis_idx}]"])  # Use filtered gyro if available

            if rc_col and gyro_col and rc_col in df and gyro_col in df:
                rc_data = df[rc_col].dropna()
                gyro_data = df[gyro_col].dropna()

                # Align data
                rc_data, gyro_data = rc_data.align(gyro_data, join='inner')

                if len(rc_data) > 100:  # Need sufficient overlapping data
                    try:
                        # --- Cross-Correlation Method ---
                        # Normalize signals to have zero mean and unit variance
                        rc_norm = (rc_data - rc_data.mean()) / rc_data.std()
                        gyro_norm = (gyro_data - gyro_data.mean()) / gyro_data.std()

                        # Calculate cross-correlation
                        correlation = signal.correlate(gyro_norm, rc_norm, mode='full')
                        # Lags corresponding to the correlation result
                        lags = signal.correlation_lags(len(gyro_norm), len(rc_norm), mode='full')

                        # Find the lag with the maximum correlation
                        lag_at_max_corr = lags[np.argmax(correlation)]
                        lag_time_ms = lag_at_max_corr * time_step * 1000  # Convert lag index to milliseconds

                        # Basic sanity check on lag time
                        if abs(lag_time_ms) < 500:  # Assume lag < 500ms
                            results[f'{axis_name.lower()}_lag_ms_xcorr'] = round(lag_time_ms, 1)
                            lag_found = True
                        else:
                            results[f'{axis_name.lower()}_lag_ms_xcorr'] = None  # Unrealistic lag
                            logger.warning(
                                f"Unrealistic lag ({lag_time_ms:.1f} ms) found for {axis_name} via cross-correlation."
                            )

                        # --- Peak Matching Method (as fallback or comparison) ---
                        # (Code similar to original, potentially less robust than xcorr)
                        rc_rate = rc_data.diff().abs() / time_step  # Look at absolute rate of change
                        gyro_rate = gyro_data.diff().abs() / time_step
                        rc_rate, gyro_rate = rc_rate.align(gyro_rate, join='inner')
                        rc_rate = rc_rate.dropna()
                        gyro_rate = gyro_rate.dropna()

                        if len(rc_rate) > 50:
                            rc_peaks, _ = signal.find_peaks(
                                rc_rate,
                                height=rc_rate.quantile(0.90),
                                distance=int(0.05 / time_step)
                            )  # 50ms distance
                            peak_lags_ms = []

                            for peak_idx in rc_peaks:
                                if peak_idx >= len(rc_rate):
                                    continue  # Boundary check
                                rc_peak_time = rc_rate.index[peak_idx]

                                # Search for corresponding gyro peak in a window *after* the RC peak
                                search_start_time = rc_peak_time  # Start search immediately
                                search_end_time = rc_peak_time + pd.Timedelta(seconds=0.15)  # Look 150ms ahead

                                gyro_search_window = gyro_rate.loc[search_start_time:search_end_time]

                                if not gyro_search_window.empty:
                                    # Find the *first* significant peak in the gyro window
                                    gyro_peaks_in_window, _ = signal.find_peaks(
                                        gyro_search_window,
                                        height=gyro_rate.quantile(0.75)
                                    )  # Lower threshold for response peak
                                    if len(gyro_peaks_in_window) > 0:
                                        gyro_peak_time = gyro_search_window.index[gyro_peaks_in_window[0]]
                                        # Calculate lag
                                        lag_val = gyro_peak_time - rc_peak_time
                                        # Convert lag to milliseconds (handle both Timedelta and numeric indices)
                                        if hasattr(lag_val, 'total_seconds'):
                                            lag_ms = lag_val.total_seconds() * 1000
                                        elif isinstance(lag_val, (int, float, np.number)):
                                            lag_ms = lag_val * 1000  # Assuming index was already seconds
                                        else:
                                            logger.warning(
                                                f"Could not convert lag value type {type(lag_val)} to ms."
                                            )
                                            lag_ms = None

                                        if lag_ms is not None and lag_ms > 0 and lag_ms < 150:
                                            # Add validity check for positive lag within window
                                            peak_lags_ms.append(lag_ms)

                            # --- End of peak matching loop ---
                            if peak_lags_ms:
                                median_lag_peak = round(np.nanmedian(peak_lags_ms), 1)
                                results[f'{axis_name.lower()}_lag_ms_peak'] = median_lag_peak
                                lag_found = True  # Mark lag as found if peak method works
                            else:
                                results[f'{axis_name.lower()}_lag_ms_peak'] = None
                        else:
                            results[f'{axis_name.lower()}_lag_error_peak'] = "Insufficient rate data for peak matching"

                    except Exception as e_lag_calc:
                        logger.error(
                            f"Error during {axis_name} lag calculation: {e_lag_calc}",
                            exc_info=True
                        )
                        results[f'{axis_name.lower()}_lag_error'] = f"Calculation Error: {e_lag_calc}"
                else:
                    results[f'{axis_name.lower()}_lag_error'] = "Insufficient overlapping data points (<100)"
            else:
                results[f'{axis_name.lower()}_lag_error'] = "RC Command or Gyro column missing"

        if not lag_found:
            results["error_overall"] = "Could not calculate lag for any axis."

        logger.debug("RC vs Gyro latency analysis finished.")
        return {"rc_gyro_latency": results}


    def diagnose_data_quality(self, df: pd.DataFrame, metadata: dict) -> dict:
        logger.debug("Diagnosing data quality...")
        diagnostics = {
            "missing_data": {},
            "data_range_issues": [],
            "sampling_issues": {},
            "quality_score": 1.0,  # Start with perfect score
            "issues_found": []     # List of human-readable issues
        }
        quality_deduction = 0.0

        # --- Check for Essential Columns ---
        essential_groups = {
            'time': ['time'],  # Handled by index usually, but check presence before prep
            'gyro': [f"gyro{ax}" for ax in ['Roll', 'Pitch', 'Yaw']] + [f"gyroADC[{i}]" for i in range(3)],
            'rc_command': [f"rcCommand{ax}" for ax in ['Roll', 'Pitch', 'Yaw', 'Throttle']] + [f"rcCommand[{i}]" for i in range(4)],
            'motors': [f"motor[{i}]" for i in range(4)]  # Assume at least 4 motors for basic flight
        }
        missing_essentials = []
        for group, alternatives in essential_groups.items():
            if not any(find_column(df, [alt]) for alt in alternatives):
                missing_essentials.append(group)

        if 'time' in missing_essentials and not isinstance(df.index, pd.Index):
            msg = "Essential 'time' information missing (column or index)."
            logger.error(msg)
            diagnostics['issues_found'].append(msg)
            quality_deduction += 0.5  # Major issue
        if 'gyro' in missing_essentials:
            msg = "Essential 'gyro' data missing (gyroRoll/Pitch/Yaw or gyroADC[0/1/2])."
            logger.error(msg)
            diagnostics['issues_found'].append(msg)
            quality_deduction += 0.3
        if 'motors' in missing_essentials:
            msg = "Essential 'motor' data missing (motor[0-3])."
            logger.warning(msg)  # Might be analysis of log without motor output enabled
            diagnostics['issues_found'].append(msg)
            quality_deduction += 0.1

        # --- Check Overall Missing Data Percentage ---
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        if total_cells > 0:
            missing_pct = (missing_cells / total_cells) * 100
            diagnostics['missing_data']['overall_missing_pct'] = round(missing_pct, 2)
            if missing_pct > 10:
                msg = f"High overall missing data: {missing_pct:.1f}%."
                logger.warning(msg)
                diagnostics['issues_found'].append(msg)
                quality_deduction += min(0.2, missing_pct / 50.0)  # Deduct up to 0.2 based on percentage
        else:
            logger.warning("DataFrame has zero size during quality check.")

        # --- Check Sampling Rate Consistency ---
        time_step = None
        time_std_dev_ratio = 0
        if isinstance(df.index, pd.TimedeltaIndex):
            time_diffs = np.diff(df.index.total_seconds())
        elif pd.api.types.is_numeric_dtype(df.index.dtype):
            time_diffs = np.diff(df.index)
        else:
            time_diffs = []

        if len(time_diffs) > 10:
            time_step = np.nanmedian(time_diffs)
            time_std_dev = np.nanstd(time_diffs)
            if time_step and time_step > 1e-9:
                time_std_dev_ratio = time_std_dev / time_step
                diagnostics['sampling_issues']['median_time_step_ms'] = round(time_step * 1000, 3)
                diagnostics['sampling_issues']['std_dev_time_step_ms'] = round(time_std_dev * 1000, 3)
                diagnostics['sampling_issues']['std_dev_ratio'] = round(time_std_dev_ratio, 3)
                if time_std_dev_ratio > 0.1:
                    msg = f"Irregular sampling detected (StdDev/Median time step = {time_std_dev_ratio:.2f} > 0.1)."
                    logger.warning(msg)
                    diagnostics['issues_found'].append(msg)
                    quality_deduction += 0.15

                # Check for large gaps (e.g., > 5x median step)
                large_gaps = time_diffs[time_diffs > 5 * time_step]
                if len(large_gaps) > 0:
                    msg = f"Found {len(large_gaps)} large gaps in time data (>{5 * time_step * 1000:.1f} ms)."
                    logger.warning(msg)
                    diagnostics['sampling_issues']['large_gap_count'] = len(large_gaps)
                    diagnostics['issues_found'].append(msg)
                    quality_deduction += 0.1
            else:
                diagnostics['sampling_issues']['error'] = "Could not reliably determine sampling rate."
                if 'time' not in missing_essentials:
                    quality_deduction += 0.1
        else:
            diagnostics['sampling_issues']['error'] = "Not enough data points to analyze sampling rate."

        # --- Check Data Ranges ---
        # Gyro range check (e.g., deg/s)
        gyro_cols_to_check = [
            col for col in df.columns
            if ('gyro' in col.lower() or 'gyroadc' in col.lower()) and pd.api.types.is_numeric_dtype(df[col])
        ]
        for col in gyro_cols_to_check:
            max_abs_gyro = df[col].abs().max()
            if max_abs_gyro > 3000:  # Expecting values typically below 2000-2500 deg/s
                msg = f"Gyro column '{col}' has unusually high absolute maximum value: {max_abs_gyro:.1f} deg/s."
                logger.warning(msg)
                diagnostics['data_range_issues'].append(msg)
                quality_deduction += 0.05

        # Motor range check (based on detected range if possible)
        motor_cols_to_check = [
            col for col in df.columns if 'motor' in col.lower() and pd.api.types.is_numeric_dtype(df[col])
        ]
        if motor_cols_to_check:
            min_motor = df[motor_cols_to_check].min().min()
            max_motor = df[motor_cols_to_check].max().max()
            # Crude check for wildly out-of-bounds values
            if min_motor < -100 or max_motor > 2500:
                msg = f"Motor values outside expected bounds (min: {min_motor:.1f}, max: {max_motor:.1f}). Check scale/corruption."
                logger.warning(msg)
                diagnostics['data_range_issues'].append(msg)
                quality_deduction += 0.1

        # --- Final Score and Summary ---
        diagnostics['quality_score'] = max(0.0, 1.0 - quality_deduction)  # Ensure score is between 0 and 1

        score = diagnostics['quality_score']
        if score >= 0.9:
            summary = "Excellent data quality."
        elif score >= 0.7:
            summary = "Good data quality, minor issues detected."
        elif score >= 0.5:
            summary = "Fair data quality, some analysis might be affected."
        elif score >= 0.3:
            summary = "Poor data quality, reliability of analysis may be low."
        else:
            summary = "Very poor data quality, analysis results are likely unreliable."
        diagnostics['summary'] = summary

        logger.info(f"Data Quality Score: {score:.2f}. Summary: {summary}")
        if diagnostics['issues_found']:
            logger.warning(f"Data quality issues found: {diagnostics['issues_found']}")

        return diagnostics


    def generate_flight_assessment(self, analysis_results: dict) -> dict:
        logger.debug("Generating flight assessment...")
        assessment = {
            "flight_quality": 0.5,
            "strengths": [],
            "weaknesses": [],
            "summary": "Assessment N/A"
        }
        factors = []  # Tuples of (quality_score_0_to_1, weight)

        # --- PID Tracking Quality ---
        pid_tracking = analysis_results.get('pid', {}).get('tracking_error', {})
        if pid_tracking:
            tracking_maes = [pid_tracking.get(f"{ax}_mae", None) for ax in ['roll', 'pitch', 'yaw']]
            valid_maes = [mae for mae in tracking_maes if mae is not None and pd.notna(mae)]
            if valid_maes:
                avg_mae = np.mean(valid_maes)
                # Score: Higher MAE = lower score. Scale: 0 MAE = 1.0, 20 MAE = 0.0 (linear)
                tracking_quality = max(0.0, min(1.0, 1.0 - (avg_mae / 20.0)))
                factors.append((tracking_quality, 0.4))  # Weight PID tracking heavily

                if tracking_quality >= 0.8:
                    assessment["strengths"].append(f"Good PID tracking (Avg MAE: {avg_mae:.2f} deg/s)")
                elif tracking_quality < 0.5:
                    assessment["weaknesses"].append(f"Poor PID tracking (Avg MAE: {avg_mae:.2f} deg/s)")
                else:
                    assessment["strengths"].append(f"Acceptable PID tracking (Avg MAE: {avg_mae:.2f} deg/s)")

        # --- Noise Levels (from Spectral Analysis) ---
        spectral = analysis_results.get('spectral', {}).get('spectra', {})
        if spectral:
            noise_levels = []
            for axis_name, axis_data in spectral.items():
                bands = axis_data.get('band_avg_magnitude', {})
                # Consider high frequency bands as noise indicators
                high_noise = bands.get("high_(80-200Hz)", 0) + bands.get("vhigh_(>200Hz)", 0)
                noise_levels.append(high_noise)
            if noise_levels:
                avg_noise = np.mean(noise_levels)
                # Score: Lower noise = higher score. Scale: 0 Noise = 1.0, 0.5 Noise = 0.0 (linear)
                noise_quality = max(0.0, min(1.0, 1.0 - (avg_noise / 0.5)))
                factors.append((noise_quality, 0.3))  # Weight noise moderately

                if noise_quality >= 0.8:
                    assessment["strengths"].append(f"Low gyro noise levels (Avg High Freq Mag: {avg_noise:.3f})")
                elif noise_quality < 0.5:
                    assessment["weaknesses"].append(f"High gyro noise levels (Avg High Freq Mag: {avg_noise:.3f})")

        # --- Motor Saturation ---
        motors = analysis_results.get('motors', {}).get('motors', {})  # Nested structure
        if motors:
            saturation = motors.get('motor_saturation_pct_overall_any', None)
            if saturation is not None and pd.notna(saturation):
                # Score: Lower saturation = higher score. Scale: 0% = 1.0, 20% = 0.0 (linear)
                saturation_quality = max(0.0, min(1.0, 1.0 - (saturation / 20.0)))
                factors.append((saturation_quality, 0.2))  # Weight saturation less heavily

                if saturation > 10:
                    assessment["weaknesses"].append(f"High motor saturation ({saturation:.1f}%)")
                elif saturation < 1:
                    assessment["strengths"].append(f"Low motor saturation ({saturation:.1f}%)")

        # --- Voltage Sag ---
        power = analysis_results.get('alt_power', {}).get('power', {})
        if power:
            sag_pct = power.get('voltage_sag_percent', None)
            if sag_pct is not None and pd.notna(sag_pct):
                # Score: Lower sag = higher score. Scale: 0% = 1.0, 25% = 0.0 (linear)
                sag_quality = max(0.0, min(1.0, 1.0 - (sag_pct / 25.0)))
                factors.append((sag_quality, 0.1))  # Weight sag less heavily

                if sag_pct > 15:
                    assessment["weaknesses"].append(f"Significant voltage sag ({sag_pct:.1f}%)")
                elif sag_pct < 5:
                    assessment["strengths"].append(f"Low voltage sag ({sag_pct:.1f}%)")

        # --- Calculate Final Weighted Score ---
        if factors:
            total_weight = sum(w for q, w in factors)
            weighted_sum = sum(q * w for q, w in factors)
            assessment["flight_quality"] = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            assessment["flight_quality"] = 0.0  # No factors to assess quality
            assessment["summary"] = "Could not assess flight quality due to missing analysis data."

        # --- Generate Summary Text ---
        fq = assessment["flight_quality"]
        if fq == 0 and factors:
            assessment["summary"] = "Poor flight performance indicated across multiple factors."
        elif fq > 0:
            if fq >= 0.85:
                assessment["summary"] = "Excellent flight performance detected. Well-tuned with good metrics."
            elif fq >= 0.65:
                assessment["summary"] = "Good flight performance. Minor tuning opportunities or improvements may exist."
            elif fq >= 0.45:
                assessment["summary"] = "Acceptable flight performance, but noticeable areas for improvement."
            else:
                assessment["summary"] = "Suboptimal flight performance. Significant tuning or setup issues likely present."

        if assessment["weaknesses"]:
            assessment["summary"] += f" Key weaknesses include: {'; '.join(assessment['weaknesses'][:2])}."
        elif assessment["strengths"] and fq > 0.6:
            assessment["summary"] += f" Key strengths include: {'; '.join(assessment['strengths'][:2])}."

        logger.info(f"Flight Assessment Score: {assessment['flight_quality']:.2f}")
        return assessment


    def identify_problem_patterns(self, analysis_results: dict, metadata: dict) -> list:
        logger.debug("Identifying problem patterns for tuning recommendations...")
        problems = []

        # --- Check Spectral Analysis for Oscillations ---
        spectral_data = analysis_results.get('spectral', {}).get('spectra', {})
        current_pids = metadata.get('pid_values', {})

        for axis_name_upper in ['Roll', 'Pitch', 'Yaw']:
            axis_name_lower = axis_name_upper.lower()
            gyro_key = None
            # Find the gyro key used in spectral analysis (could be gyroRoll or gyroADC[0], etc.)
            for key in spectral_data.keys():
                if axis_name_lower in key.lower() and 'gyro' in key.lower():
                    gyro_key = key
                    break

            if gyro_key and gyro_key in spectral_data and 'error' not in spectral_data[gyro_key]:
                peaks = spectral_data[gyro_key].get("dominant_peaks_hz_mag", [])
                band_mags = spectral_data[gyro_key].get("band_avg_magnitude", {})

                if peaks:
                    # --- Mid-Frequency Oscillation (Potential P/D Tuning Issue) ---
                    # Look for peaks between 20Hz and 80Hz
                    mid_freq_threshold_mag = 0.1
                    mid_peaks = [(f, m) for f, m in peaks if 20 <= f <= 80 and m > mid_freq_threshold_mag]

                    if mid_peaks:
                        freq, mag = max(mid_peaks, key=lambda item: item[1])
                        severity = np.clip((mag / 0.5) * 10, 1, 10)

                        # Retrieve current PIDs (handle potential missing keys)
                        current_P = current_pids.get(f'p_{axis_name_lower}', current_pids.get(f'pid{axis_name_upper}P'))
                        current_D = current_pids.get(f'd_{axis_name_lower}', current_pids.get(f'pid{axis_name_upper}D'))

                        recommendation_text = f"Reduce {axis_name_upper} P or D gain"
                        explanation = (
                            f"Detected mid-frequency oscillation at **{freq:.0f} Hz** (magnitude {mag:.3f}) on the {axis_name_upper} axis. "
                            f"This often indicates that the **P gain** (current: {current_P}) or **D gain** (current: {current_D}) might be too high, "
                            f"causing instability or amplifying noise near this frequency. Consider lowering P and/or D gain for this axis."
                        )
                        cli_commands = [f"# Recommendation: Reduce P/D for {axis_name_upper} due to {freq:.0f}Hz oscillation"]
                        sim_details = {}

                        # Attempt simulation-based optimization if P and D are found
                        if current_P is not None and current_D is not None:
                            try:
                                opt_P, opt_D, score = optimize_pid_for_axis(float(current_P), float(current_D))
                                if score != float('inf'):
                                    recommendation_text = f"Optimize {axis_name_upper} PID (Simulated)"
                                    explanation += (
                                        f" Simulation suggests adjusting to **P={opt_P:.1f}** and **D={opt_D:.1f}** "
                                        f"might improve response based on a simplified model (Score: {score:.3f})."
                                    )
                                    cli_commands = [
                                        f"set p_{axis_name_lower} = {opt_P:.1f}",
                                        f"set d_{axis_name_lower} = {opt_D:.1f}",
                                        "# save"
                                    ]
                                    sim_details = {
                                        "current": {"P": current_P, "D": current_D},
                                        "recommended": {"P": opt_P, "D": opt_D},
                                        "score": score
                                    }
                                else:
                                    explanation += " (Simulation optimization failed to find improvement)."
                            except ValueError as ve:
                                logger.warning(
                                    f"Could not run optimization for {axis_name_upper}, invalid PID value? P={current_P}, D={current_D}. Error: {ve}"
                                )
                                explanation += " (Could not run simulation optimization - check PID values)."
                        else:
                            explanation += " (Cannot run simulation optimization - current P/D values missing)."

                        problems.append({
                            "id": f"mid_freq_osc_{axis_name_lower}",
                            "severity": severity,
                            "title": f"Mid-Frequency Oscillation ({axis_name_upper})",
                            "recommendation": recommendation_text,
                            "explanation": explanation,
                            "commands": cli_commands,
                            "category": "Tuning (PID)",
                            "details": {"frequency_hz": freq, "magnitude": mag, **sim_details}
                        })

                # --- High-Frequency Noise (Potential Filter Issue or D-term Noise) ---
                high_freq_mag = band_mags.get("high_(80-200Hz)", 0) + band_mags.get("vhigh_(>200Hz)", 0)
                if high_freq_mag > 0.1:
                    severity = np.clip((high_freq_mag / 0.3) * 10, 1, 10)
                    explanation = (
                        f"High-frequency noise detected on {axis_name_upper} gyro (Avg Mag > 80Hz: {high_freq_mag:.3f}). "
                        f"This could be caused by motor/frame vibrations or excessive D-term gain amplifying noise. "
                        f"Consider checking hardware for vibrations (motors, props), ensuring filters are appropriately configured "
                        f"(e.g., Gyro LPF, D-term LPF, Notches), or potentially reducing D gain if filtering is already optimized."
                    )
                    problems.append({
                        "id": f"high_freq_noise_{axis_name_lower}",
                        "severity": severity,
                        "title": f"High-Frequency Noise ({axis_name_upper})",
                        "recommendation": "Investigate filters, vibrations, or D gain",
                        "explanation": explanation,
                        "commands": ["# Review filter settings & check for vibrations"],
                        "category": "Tuning (Filters/Noise)",
                        "details": {"avg_high_freq_magnitude": high_freq_mag}
                    })

                # --- Low-Frequency Oscillation (Prop Wash / Slow Wobble) ---
                low_freq_mag = band_mags.get("low_(<20Hz)", 0)
                if low_freq_mag > 0.15:
                    severity = np.clip((low_freq_mag / 0.4) * 10, 1, 10)
                    explanation = (
                        f"Significant low-frequency noise/oscillation detected on {axis_name_upper} gyro (Avg Mag < 20Hz: {low_freq_mag:.3f}). "
                        f"This can sometimes be related to prop wash handling (I-term behavior), slow wobbles, or very low frequency vibrations. "
                        f"Consider reviewing I gain, anti-gravity settings, or thrust linearization if applicable. Ensure propellers are balanced and undamaged."
                    )
                    problems.append({
                        "id": f"low_freq_osc_{axis_name_lower}",
                        "severity": severity,
                        "title": f"Low-Frequency Noise/Oscillation ({axis_name_upper})",
                        "recommendation": "Investigate I gain, prop wash handling, or vibrations",
                        "explanation": explanation,
                        "commands": ["# Review I gain, Anti-Gravity, Props"],
                        "category": "Tuning (I-Term/Vibrations)",
                        "details": {"avg_low_freq_magnitude": low_freq_mag}
                    })

        # --- Add checks based on other analysis results ---
        # Example: High Motor Saturation
        motors_data = analysis_results.get('motors', {}).get('motors', {})
        if motors_data:
            saturation = motors_data.get('motor_saturation_pct_overall_any')
            if saturation is not None and saturation > 15:
                severity = np.clip((saturation / 30) * 10, 2, 10)
                explanation = (
                    f"High overall motor saturation detected ({saturation:.1f}%). This means motors frequently reached their maximum or minimum output limit. "
                    f"This limits control authority and can lead to poor performance, especially during aggressive maneuvers or high wind. "
                    f"Possible causes include excessive PID gains (especially D), insufficient motor power/thrust for the craft's weight, or aggressive flying style."
                )
                problems.append({
                    "id": "high_motor_saturation",
                    "severity": severity,
                    "title": "High Motor Saturation",
                    "recommendation": "Reduce PID gains or check motor/prop setup",
                    "explanation": explanation,
                    "commands": ["# Consider reducing PID gains (especially D)", "# Check motor/prop suitability"],
                    "category": "Tuning (PID/Hardware)",
                    "details": {"saturation_percent": saturation}
                })

        # Remove duplicate problem IDs if generated
        unique_problems = []
        seen_ids = set()
        for problem in problems:
            if problem['id'] not in seen_ids:
                unique_problems.append(problem)
                seen_ids.add(problem['id'])

        # Sort problems by severity (descending)
        sorted_problems = sorted(unique_problems, key=lambda x: x.get("severity", 0), reverse=True)
        logger.debug(f"Identified {len(sorted_problems)} problem patterns.")
        return sorted_problems


    def generate_tuning_recommendations(self, analysis_results: dict, metadata: dict) -> dict:
        logger.debug("Generating tuning recommendations...")
        problem_patterns = self.identify_problem_patterns(analysis_results, metadata)
        flight_assessment = self.generate_flight_assessment(analysis_results)
        recommendations = {
            "flight_assessment": flight_assessment,
            "problem_patterns": problem_patterns
        }
        logger.info(f"Generated {len(problem_patterns)} problem patterns/recommendations.")
        return recommendations


    def full_log_analysis(self, file_path: str) -> dict:
        logger.info(f"--- Starting Full Analysis for {os.path.basename(file_path)} ---")
        analysis_start_time = datetime.now()
        analysis_results = {}
        df = None
        df_raw = None
        metadata = {}
        recommendations = {}
        data_quality = {}
        log_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}"

        try:
            lines = self._read_log_file(file_path)
            metadata_lines, header_line, data_start_index = self._find_header_and_data(lines)
            metadata = self.parse_metadata(metadata_lines)
            metadata['filename'] = os.path.basename(file_path)
            metadata['log_id'] = log_id  # Store log_id in metadata as well

            df_raw = self.parse_data(header_line, lines[data_start_index:])
            df = self.prepare_data(df_raw.copy(), metadata)  # Pass copy to avoid modifying raw

            # --- Perform Analyses ---
            data_quality = self.diagnose_data_quality(df, metadata)
            # Only proceed with detailed analysis if data quality is acceptable
            if data_quality.get('quality_score', 0.0) >= 0.3:
                logger.info(f"Data quality score ({data_quality.get('quality_score', 0.0):.2f}) sufficient for detailed analysis.")
                analysis_results['pid'] = self.analyze_pid_performance(df)
                analysis_results['setpoint_vs_rc'] = self.analyze_setpoint_vs_rc(df)
                analysis_results['motors'] = self.analyze_motors(df)
                analysis_results['spectral'] = self.perform_spectral_analysis(df)
                analysis_results['gyro_accel'] = self.analyze_gyro_accel(df)  # Combines gyro and accel analysis
                analysis_results['rc_commands'] = self.analyze_rc_commands(df)
                analysis_results['alt_power'] = self.analyze_altitude_power(df)  # Combines alt and power
                analysis_results['rc_gyro_latency'] = self.analyze_rc_vs_gyro(df)
                analysis_results['trajectory'] = self.analyze_flight_trajectory(df)
                analysis_results['control_effort'] = self.analyze_control_effort(df)
                # Generate 3D plot data but don't store the figure object directly in results
                # analysis_results['3d_flight_plot'] = self.plot_3d_flight(df)  # Don't store figure

                recommendations = self.generate_tuning_recommendations(analysis_results, metadata)
            else:
                logger.warning(f"Skipping detailed analysis due to low data quality score: {data_quality.get('quality_score', 0.0):.2f}")
                recommendations = {
                    "flight_assessment": {"summary": "Analysis skipped due to poor data quality."},
                    "problem_patterns": []
                }

            logger.info("Saving analysis results...")
            save_success = self.save_log_analysis(log_id, metadata, analysis_results, recommendations, data_quality)
            if not save_success:
                logger.warning("Failed to save analysis results to database.")

            analysis_duration = (datetime.now() - analysis_start_time).total_seconds()
            logger.info(f"--- Finished Full Analysis for {metadata['filename']} in {analysis_duration:.2f} seconds ---")

            return {
                "log_id": log_id,
                "metadata": metadata,
                "analysis_results": analysis_results,
                "recommendations": recommendations,
                "data_quality": data_quality,
                "df": df  # Return the processed DataFrame for UI plotting
                # "df_raw": df_raw  # Optionally return raw DF for debugging
            }

        except FileNotFoundError as e:
            logger.error(f"File not found error during analysis: {e}")
            return {"error": str(e), "log_id": log_id, "metadata": metadata}
        except ValueError as e:
            logger.error(f"Value error during analysis (likely parsing/prep): {e}", exc_info=True)
            return {"error": f"Data Parsing/Preparation Error: {e}", "log_id": log_id, "metadata": metadata, "df_raw": df_raw}
        except Exception as e:
            logger.error(f"Unexpected error during full_log_analysis: {e}", exc_info=True)
            return {
                "error": f"An unexpected error occurred: {e}",
                "log_id": log_id,
                "metadata": metadata,
                "analysis_results": analysis_results,  # Might contain partial results
                "df": df if df is not None else None,
                # "df_raw": df_raw
            }


    def save_log_analysis(self, log_id: str, metadata: dict, analysis_results: dict, recommendations: dict, data_quality: dict) -> bool:
        logger.debug(f"Saving analysis summary for log ID: {log_id}")
        try:
            logs_db = {}
            if os.path.exists(self.logs_db_path):
                try:
                    with open(self.logs_db_path, 'r') as f:
                        file_content = f.read()
                        if file_content:  # Check if file is not empty
                            logs_db = json.loads(file_content)
                        if not isinstance(logs_db, dict):
                            logger.warning(f"Log database file {self.logs_db_path} is not a dictionary. Resetting.")
                            logs_db = {}
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.warning(f"Log database file not found or corrupted ({e}). Creating new one.")
                    logs_db = {}

            # Create summary object with serializable data
            summary = {
                "timestamp": metadata.get('analysis_timestamp', datetime.now().isoformat()),
                "log_filename": metadata.get('filename', 'Unknown'),
                "metadata_summary": {
                    'betaflight_version': metadata.get('betaflight_version'),
                    'firmware_target': metadata.get('firmware_target'),
                    'board': metadata.get('board'),
                    'craft_name': metadata.get('craft_name'),
                    'pid_profile': metadata.get('other_settings', {}).get('pid_profile'),
                },
                "analysis_results_summary": make_serializable(analysis_results),
                "recommendations_summary": make_serializable(recommendations),
                "data_quality_summary": make_serializable(data_quality)
            }

            logs_db[log_id] = summary

            with open(self.logs_db_path, 'w') as f:
                json.dump(logs_db, f, indent=2)
            logger.debug(f"Successfully saved analysis summary for log ID: {log_id}")

            # Save key metrics to tuning history
            self.save_to_tuning_history(log_id, metadata, analysis_results, recommendations, data_quality)

            return True

        except TypeError as te:
            logger.error(f"Serialization error saving log analysis: {te}. Check data types.", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving log analysis: {e}", exc_info=True)
            return False


    def calculate_noise_level(self, analysis_results: dict) -> float | None:
        """Helper to calculate a single noise metric from spectral results."""
        noise = None
        try:
            spectral = analysis_results.get('spectral', {}).get('spectra', {})
            if spectral:
                noise_levels = []
                for axis_data in spectral.values():
                    if 'error' not in axis_data:
                        bands = axis_data.get('band_avg_magnitude', {})
                        high_noise = bands.get("high_(80-200Hz)", 0) + bands.get("vhigh_(>200Hz)", 0)
                        noise_levels.append(high_noise)
                if noise_levels:
                    noise = float(np.mean(noise_levels))
        except Exception as e:
            logger.warning(f"Could not calculate aggregate noise level: {e}")
        return noise


    def save_to_tuning_history(self, log_id: str, metadata: dict, analysis_results: dict, recommendations: dict, data_quality: dict) -> bool:
        logger.debug(f"Saving key metrics to tuning history for log ID: {log_id}")
        try:
            history = []
            if os.path.exists(self.tuning_history_path):
                try:
                    with open(self.tuning_history_path, 'r') as f:
                        file_content = f.read()
                        if file_content:
                            history = json.loads(file_content)
                        if not isinstance(history, list):
                            logger.warning(f"Tuning history file {self.tuning_history_path} is not a list. Resetting.")
                            history = []
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.warning(f"Tuning history file not found or corrupted ({e}). Creating new one.")
                    history = []

            pid_tracking = analysis_results.get('pid', {}).get('tracking_error', {})
            motors_data = analysis_results.get('motors', {}).get('motors', {})
            flight_assess = recommendations.get('flight_assessment', {})

            key_metrics = {
                "timestamp": metadata.get('analysis_timestamp', datetime.now().isoformat()),
                "log_id": log_id,
                "log_filename": metadata.get('filename'),
                "betaflight_version": metadata.get('betaflight_version'),
                "firmware_target": metadata.get('firmware_target'),
                "pid_values": metadata.get("pid_values", {}),
                "performance": {
                    "quality_score": data_quality.get("quality_score"),
                    "flight_quality": flight_assess.get("flight_quality"),
                    "roll_mae": pid_tracking.get("roll_mae"),
                    "pitch_mae": pid_tracking.get("pitch_mae"),
                    "yaw_mae": pid_tracking.get("yaw_mae"),
                    "motor_saturation_pct": motors_data.get("motor_saturation_pct_overall_any"),
                    "motor_imbalance_pct": motors_data.get("motor_imbalance_pct_of_avg"),
                    "noise_level": self.calculate_noise_level(analysis_results),
                }
            }

            history.append(make_serializable(key_metrics))

            max_history_items = 100
            history = history[-max_history_items:]

            with open(self.tuning_history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.debug(f"Saved tuning history entry for log ID: {log_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving tuning history: {e}", exc_info=True)
            return False


    def get_log_summary(self, log_id: str) -> dict | None:
        """Retrieves the saved analysis summary for a given log ID."""
        try:
            if not os.path.exists(self.logs_db_path):
                logger.warning("Log database file does not exist.")
                return None
            with open(self.logs_db_path, 'r') as f:
                logs_db = json.load(f)
            return logs_db.get(log_id)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading log database: {e}")
            return None


    def compare_logs(self, log_id1: str, log_id2: str) -> dict:
        logger.debug(f"Comparing logs: {log_id1} vs {log_id2}")
        comparison = {"log1_id": log_id1, "log2_id": log_id2, "changes": {}, "assessment": {}}

        log1_summary = self.get_log_summary(log_id1)
        log2_summary = self.get_log_summary(log_id2)

        if not log1_summary or not log2_summary:
            comparison["error"] = "One or both log IDs not found in database or database read error."
            logger.error(comparison["error"])
            return comparison

        # --- Compare Metadata (e.g., PIDs) ---
        comparison["changes"]["metadata"] = {}
        meta1 = log1_summary.get("metadata_summary", {})
        meta2 = log2_summary.get("metadata_summary", {})
        for key in ['betaflight_version', 'firmware_target', 'pid_profile']:
            if meta1.get(key) != meta2.get(key):
                comparison["changes"]["metadata"][key] = {"log1": meta1.get(key), "log2": meta2.get(key)}

        # Compare PID values more comprehensively
        pids1 = log1_summary.get("metadata_summary", {}).get("pid_values", {})
        pids2 = log2_summary.get("metadata_summary", {}).get("pid_values", {})
        all_pid_keys = set(pids1.keys()) | set(pids2.keys())
        pid_changes = {}
        for key in all_pid_keys:
            v1 = pids1.get(key)
            v2 = pids2.get(key)
            if v1 != v2:
                try:
                    v1_f = float(v1) if v1 is not None else None
                    v2_f = float(v2) if v2 is not None else None
                    diff = v2_f - v1_f if v1_f is not None and v2_f is not None else None
                    pct_change = (diff / abs(v1_f)) * 100 if diff is not None and v1_f != 0 else None
                    pid_changes[key] = {
                        "log1": v1,
                        "log2": v2,
                        "diff": round(diff, 2) if diff is not None else None,
                        "pct_change": round(pct_change, 1) if pct_change is not None else None
                    }
                except (ValueError, TypeError):
                    pid_changes[key] = {"log1": v1, "log2": v2}
        if pid_changes:
            comparison["changes"]["pids"] = pid_changes

        # --- Compare Performance Metrics ---
        comparison["changes"]["performance"] = {}
        perf1 = log1_summary.get("analysis_results_summary", {})
        perf2 = log2_summary.get("analysis_results_summary", {})

        metrics_to_compare = {
            "Roll MAE": ('pid', 'tracking_error', 'roll_mae'),
            "Pitch MAE": ('pid', 'tracking_error', 'pitch_mae'),
            "Yaw MAE": ('pid', 'tracking_error', 'yaw_mae'),
            "Noise Level": ('calculated', 'noise_level'),
            "Motor Saturation %": ('motors', 'motors', 'motor_saturation_pct_overall_any'),
            "Motor Imbalance %": ('motors', 'motors', 'motor_imbalance_pct_of_avg'),
            "Voltage Sag %": ('alt_power', 'power', 'voltage_sag_percent'),
            "Avg Latency XCorr (ms)": ('rc_gyro_latency', 'rc_gyro_latency', 'roll_lag_ms_xcorr'),
        }

        def get_nested(data, keys, default=None):
            if not data or not keys:
                return default
            temp = data
            for key in keys:
                if isinstance(temp, dict) and key in temp:
                    temp = temp[key]
                else:
                    return default
            return temp

        improvements = 0
        regressions = 0
        for metric_name, keys in metrics_to_compare.items():
            if keys[0] == 'calculated':
                v1 = self.calculate_noise_level(perf1)
                v2 = self.calculate_noise_level(perf2)
            else:
                v1 = get_nested(perf1, keys[1:])
                v2 = get_nested(perf2, keys[1:])

            if v1 is not None and v2 is not None:
                try:
                    v1_f = float(v1)
                    v2_f = float(v2)
                    if not np.isclose(v1_f, v2_f):
                        diff = v2_f - v1_f
                        pct_change = (diff / abs(v1_f)) * 100 if abs(v1_f) > 1e-6 else None
                        is_improvement = diff < 0
                        if is_improvement:
                            improvements += 1
                        else:
                            regressions += 1
                        comparison["changes"]["performance"][metric_name] = {
                            "log1": round(v1_f, 3),
                            "log2": round(v2_f, 3),
                            "diff": round(diff, 3),
                            "pct_change": round(pct_change, 1) if pct_change is not None else None,
                            "status": "Improved" if is_improvement else "Regressed"
                        }
                except (ValueError, TypeError):
                    logger.debug(f"Could not compare non-numeric metric '{metric_name}': {v1} vs {v2}")
                    if v1 != v2:
                        comparison["changes"]["performance"][metric_name] = {"log1": v1, "log2": v2, "status": "Changed"}

        # --- Overall Assessment ---
        q1 = get_nested(log1_summary, ("recommendations_summary", "flight_assessment", "flight_quality"))
        q2 = get_nested(log2_summary, ("recommendations_summary", "flight_assessment", "flight_quality"))

        if q1 is not None and q2 is not None:
            comparison["assessment"]["quality_change"] = {
                "log1": round(q1, 2),
                "log2": round(q2, 2),
                "diff": round(q2 - q1, 2)
            }
            if q2 > q1 + 0.05:
                verdict = "Improved"
            elif q1 > q2 + 0.05:
                verdict = "Regressed"
            else:
                verdict = "Similar"
        else:
            verdict = "Undetermined"

        comparison["assessment"]["overall_verdict"] = verdict
        comparison["assessment"]["improvements_count"] = improvements
        comparison["assessment"]["regressions_count"] = regressions

        logger.debug(f"Comparison complete: {log_id1} vs {log_id2}. Verdict: {verdict}")
        return comparison


    def get_tuning_history(self) -> List[Dict[str, Any]]:
        logger.debug("Loading tuning history...")
        try:
            if os.path.exists(self.tuning_history_path):
                with open(self.tuning_history_path, 'r') as f:
                    file_content = f.read()
                    if file_content:
                        history = json.loads(file_content)
                        if isinstance(history, list):
                            return history
                        else:
                            logger.warning("Tuning history file is not a list. Returning empty.")
                            return []
                    else:
                        return []  # Empty file
            else:
                return []  # File doesn't exist
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading tuning history: {e}", exc_info=True)
            return []


    def get_available_log_ids(self) -> List[str]:
        """Returns a list of log IDs present in the database."""
        try:
            if not os.path.exists(self.logs_db_path):
                return []
            with open(self.logs_db_path, 'r') as f:
                logs_db = json.load(f)
                if isinstance(logs_db, dict):
                    return list(logs_db.keys())
                else:
                    return []
        except Exception as e:
            logger.error(f"Error reading log database keys: {e}")
            return []


    def plot_pid_tracking(self, df: pd.DataFrame) -> go.Figure:
        logger.debug("Generating PID tracking plot...")
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Roll Axis", "Pitch Axis", "Yaw Axis"),
            vertical_spacing=0.05
        )

        plot_created = False
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            gyro_col = find_column(df, [f"gyro{axis}", f"gyroADC[{i}]"])
            setpoint_col = find_column(df, [f"setpoint{axis}", f"setpoint[{i}]"])

            if gyro_col and gyro_col in df:
                fig.add_trace(
                    go.Scattergl(
                        x=df.index,
                        y=df[gyro_col],
                        mode='lines',
                        name=f'Gyro {axis}',
                        line=dict(width=1)
                    ),
                    row=i + 1, col=1
                )
                plot_created = True
            else:
                logger.warning(f"Gyro column for {axis} not found.")

            if setpoint_col and setpoint_col in df:
                fig.add_trace(
                    go.Scattergl(
                        x=df.index,
                        y=df[setpoint_col],
                        mode='lines',
                        name=f'Setpoint {axis}',
                        line=dict(dash='dash', width=1.5)
                    ),
                    row=i + 1, col=1
                )
                plot_created = True
            else:
                logger.warning(f"Setpoint column for {axis} not found.")

            fig.update_yaxes(title_text="Rate (°/s)", row=i + 1, col=1)

        if not plot_created:
            fig.add_annotation(
                text="No PID tracking data found.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )

        fig.update_layout(
            title="PID Tracking: Gyro vs Setpoint",
            height=600,
            legend_title_text='Trace',
            hovermode='x unified'
        )
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        return fig


    def plot_motor_output(self, df: pd.DataFrame) -> go.Figure:
        logger.debug("Generating motor output plot...")
        motor_cols_numeric = sorted([
            col for col in df.columns
            if col.lower().startswith('motor[') and col.endswith(']')
        ])
        motor_cols_named = sorted([
            col for col in df.columns
            if col.lower().startswith('motor') and col[5:].isdigit()
        ])
        motor_cols = motor_cols_numeric if motor_cols_numeric else motor_cols_named
        if not motor_cols:
            motor_cols = sorted([col for col in df.columns if 'motor' in col.lower()])

        fig = go.Figure()
        plot_created = False
        if not motor_cols:
            fig.add_annotation(
                text="No motor data found.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
        else:
            for col in motor_cols:
                if col in df and pd.api.types.is_numeric_dtype(df[col]):
                    fig.add_trace(
                        go.Scattergl(
                            x=df.index,
                            y=df[col],
                            mode='lines',
                            name=col,
                            line=dict(width=1)
                        )
                    )
                    plot_created = True
                else:
                    logger.warning(f"Motor column '{col}' not found or not numeric, skipping plot.")

        fig.update_layout(
            title="Motor Outputs"
        )
        return fig
# --- Streamlit UI ---

# Instantiate the analyzer
@st.cache_resource  # Cache the analyzer instance for efficiency
def get_analyzer():
    logger.info("Initializing BetaflightLogAnalyzer instance.")
    return BetaflightLogAnalyzer()


analyzer = get_analyzer()

# --- Session State Initialization ---
if 'current_analysis_results' not in st.session_state:
    st.session_state.current_analysis_results = None  # Stores results of the latest analysis

if 'available_logs' not in st.session_state:
    # Load available logs from the database at the start
    st.session_state.available_logs = analyzer.get_available_log_ids()

if 'compare_log1' not in st.session_state:
    st.session_state.compare_log1 = None

if 'compare_log2' not in st.session_state:
    st.session_state.compare_log2 = None

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None


# --- Helper Function to update available logs ---
def refresh_available_logs():
    st.session_state.available_logs = analyzer.get_available_log_ids()


# --- Sidebar ---
st.sidebar.title("Betaflight Log Analyzer")
st.sidebar.markdown("Upload, analyze, and compare your blackbox logs.")

# Mode Selection
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Analyze New Log", "Compare Logs", "View Tuning History"]
)

st.sidebar.divider()

# --- Analyze New Log Mode ---
if app_mode == "Analyze New Log":
    st.sidebar.header("Upload Log")
    # Allow CSV and BBL (though BBL might need external decoder first in a real scenario)
    uploaded_file = st.sidebar.file_uploader("Choose a log file (.csv, .bbl)", type=['csv', 'bbl', 'txt'])

    if uploaded_file is not None:
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")

        # Button to trigger analysis
        if st.sidebar.button("Analyze Log", key="analyze_button"):
            st.session_state.current_analysis_results = None  # Clear previous results
            st.session_state.comparison_results = None  # Clear comparison

            # Use a temporary file to pass the path to the analyzer
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                with st.spinner(f"Analyzing {uploaded_file.name}, please wait..."):
                    # Perform the full analysis
                    analysis_data = analyzer.full_log_analysis(tmp_file_path)

                if "error" in analysis_data:
                    st.error(f"Analysis failed: {analysis_data['error']}")
                    # Optionally display raw DF if available for debugging parse errors
                    if 'df_raw' in analysis_data and analysis_data['df_raw'] is not None:
                        st.warning("Raw data loaded before error:")
                        st.dataframe(analysis_data['df_raw'].head())
                    st.session_state.current_analysis_results = None
                else:
                    st.success(f"Analysis complete for {uploaded_file.name}!")
                    st.session_state.current_analysis_results = analysis_data
                    # Refresh log list after successful analysis and save
                    refresh_available_logs()

            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                logger.error(f"UI level analysis error: {e}", exc_info=True)
                st.session_state.current_analysis_results = None
            finally:
                # Clean up the temporary file
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                    logger.debug(f"Removed temporary file: {tmp_file_path}")

    st.sidebar.divider()
    # Allow selecting a previously analyzed log to view its results
    st.sidebar.header("View Previous Analysis")
    if st.session_state.available_logs:
        selected_log_id = st.sidebar.selectbox(
            "Select a saved log to view:",
            options=[""] + st.session_state.available_logs,  # Add empty option
            key="view_log_select"
        )
        if selected_log_id and st.sidebar.button("Load Analysis", key="load_analysis"):
            with st.spinner(f"Loading analysis for {selected_log_id}..."):
                # NOTE: This currently loads the SAVED SUMMARY.
                # Re-generating plots would require re-analysis or storing plot data/DFs.
                # For simplicity, we'll load the summary data.
                log_summary = analyzer.get_log_summary(selected_log_id)
                if log_summary:
                    # Create a structure similar to live analysis for display consistency
                    # We lack the 'df' here, so plots won't regenerate directly from this.
                    st.session_state.current_analysis_results = {
                        "log_id": selected_log_id,
                        "metadata": log_summary.get("metadata_summary", {}),  # Load summarized meta
                        "analysis_results": log_summary.get("analysis_results_summary", {}),
                        "recommendations": log_summary.get("recommendations_summary", {}),
                        "data_quality": log_summary.get("data_quality_summary", {}),
                        "df": None  # Mark DataFrame as unavailable for historical view
                    }
                    st.sidebar.success(f"Loaded summary for {selected_log_id}")
                else:
                    st.sidebar.error(f"Could not load summary for {selected_log_id}")

# --- Compare Logs Mode ---
elif app_mode == "Compare Logs":
    st.sidebar.header("Select Logs to Compare")
    if not st.session_state.available_logs:
        st.sidebar.warning("No saved logs available to compare. Analyze some logs first.")
    else:
        st.session_state.compare_log1 = st.sidebar.selectbox(
            "Select Log 1 (Baseline):",
            options=[""] + st.session_state.available_logs,
            key="compare_select1",
            index=0  # Default to empty
        )
        st.session_state.compare_log2 = st.sidebar.selectbox(
            "Select Log 2 (Comparison):",
            options=[""] + st.session_state.available_logs,
            key="compare_select2",
            index=0  # Default to empty
        )

        if st.session_state.compare_log1 and st.session_state.compare_log2 and st.session_state.compare_log1 != st.session_state.compare_log2:
            if st.sidebar.button("Compare Selected Logs", key="compare_button"):
                st.session_state.comparison_results = None  # Clear previous
                st.session_state.current_analysis_results = None  # Clear single view
                log_id1 = st.session_state.compare_log1
                log_id2 = st.session_state.compare_log2
                with st.spinner(f"Comparing {log_id1} and {log_id2}..."):
                    comparison_data = analyzer.compare_logs(log_id1, log_id2)
                if "error" in comparison_data:
                    st.error(f"Comparison failed: {comparison_data['error']}")
                else:
                    st.success("Comparison complete!")
                    st.session_state.comparison_results = comparison_data
        elif st.session_state.compare_log1 and st.session_state.compare_log2 and st.session_state.compare_log1 == st.session_state.compare_log2:
            st.sidebar.warning("Please select two different logs to compare.")

# --- View Tuning History Mode ---
elif app_mode == "View Tuning History":
    st.sidebar.info("This section shows key metrics over time from your analyzed logs.")

# --- Main Area Display Logic ---

st.header(f"Mode: {app_mode}")

# Display results of the currently loaded/analyzed log
if app_mode == "Analyze New Log" and st.session_state.current_analysis_results:
    results_data = st.session_state.current_analysis_results
    log_id = results_data.get("log_id", "N/A")
    metadata = results_data.get("metadata", {})
    analysis_results = results_data.get("analysis_results", {})
    recommendations = results_data.get("recommendations", {})
    data_quality = results_data.get("data_quality", {})
    df = results_data.get("df")  # The processed DataFrame
    filename = metadata.get('filename', log_id)

    st.subheader(f"Analysis Results for: `{filename}`")
    st.caption(f"Log ID: `{log_id}`")

    # --- Main Tabs for Analysis Results ---
    tab_summary, tab_plots, tab_reco, tab_quality, tab_meta, tab_raw = st.tabs([
        "📊 Summary", "📈 Plots", "💡 Recommendations", "️‍ Data Quality", "⚙️ Metadata", "📋 Raw Analysis"
    ])

    with tab_summary:
        st.subheader("Flight Summary")
        assess = recommendations.get('flight_assessment', {})
        qual = data_quality.get("quality_score", 0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Data Quality Score", f"{qual:.2f}/1.0",
                delta=f"{qual-1.0:.2f}" if qual != 1 else None,
                help=data_quality.get("summary", "Overall data quality assessment (1.0 is best).")
            )
        with col2:
            fq = assess.get("flight_quality", 0)
            st.metric(
                "Flight Quality Score", f"{fq:.2f}/1.0",
                delta=f"{fq-0.5:.2f}", delta_color="normal",
                help=assess.get("summary", "Overall flight performance assessment (1.0 is best).")
            )

        st.markdown("**Key Assessment Points:**")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("Strengths:")
            if assess.get("strengths"):
                for strength in assess.get("strengths", []):
                    st.success(f"- {strength}", icon="✔️")
            else:
                st.info("No specific strengths highlighted.")
        with cols[1]:
            st.markdown("Weaknesses / Areas for Improvement:")
            if assess.get("weaknesses"):
                for weakness in assess.get("weaknesses", []):
                    st.warning(f"- {weakness}", icon="⚠️")
            else:
                st.info("No specific weaknesses highlighted.")

        st.divider()
        st.subheader("Key Performance Indicators")
        # Display key KPIs (customize as needed)
        kpi_cols = st.columns(3)
        pid_res = analysis_results.get('pid', {}).get('tracking_error', {})
        motor_res = analysis_results.get('motors', {}).get('motors', {})
        spec_res = analysis_results.get('spectral', {})
        power_res = analysis_results.get('alt_power', {}).get('power', {})

        def format_kpi(value, decimals=2):
            return f"{value:.{decimals}f}" if value is not None and pd.notna(value) else "N/A"

        with kpi_cols[0]:
            st.markdown("**PID Tracking**")
            st.text(f"Roll MAE: {format_kpi(pid_res.get('roll_mae'))}")
            st.text(f"Pitch MAE: {format_kpi(pid_res.get('pitch_mae'))}")
            st.text(f"Yaw MAE: {format_kpi(pid_res.get('yaw_mae'))}")
        with kpi_cols[1]:
            st.markdown("**Motors**")
            st.text(f"Saturation %: {format_kpi(motor_res.get('motor_saturation_pct_overall_any'), 1)}")
            st.text(f"Imbalance %: {format_kpi(motor_res.get('motor_imbalance_pct_of_avg'), 1)}")
            st.text(f"Avg Output: {format_kpi(motor_res.get('avg_motor_output_overall'))}")
        with kpi_cols[2]:
            st.markdown("**Noise & Power**")
            noise_level = analyzer.calculate_noise_level(analysis_results)
            st.text(f"Noise Level: {format_kpi(noise_level, 3)}")
            st.text(f"Voltage Sag %: {format_kpi(power_res.get('voltage_sag_percent'), 1)}")
            st.text(f"Max Power (W): {format_kpi(power_res.get('power_max_watts'), 0)}")

    with tab_plots:
        if df is not None and not df.empty:
            st.subheader("Flight Data Plots")
            with st.expander("PID Tracking (Gyro vs Setpoint)", expanded=True):
                st.plotly_chart(analyzer.plot_pid_tracking(df), use_container_width=True)
            with st.expander("Motor Outputs"):
                st.plotly_chart(analyzer.plot_motor_output(df), use_container_width=True)
            with st.expander("Spectral Analysis (Gyro Noise FFT)"):
                spectral_plots = analyzer.plot_spectral_analysis(analysis_results)
                if spectral_plots:
                    for name, fig in spectral_plots.items():
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No spectral analysis plots could be generated.")
            with st.expander("Throttle vs Frequency Heatmap"):
                heatmap_fig = analyzer.plot_throttle_frequency_heatmap(analysis_results)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    st.info("Throttle vs Frequency heatmap could not be generated (likely missing data).")
            with st.expander("RC Commands"):
                st.plotly_chart(analyzer.plot_rc_commands(df), use_container_width=True)
            with st.expander("3D Flight Path (GPS Required)"):
                fig_3d = analyzer.plot_3d_flight(df)
                if fig_3d:
                    if fig_3d.data and any(trace.type == 'scatter3d' and trace.name == 'Trajectory' for trace in fig_3d.data):
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info("No GPS Cartesian data (gpsCartesianCoords[0/1/2]) found in the log for 3D plot.")
                else:
                    st.info("3D plot could not be generated.")
        elif df is None:
            st.warning("Plots cannot be generated because the DataFrame is not available (viewing historical summary).")
        else:
            st.warning("No valid data in DataFrame to generate plots.")

    with tab_reco:
        st.subheader("Tuning Recommendations & Problem Patterns")
        patterns = recommendations.get("problem_patterns", [])
        if patterns:
            st.info(f"Found {len(patterns)} potential area(s) for tuning attention, sorted by estimated severity.")
            for i, p in enumerate(patterns):
                severity = p.get('severity', 5)  # Default severity if missing
                severity_icon = "🟥" if severity > 7 else ("🟧" if severity > 4 else "🟨")
                with st.expander(
                    f"{severity_icon} **{p.get('title', 'Unknown Pattern')}** (Severity: {severity:.1f}/10)",
                    expanded=(i < 2)
                ):
                    st.markdown(f"**Category:** `{p.get('category', 'N/A')}`")
                    st.markdown("**Recommendation:**")
                    st.success(f"`{p.get('recommendation', 'N/A')}`")
                    st.markdown("**Explanation:**")
                    st.markdown(p.get('explanation', 'N/A'))
                    if p.get('commands'):
                        st.markdown("**Suggested CLI Commands:**")
                        st.code("\n".join(p.get('commands', [])), language="bash")
                    if p.get('details'):
                        st.markdown("**Details:**")
                        st.json(p.get('details'), expanded=False)
        else:
            st.success("No specific problem patterns automatically identified based on current thresholds. Review plots manually.")

    with tab_quality:
        st.subheader("Data Quality Diagnostics")
        st.metric("Overall Quality Score", f"{data_quality.get('quality_score', 0.0):.2f}/1.0")
        st.markdown(f"**Summary:** {data_quality.get('summary', 'N/A')}")
        if data_quality.get('issues_found'):
            st.markdown("**Issues Detected:**")
            for issue in data_quality.get('issues_found', []):
                st.warning(f"- {issue}")
        st.subheader("Detailed Checks")
        with st.expander("Missing Data"):
            st.json(data_quality.get("missing_data", {}))
        with st.expander("Sampling Issues"):
            st.json(data_quality.get("sampling_issues", {}))
        with st.expander("Data Range Issues"):
            if data_quality.get("data_range_issues"):
                for issue in data_quality.get("data_range_issues", []):
                    st.markdown(f"- {issue}")
            else:
                st.markdown("No significant range issues detected.")

    with tab_meta:
        st.subheader("Log Metadata")
        st.text(f"Filename: {metadata.get('filename')}")
        st.text(f"Log ID: {metadata.get('log_id')}")
        st.text(f"Analysis Timestamp: {metadata.get('analysis_timestamp')}")
        st.text(f"Betaflight Version: {metadata.get('betaflight_version', 'N/A')}")
        st.text(f"Firmware Target: {metadata.get('firmware_target', 'N/A')}")
        st.text(f"Board: {metadata.get('board', 'N/A')}")
        st.text(f"Craft Name: {metadata.get('craft_name', 'N/A')}")
        st.subheader("PID Values (from Metadata/Log)")
        st.json(metadata.get('pid_values', {}), expanded=False)
        st.subheader("Filter Settings (from Metadata/Log)")
        st.json(metadata.get('filters', {}), expanded=False)
        st.subheader("Rate Settings (from Metadata/Log)")
        st.json(metadata.get('rates', {}), expanded=False)
        st.json(metadata.get('rc_rates', {}), expanded=False)
        st.subheader("Other Settings (from Metadata/Log)")
        st.json(metadata.get('other_settings', {}), expanded=False)

    with tab_raw:
        st.subheader("Raw Analysis Output")
        st.info("This section shows the structured data generated by the analysis modules.")
        st.json(analysis_results, expanded=False)

# Display comparison results
elif app_mode == "Compare Logs" and st.session_state.comparison_results:
    results = st.session_state.comparison_results
    log1 = results.get("log1_id")
    log2 = results.get("log2_id")
    changes = results.get("changes", {})
    assessment = results.get("assessment", {})

    st.subheader(f"Comparison: `{log1}` vs `{log2}`")
    st.metric(
        "Overall Verdict",
        assessment.get("overall_verdict", "N/A"),
        delta=f"{assessment.get('quality_change', {}).get('diff'):.2f}"
        if assessment.get('quality_change', {}).get('diff') is not None else None,
        delta_color="normal",
        help="Change in Flight Quality Score (Log2 - Log1). Positive means improvement."
    )
    st.text(f"Improvements: {assessment.get('improvements_count', 0)} metric(s)")
    st.text(f"Regressions: {assessment.get('regressions_count', 0)} metric(s)")

    with st.expander("Metadata & PID Changes", expanded=True):
        meta_changes = changes.get("metadata", {})
        if meta_changes:
            st.markdown("**Metadata Differences:**")
            st.dataframe(pd.DataFrame.from_dict(meta_changes, orient='index'))
        else:
            st.markdown("_No metadata differences detected._")
        pid_changes = changes.get("pids", {})
        if pid_changes:
            st.markdown("**PID Differences:**")
            pid_df = pd.DataFrame.from_dict(pid_changes, orient='index')
            pid_df['pct_change'] = pid_df['pct_change'].map('{:.1f}%'.format).replace('nan%', '')
            pid_df['diff'] = pid_df['diff'].map('{:.2f}'.format).replace('nan', '')
            st.dataframe(pid_df[['log1', 'log2', 'diff', 'pct_change']])
        else:
            st.markdown("_No PID differences detected._")

    with st.expander("Performance Metric Changes", expanded=True):
        perf_changes = changes.get("performance", {})
        if perf_changes:
            perf_df = pd.DataFrame.from_dict(perf_changes, orient='index')
            if 'pct_change' in perf_df.columns:
                perf_df['pct_change'] = perf_df['pct_change'].map('{:.1f}%'.format).replace('nan%', '')
            if 'diff' in perf_df.columns:
                perf_df['diff'] = perf_df['diff'].map('{:.3f}'.format).replace('nan', '')
            st.dataframe(perf_df)
        else:
            st.markdown("_No significant performance metric differences detected._")

# Display Tuning History
elif app_mode == "View Tuning History":
    st.subheader("Tuning History")
    history = analyzer.get_tuning_history()
    if not history:
        st.info("No tuning history found. Analyze some logs to build history.")
    else:
        history_df = pd.DataFrame(history)
        # --- Prepare data for display ---
        perf_data = history_df['performance'].apply(pd.Series)
        pid_data = history_df['pid_values'].apply(lambda x: {
            'P_Roll': x.get('p_roll'),
            'D_Roll': x.get('d_roll'),
            'P_Pitch': x.get('p_pitch'),
            'D_Pitch': x.get('d_pitch'),
        }).apply(pd.Series)
        display_df = pd.concat([
            history_df[['timestamp', 'log_filename', 'betaflight_version', 'firmware_target']],
            pid_data[['P_Roll', 'D_Roll', 'P_Pitch', 'D_Pitch']],
            perf_data[['quality_score', 'flight_quality', 'roll_mae', 'pitch_mae', 'motor_saturation_pct', 'noise_level']]
        ], axis=1)
        try:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
            display_df = display_df.sort_values(by='timestamp', ascending=False)
        except Exception as e:
            logger.warning(f"Could not parse or sort history timestamps: {e}")
        st.dataframe(display_df)
        st.subheader("Performance Trends")
        metrics_to_plot = ['flight_quality', 'noise_level', 'motor_saturation_pct']
        plot_df = display_df[['timestamp'] + metrics_to_plot].copy()
        for col in metrics_to_plot:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df = plot_df.dropna(subset=metrics_to_plot, how='any')
        plot_df = plot_df.sort_values(by='timestamp')
        if not plot_df.empty and len(plot_df) > 1:
            trend_fig = make_subplots(
                rows=len(metrics_to_plot), cols=1, shared_xaxes=True,
                subplot_titles=metrics_to_plot
            )
            for i, metric in enumerate(metrics_to_plot):
                trend_fig.add_trace(
                    go.Scattergl(
                        x=plot_df['timestamp'], y=plot_df[metric],
                        mode='lines+markers', name=metric
                    ),
                    row=i + 1, col=1
                )
                trend_fig.update_yaxes(title_text=metric, row=i + 1, col=1)
            trend_fig.update_layout(
                height=250 * len(metrics_to_plot),
                title="Key Metric Trends Over Time",
                showlegend=False
            )
            trend_fig.update_xaxes(title_text="Date", row=len(metrics_to_plot), col=1)
            st.plotly_chart(trend_fig, use_container_width=True)
        elif len(plot_df) <= 1:
            st.info("Not enough history data points (need >1) with valid metrics to plot trends.")
        else:
            st.info("No history data with valid metrics to plot trends.")

# Fallback for when no mode matches or no data is loaded
elif not st.session_state.current_analysis_results and not st.session_state.comparison_results:
    st.info("Upload a log file using the sidebar to begin analysis, or select a mode.")
    # Display the default quadcopter SVG
    if default_svg_data_uri:
        st.image(default_svg_data_uri, width=200)
