# Standard library imports
import os
import io
import json
import re
import tempfile
import pathlib
import traceback
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union

# Third-party library imports
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- Configuration ---

# Set page config as the very first Streamlit command (if running as a Streamlit app)
# Ensure this runs only once at the top of your main script file if modularizing.
# st.set_page_config(page_title="Advanced Betaflight Log Analyzer", layout="wide")

# Configure logging
# Use INFO for general use, DEBUG for detailed development/troubleshooting
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- PID Simulation & Optimization Helpers ---

def simulate_pid_response(
    P: float,
    D: float,
    wn: float = 5.0,
    zeta: float = 0.7,
    time_end: float = 5.0,
    num_points: int = 500
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Simulate the closed-loop step response of a system controlled by a PD controller.

    Assumes a standard second-order plant: G(s) = 1 / (s^2 + 2*zeta*wn*s + wn^2)
    and a PD controller: C(s) = P + D*s.

    Args:
        P (float): Proportional gain.
        D (float): Derivative gain.
        wn (float): Natural frequency of the plant (rad/s). Default is 5.0.
        zeta (float): Damping ratio of the plant. Default is 0.7.
        time_end (float): Simulation duration in seconds. Default is 5.0.
        num_points (int): Number of points for the time vector. Default is 500.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
            - time (np.ndarray): Time vector for the response, or None on error.
            - response (np.ndarray): System response to a unit step input, or None on error.
            - overshoot (float): Percentage overshoot ((max_value - final_value) / final_value), or None on error.
            - rise_time (float): Time taken to rise from 10% to 90% of the final value, or None on error.
    """
    logger.debug(f"Simulating PID response with P={P}, D={D}, wn={wn}, zeta={zeta}")
    try:
        # --- Define Transfer Functions ---
        # Plant: G(s) = 1 / (s^2 + 2*zeta*wn*s + wn^2)
        num_plant = [1.0]
        den_plant = [1.0, 2 * zeta * wn, wn**2]
        plant_tf = signal.TransferFunction(num_plant, den_plant)

        # Controller (PD): C(s) = D*s + P
        num_controller = [D, P]
        den_controller = [1.0] # Denominator for C(s) is 1
        controller_tf = signal.TransferFunction(num_controller, den_controller)

        # --- Calculate Open-Loop and Closed-Loop Systems ---
        # Open-loop transfer function: L(s) = C(s) * G(s)
        # Using signal.series for multiplication is safer for TF objects
        open_loop_tf = signal.series(controller_tf, plant_tf)

        # Closed-loop transfer function: T(s) = L(s) / (1 + L(s))
        # Using signal.feedback is the standard way
        # Feedback assumes negative feedback by default
        closed_loop_tf = signal.feedback(open_loop_tf, sys2=1.0) # Feedback with H(s)=1

        # --- Simulate Step Response ---
        t = np.linspace(0, time_end, num_points)
        t_out, y_out = signal.step(closed_loop_tf, T=t)

        if t_out is None or y_out is None or len(t_out) == 0:
             logger.warning("PID simulation signal.step returned empty result.")
             return None, None, None, None

        # --- Calculate Metrics ---
        final_value = y_out[-1] if len(y_out) > 0 else 1.0 # Assume final value is 1 if empty, though unlikely

        # Overshoot calculation (relative to final value)
        max_y = np.max(y_out)
        overshoot = 0.0
        if final_value > 1e-6: # Avoid division by zero or near-zero
            overshoot = ((max_y - final_value) / final_value) * 100.0 if max_y > final_value else 0.0
        overshoot = max(0.0, overshoot) # Ensure non-negative

        # Rise Time (10% to 90% of final value)
        rise_time = None
        try:
            # Find indices where response crosses 10% and 90% thresholds
            indices_10 = np.where(y_out >= 0.1 * final_value)[0]
            indices_90 = np.where(y_out >= 0.9 * final_value)[0]

            if len(indices_10) > 0 and len(indices_90) > 0:
                t10 = t_out[indices_10[0]]
                t90 = t_out[indices_90[0]]
                if t90 >= t10: # Ensure 90% time is after 10% time
                    rise_time = t90 - t10
        except IndexError:
            logger.debug("Could not find 10% or 90% crossing for rise time calculation.")
        except Exception as e_rise:
            logger.warning(f"Error calculating rise time: {e_rise}")

        logger.debug(f"Simulation successful: Overshoot={overshoot:.2f}%, Rise Time={rise_time:.3f}s")
        return t_out, y_out, overshoot, rise_time

    except Exception as e:
        logger.error(f"Error during PID simulation: {e}", exc_info=True)
        return None, None, None, None


def optimize_pid_for_axis(
    current_P: float,
    current_D: float,
    desired_overshoot: float = 5.0, # Percentage
    desired_rise_time: float = 0.1, # Seconds
    p_range_factor: float = 0.5, # Search +/- 50% of current P
    d_range_factor: float = 0.5, # Search +/- 50% of current D
    num_steps: int = 10
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Sweeps a range around the current P and D values, simulates step response,
    and finds the P/D combination that minimizes a cost function based on
    desired overshoot and rise time.

    Args:
        current_P (float): Current Proportional gain.
        current_D (float): Current Derivative gain.
        desired_overshoot (float): Target overshoot percentage (e.g., 5.0 for 5%). Default 5.0.
        desired_rise_time (float): Target rise time in seconds (e.g., 0.1). Default 0.1.
        p_range_factor (float): Factor to determine P search range (+/- this factor * current_P). Default 0.5.
        d_range_factor (float): Factor to determine D search range (+/- this factor * current_D). Default 0.5.
        num_steps (int): Number of steps for P and D in the search grid. Default 10.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]:
            - best_P (float): Recommended P gain, or None if optimization failed.
            - best_D (float): Recommended D gain, or None if optimization failed.
            - best_score (float): Lowest cost score achieved, or None if optimization failed.
    """
    logger.info(f"Optimizing PID around P={current_P}, D={current_D} for Overshoot={desired_overshoot}%, Rise Time={desired_rise_time}s")

    best_score = float('inf')
    best_P = None
    best_D = None
    results_log = [] # Optional: Store simulation results

    # Define search ranges, ensuring non-negative values
    p_min = max(0.1, current_P * (1 - p_range_factor))
    p_max = current_P * (1 + p_range_factor)
    d_min = max(0.1, current_D * (1 - d_range_factor))
    d_max = current_D * (1 + d_range_factor)

    # Create search grid
    p_values = np.linspace(p_min, p_max, num_steps)
    d_values = np.linspace(d_min, d_max, num_steps)

    optimization_performed = False
    for P_test in p_values:
        for D_test in d_values:
            optimization_performed = True
            t, y, overshoot, rise_time = simulate_pid_response(P_test, D_test)

            # Skip if simulation failed or metrics are invalid
            if t is None or overshoot is None or rise_time is None:
                logger.debug(f"Skipping P={P_test:.2f}, D={D_test:.2f} due to simulation failure or invalid metrics.")
                continue

            # --- Cost Function ---
            # Penalize deviation from desired overshoot and rise time.
            # Normalize errors to make them comparable (e.g., relative error)
            overshoot_error = 0.0
            if desired_overshoot > 1e-6:
                 overshoot_error = abs(overshoot - desired_overshoot) / desired_overshoot
            else: # Handle desired overshoot of 0
                 overshoot_error = abs(overshoot) / 5.0 # Penalize relative to a small overshoot (e.g., 5%)

            rise_time_error = 0.0
            if desired_rise_time > 1e-6:
                rise_time_error = abs(rise_time - desired_rise_time) / desired_rise_time
            else: # Handle desired rise time of 0 (theoretically impossible)
                rise_time_error = abs(rise_time) / 0.1 # Penalize relative to a fast rise time (e.g., 0.1s)

            # Combine errors (simple sum, could use weights)
            # Add a small penalty for instability (very high overshoot)
            instability_penalty = (overshoot / 100.0)**2 if overshoot > 100 else 0
            score = overshoot_error + rise_time_error + instability_penalty * 5.0 # Weight penalty

            results_log.append({'P': P_test, 'D': D_test, 'overshoot': overshoot, 'rise_time': rise_time, 'score': score})

            # Update best score if current score is better
            if score < best_score:
                best_score = score
                best_P = P_test
                best_D = D_test
                logger.debug(f"New best PID found: P={best_P:.2f}, D={best_D:.2f}, Score={best_score:.3f} (Overshoot={overshoot:.1f}%, RiseTime={rise_time:.3f}s)")

    if not optimization_performed:
         logger.warning("PID optimization loop did not run (check ranges/steps).")
         return None, None, None

    if best_P is None or best_D is None:
        logger.warning("PID optimization failed to find a valid solution.")
        return None, None, None

    logger.info(f"Optimization complete. Best P={best_P:.2f}, Best D={best_D:.2f}, Score={best_score:.3f}")
    return best_P, best_D, best_score


# --- Helper Function for JSON Serialization ---

def make_serializable(obj: Any, max_depth: int = 10, _current_depth: int = 0) -> Any:
    """
    Recursively convert non-serializable types into serializable types for JSON.

    Handles common types like numpy numerics, arrays, datetime, pandas Timestamps,
    Path objects, and prevents infinite recursion in dicts/lists. Limits depth
    and size of serialized structures to prevent excessive memory/time usage.

    Args:
        obj (Any): The object to convert.
        max_depth (int): Maximum recursion depth for nested structures. Default 10.
        _current_depth (int): Internal counter for recursion depth.

    Returns:
        Any: A JSON-serializable representation of the object.
    """
    # Handle recursion depth
    if _current_depth > max_depth:
        logger.warning(f"Serialization truncated at depth {max_depth} for type {type(obj)}")
        return f"Max depth ({max_depth}) reached"

    # Basic types (already serializable)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Numpy numeric types
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        if np.isnan(obj): return None  # Represent NaN as null
        if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity" # Represent inf
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # --- Updated complex check ---
    elif isinstance(obj, (np.complex64, np.complex128)): # Use specific complex types
        return str(obj) # Represent complex as string
    # --- End of updated check ---

    # Numpy arrays
    elif isinstance(obj, np.ndarray):
        # Limit array size to prevent large JSON objects
        max_array_elements = 1000
        if obj.size > max_array_elements:
            return f"Large NumPy array {obj.shape} (>{max_array_elements} elements) - Not fully serialized"
        # Recursively serialize elements
        return [make_serializable(item, max_depth, _current_depth + 1) for item in obj.tolist()]

    # Python lists/tuples
    elif isinstance(obj, (list, tuple)):
        # Limit list/tuple size
        max_list_items = 500
        if len(obj) > max_list_items:
            return f"Large list/tuple ({len(obj)} items) - Not fully serialized"
        # Recursively serialize items
        return [make_serializable(item, max_depth, _current_depth + 1) for item in obj]

    # Dictionaries
    elif isinstance(obj, dict):
        # Limit dictionary size
        max_dict_items = 500
        if len(obj) > max_dict_items:
            return f"Large dictionary ({len(obj)} items) - Not fully serialized"
        try:
            # Recursively serialize keys and values
            return {
                str(k): make_serializable(v, max_depth, _current_depth + 1)
                for k, v in obj.items()
            }
        except RecursionError:
            logger.error("Recursion error during dict serialization.")
            return "Dict serialization failed (RecursionError)"

    # Datetime objects
    elif isinstance(obj, (datetime, pd.Timestamp)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj) # Fallback to string representation

    # Pathlib objects
    elif isinstance(obj, pathlib.Path):
        return str(obj)

    # Pandas NA or NoneType
    elif pd.isna(obj):
        return None

    # Plotly figures (avoid serializing the whole figure)
    elif isinstance(obj, go.Figure):
        return "Plotly Figure Object (Not Serialized)"

    # Objects with to_dict (like some Pandas objects, but be careful with DataFrames)
    elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            # Avoid serializing large DataFrames/Series via to_dict
            if isinstance(obj, (pd.DataFrame, pd.Series)) and obj.size > 1000:
                return f"Large Pandas {type(obj).__name__} {getattr(obj, 'shape', '')} (Not Serialized)"
            temp_dict = obj.to_dict()
            # Recursively serialize the resulting dictionary
            return make_serializable(temp_dict, max_depth, _current_depth + 1)
        except Exception as e:
            logger.warning(f"Error serializing {type(obj).__name__} using to_dict: {e}")
            return f"Error serializing {type(obj).__name__} via to_dict"

    # Fallback for other types: try string conversion
    else:
        try:
            # Avoid attempting to serialize bytes objects directly
            if isinstance(obj, bytes):
                return f"Bytes object of length {len(obj)} (Not Serialized)"
            # Attempt string conversion for unknown types
            return str(obj)
        except Exception as e:
            logger.debug(f"Failed to serialize object of type {type(obj)} using str(): {e}")
            return f"Non-serializable type: {type(obj).__name__}"

# --- Main Class for Betaflight Log Analysis ---

class BetaflightLogAnalyzer:
    """
    Analyzes Betaflight blackbox log files (.bbl, .bfl, .csv).

    Provides methods for parsing metadata and data, performing various analyses
    (PID performance, motor output, spectral analysis, etc.), generating plots,
    identifying potential issues, suggesting recommendations, and comparing logs.
    """

    def __init__(self, base_dir: str = ".", db_filename: str = "log_analysis_history.json"):
        """
        Initializes the analyzer.

        Args:
            base_dir (str): The base directory for storing database files. Defaults to current directory.
            db_filename (str): The filename for the JSON database storing analysis history.
                               Defaults to "log_analysis_history.json".
        """
        self.base_dir = pathlib.Path(base_dir).resolve()
        # Combined database for history and potentially other log data
        self.history_db_path = self.base_dir / db_filename
        self.base_dir.mkdir(parents=True, exist_ok=True) # Ensure base directory exists
        self._ensure_db_files_exist()

        # Dictionary mapping common log field names to descriptions.
        # Useful for understanding the data and potentially for UI tooltips.
        self.column_descriptions = {
            "loopIteration": "Counter for each flight controller loop iteration",
            "time": "Timestamp in microseconds (us) since flight controller boot or arming",
            # PID Terms (Array notation common in BF 4.x+)
            "axisP[0]": "Proportional PID term output for Roll axis",
            "axisP[1]": "Proportional PID term output for Pitch axis",
            "axisP[2]": "Proportional PID term output for Yaw axis",
            "axisI[0]": "Integral PID term output for Roll axis",
            "axisI[1]": "Integral PID term output for Pitch axis",
            "axisI[2]": "Integral PID term output for Yaw axis",
            "axisD[0]": "Derivative PID term output for Roll axis",
            "axisD[1]": "Derivative PID term output for Pitch axis",
            "axisD[2]": "Derivative PID term output for Yaw axis",
            "axisF[0]": "Feedforward PID term output for Roll axis",
            "axisF[1]": "Feedforward PID term output for Pitch axis",
            "axisF[2]": "Feedforward PID term output for Yaw axis",
            # RC Commands (Array notation common in BF 4.x+)
            "rcCommand[0]": "RC input command for Roll (usually 1000-2000 or normalized)",
            "rcCommand[1]": "RC input command for Pitch",
            "rcCommand[2]": "RC input command for Yaw",
            "rcCommand[3]": "RC input command for Throttle",
            # Setpoint (Target Rates) (Array notation common in BF 4.x+)
            "setpoint[0]": "Target rotation rate for Roll axis (deg/s)",
            "setpoint[1]": "Target rotation rate for Pitch axis (deg/s)",
            "setpoint[2]": "Target rotation rate for Yaw axis (deg/s)",
            "setpoint[3]": "Throttle setpoint (internal target throttle value)",
            # Gyroscope Data (Array notation common in BF 4.x+)
            "gyroADC[0]": "Gyroscope reading for Roll axis (deg/s)",
            "gyroADC[1]": "Gyroscope reading for Pitch axis (deg/s)",
            "gyroADC[2]": "Gyroscope reading for Yaw axis (deg/s)",
            # Accelerometer Data (Array notation common in BF 4.x+)
            "accSmooth[0]": "Filtered accelerometer reading for Roll axis (G)",
            "accSmooth[1]": "Filtered accelerometer reading for Pitch axis (G)",
            "accSmooth[2]": "Filtered accelerometer reading for Z axis (G)",
            # Motor Outputs (Array notation common)
            "motor[0]": "Motor 0 output command (e.g., DShot value 0-2000)",
            "motor[1]": "Motor 1 output command",
            "motor[2]": "Motor 2 output command",
            "motor[3]": "Motor 3 output command",
            # Common Legacy/Alternative Names
            "gyroRoll": "Gyroscope reading for Roll axis (deg/s) (alternative name)",
            "gyroPitch": "Gyroscope reading for Pitch axis (deg/s) (alternative name)",
            "gyroYaw": "Gyroscope reading for Yaw axis (deg/s) (alternative name)",
            "setpointRoll": "Target rotation rate for Roll axis (deg/s) (alternative name)",
            "setpointPitch": "Target rotation rate for Pitch axis (deg/s) (alternative name)",
            "setpointYaw": "Target rotation rate for Yaw axis (deg/s) (alternative name)",
            # Power / Altitude
            "vbatLatest": "Latest battery voltage reading (cV or V, check scale)",
            "amperageLatest": "Latest current draw reading (cA or A, check scale)",
            "baroAlt": "Altitude from barometer (cm or m, check scale)",
            # GPS (Examples - actual fields vary)
            "gpsFix": "GPS Fix Type (0=No Fix, 2=2D, 3=3D)",
            "gpsNumSat": "Number of GPS satellites visible",
            "gpsLat": "GPS Latitude (degrees * 1e7)",
            "gpsLon": "GPS Longitude (degrees * 1e7)",
            "gpsAltitude": "GPS Altitude (m)",
            "gpsSpeed": "GPS Ground Speed (cm/s or m/s)",
            "gpsHeading": "GPS Course over Ground (degrees)",
            "gpsHomeLat": "GPS Home Latitude",
            "gpsHomeLon": "GPS Home Longitude",
            "gpsDistance": "Distance to Home (m)",
            "gpsHomeAzimuth": "Direction to Home (degrees)",
            "gpsCartesianCoords[0]": "GPS Cartesian X coordinate relative to home (m)",
            "gpsCartesianCoords[1]": "GPS Cartesian Y coordinate relative to home (m)",
            "gpsCartesianCoords[2]": "GPS Cartesian Z coordinate relative to home (m)",
            # System State
            "flightModeFlags": "Bitmask indicating active flight modes (e.g., Acro, Angle, Horizon)",
            "stateFlags": "Bitmask for system state (e.g., Armed, Failsafe)",
            "failsafePhase": "Current failsafe phase (if triggered)",
            # Debug Fields (Vary widely based on debug_mode setting)
            "debug[0]": "Debug value 0 (meaning depends on debug_mode)",
            "debug[1]": "Debug value 1",
            "debug[2]": "Debug value 2",
            "debug[3]": "Debug value 3",
        }
        logger.info(f"Analyzer initialized. History DB: {self.history_db_path}")

    def _ensure_db_files_exist(self):
        """Creates an empty JSON history file if it doesn't exist."""
        if not self.history_db_path.exists():
            try:
                with self.history_db_path.open('w', encoding='utf-8') as f:
                    # Initialize history as a list within a root dictionary
                    json.dump({"log_history": []}, f, indent=2)
                logger.info(f"Initialized empty history database file: {self.history_db_path}")
            except Exception as e:
                logger.error(f"Error initializing history database file {self.history_db_path}: {e}", exc_info=True)
                # Consider raising an error here if the DB is critical
                # raise IOError(f"Could not initialize history DB: {e}") from e

    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """
        Retrieves the tuning history from the JSON database file.

        Returns:
            List[Dict[str, Any]]: A list of historical log analysis entries.
                                  Returns an empty list if the file doesn't exist,
                                  is empty, or contains invalid data.
        """
        if not self.history_db_path.exists():
            logger.warning(f"Tuning history file not found at {self.history_db_path}. Returning empty list.")
            return []

        try:
            with self.history_db_path.open('r', encoding='utf-8') as f:
                # Handle empty file case
                content = f.read()
                if not content:
                    logger.warning(f"Tuning history file {self.history_db_path} is empty. Returning empty list.")
                    return []
                data = json.loads(content)

            # Expecting a dictionary with a 'log_history' key containing a list
            history = data.get("log_history", [])

            if not isinstance(history, list):
                logger.error(f"Invalid tuning history format in {self.history_db_path}. Expected a list under 'log_history' key. Found type {type(history)}. Returning empty list.")
                # Optionally, try to recover if data itself is a list (older format?)
                if isinstance(data, list):
                     logger.warning("Found list at root level, attempting to use it as history (legacy format?).")
                     return data
                return []

            logger.debug(f"Successfully loaded {len(history)} entries from tuning history.")
            return history

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON in tuning history file {self.history_db_path}. Returning empty list.", exc_info=True)
            # Consider backing up the corrupted file here
            return []
        except Exception as e:
            logger.error(f"Unexpected error reading tuning history from {self.history_db_path}: {e}. Returning empty list.", exc_info=True)
            return []

    def _read_log_file(self, file_path: Union[str, pathlib.Path]) -> List[str]:
        """Reads log file content line by line, handling basic errors."""
        path = pathlib.Path(file_path)
        logger.debug(f"Attempting to read file: {path}")

        if not path.is_file():
            msg = f"File not found or is not a regular file: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            if path.stat().st_size == 0:
                msg = f"File is empty: {path}"
                logger.error(msg)
                raise ValueError(msg)

            # Read with error replacement for robustness against encoding issues
            with path.open('r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

        except OSError as e:
            msg = f"OS error reading file {path}: {e}"
            logger.error(msg, exc_info=True)
            raise IOError(msg) from e
        except Exception as e:
            msg = f"Unexpected error reading file {path}: {e}"
            logger.error(msg, exc_info=True)
            raise IOError(msg) from e

        # Basic sanity check for minimum content (e.g., header + some data)
        min_expected_lines = 5
        if len(lines) < min_expected_lines:
            msg = f"File has very few lines ({len(lines)} < {min_expected_lines}): {path}. May be incomplete or not a valid log."
            logger.warning(msg)
            # Decide whether to raise an error or just warn
            # raise ValueError(msg)

        logger.debug(f"Successfully read {len(lines)} lines from {path}")
        return lines

    def _find_header_and_data(self, lines: List[str]) -> Tuple[List[str], str, int]:
        """
        Locates the header row and the start of the data section using multiple strategies.

        Strategies:
        1. Look for standard Betaflight header prefixes like "H Product:", "H Firmware:", etc.
        2. Look for a line starting with "loopIteration" (case-insensitive).
        3. Look for a line containing several common column names (heuristic).

        Args:
            lines (List[str]): The lines read from the log file.

        Returns:
            Tuple[List[str], str, int]:
                - metadata_lines (List[str]): Lines identified as metadata/header comments.
                - header_line (str): The identified data header line (comma-separated columns).
                - data_start_index (int): The index of the first line of actual data.

        Raises:
            ValueError: If a reliable header or data section cannot be identified.
        """
        metadata_lines: List[str] = []
        header_line: Optional[str] = None
        data_start_index: int = -1

        # Common prefixes for metadata lines in Betaflight logs
        metadata_prefixes = ('H ', '#', 'Product:', 'Firmware:', 'Board:', 'Craft name:')
        # Common columns expected in the data header to aid detection
        common_data_cols = ['gyroadc[0]', 'rccommand[0]', 'motor[0]', 'time', 'axisd[0]', 'axisp[0]']

        potential_header_index = -1

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            lower_line = stripped_line.lower()

            # --- Strategy 1 & 2: Metadata Prefixes & loopIteration ---
            if stripped_line.startswith(metadata_prefixes):
                metadata_lines.append(line)
                continue # Definitely metadata

            if not stripped_line: # Skip empty lines
                metadata_lines.append(line) # Treat empty lines before header as metadata
                continue

            # Check for 'loopIteration' as the first column (common header start)
            if lower_line.startswith("loopiteration"):
                potential_header_index = i
                logger.debug(f"Potential header found via 'loopIteration' at line {i}.")
                break # Assume this is the header

            # --- Strategy 3: Heuristic based on common columns ---
            # Check if line contains enough common data columns and looks like a header
            # (not starting with numbers, contains commas)
            if ',' in stripped_line and \
               not stripped_line[0].isdigit() and \
               not stripped_line.startswith('-') and \
               sum(col in lower_line for col in common_data_cols) >= 3:

                # Check if the *next* non-empty line looks like data (starts with number, has commas)
                for j in range(i + 1, min(i + 5, len(lines))): # Check next few lines
                    next_line_stripped = lines[j].strip()
                    if next_line_stripped: # Found next non-empty line
                        if next_line_stripped[0].isdigit() and ',' in next_line_stripped:
                            potential_header_index = i
                            logger.debug(f"Potential header found via common columns heuristic at line {i}.")
                        break # Stop checking next lines once one is found

            if potential_header_index != -1:
                 break # Exit main loop if potential header found by heuristic

            # If none of the above, assume it's still metadata (could be comments, etc.)
            metadata_lines.append(line)

        # --- Validation and Finalization ---
        if potential_header_index == -1:
            logger.error("Could not identify a candidate header row.")
            # Provide more debugging info
            logger.debug("First 20 lines for header check:")
            for k, l in enumerate(lines[:20]): logger.debug(f"L{k}: {l.strip()}")
            raise ValueError("Could not reliably identify the log data header. Check log format.")

        header_line = lines[potential_header_index].strip()
        # Add lines before the identified header index to metadata if not already added
        metadata_lines.extend(lines[:potential_header_index])
        # Remove duplicates while preserving order (simple approach)
        seen = set()
        metadata_lines = [x for x in metadata_lines if not (x in seen or seen.add(x))]


        # Find the actual start of data (first non-empty line after header)
        data_start_index = -1
        for i in range(potential_header_index + 1, len(lines)):
            if lines[i].strip():
                # Basic check: does it look like data (e.g., starts with a digit)?
                if lines[i].strip()[0].isdigit() or lines[i].strip().startswith('-'):
                    data_start_index = i
                    logger.info(f"Identified header at line {potential_header_index}, data starts at line {data_start_index}.")
                    break
                else:
                    # Found a non-empty line after header that doesn't look like data - possible issue
                    logger.warning(f"Line {i} after potential header doesn't look like data: '{lines[i].strip()[:50]}...'")
                    # Continue searching, maybe it's a stray comment line

        if data_start_index == -1:
            logger.error(f"Found header at line {potential_header_index}, but no subsequent data lines detected.")
            raise ValueError("No data found after the identified header row.")

        # Clean the identified header line (remove quotes, extra spaces)
        header_line = header_line.replace('"', '').strip()

        return metadata_lines, header_line, data_start_index


    def parse_metadata(self, metadata_lines: List[str]) -> Dict[str, Any]:
        """
        Parses key-value pairs and specific settings from metadata lines.

        Args:
            metadata_lines (List[str]): Lines identified as metadata.

        Returns:
            Dict[str, Any]: A structured dictionary containing parsed metadata.
        """
        logger.debug(f"Parsing metadata from {len(metadata_lines)} lines.")
        metadata: Dict[str, Any] = {
            'firmware': {}, 'hardware': {}, 'pids': {}, 'rates': {},
            'filters': {}, 'features': [], 'other_settings': {},
            'raw_headers': [], # Store original header lines
            'analysis_info': {} # For storing analysis-related info like time units
        }

        # --- Patterns (Grouped for Clarity) ---
        # Use re.IGNORECASE for flexibility

        # General Info (often starts with 'H ')
        general_patterns = {
            'product': re.compile(r'H Product:\s*(.*)', re.IGNORECASE),
            'bf_version': re.compile(r'H Firmware:\s*Betaflight / \w+ ([\d.]+)', re.IGNORECASE),
            'target': re.compile(r'H Firmware target:\s*(\S+)', re.IGNORECASE),
            'mcu_id': re.compile(r'H MCU Id:\s*(\S+)', re.IGNORECASE),
            'gyro_accel': re.compile(r'H Gyro/Accel:\s*(.*)', re.IGNORECASE),
            'board_name': re.compile(r'H Board:\s*(\S+)', re.IGNORECASE), # Sometimes 'H Board Name:'
            'craft_name': re.compile(r'H Craft name:\s*(.*)', re.IGNORECASE),
            'log_rate_hz': re.compile(r'H logRateHz:\s*(\d+)', re.IGNORECASE),
            'gyro_scale': re.compile(r'H gyro_scale:\s*([\d.e+-]+)', re.IGNORECASE),
            'acc_1g': re.compile(r'H acc_1G:\s*(\d+)', re.IGNORECASE),
            'motor_output_range': re.compile(r'H motorOutput:\s*(\d+)\s+(\d+)', re.IGNORECASE),
            'debug_mode': re.compile(r'H debug_mode:\s*(\w+)', re.IGNORECASE),
            'disabled_fields': re.compile(r'H DisabledFields:\s*(.*)', re.IGNORECASE),
        }

        # PID/Rate/Filter Settings (often key-value pairs, sometimes quoted)
        # Pattern to capture key-value pairs like "key,value" or "key","value"
        # Handles spaces around comma and optional quotes
        setting_pattern = re.compile(r'^"?([^",]+)"?\s*,\s*"?([^"]+)"?\s*$', re.IGNORECASE)

        # Specific settings that might need special handling or categorization
        known_settings_map = {
            # PIDs (assuming values are comma-separated for axes if applicable)
            "rollpid": "pids", "pitchpid": "pids", "yawpid": "pids", "levelpid": "pids",
            "p_roll": "pids", "i_roll": "pids", "d_roll": "pids", "f_roll": "pids",
            "p_pitch": "pids", "i_pitch": "pids", "d_pitch": "pids", "f_pitch": "pids",
            "p_yaw": "pids", "i_yaw": "pids", "d_yaw": "pids", "f_yaw": "pids",
            "d_min_roll": "pids", "d_min_pitch": "pids", "d_min_yaw": "pids",
            "d_min_gain": "pids", "d_min_advance": "pids",
            "feedforward_transition": "pids", "ff_boost": "pids",
            "iterm_relax": "pids", "iterm_relax_type": "pids", "iterm_relax_cutoff": "pids",
            "anti_gravity_gain": "pids", "anti_gravity_mode": "pids",
            "tpa_rate": "pids", "tpa_breakpoint": "pids", "tpa_mode": "pids",
            "abs_control_gain": "pids", "throttle_boost": "pids",
            # Rates
            "rates": "rates", "rc_rates": "rates", "rc_expo": "rates", "rc_rate_limit": "rates",
            "thrmid": "rates", "threxpo": "rates", "rates_type": "rates",
            "roll_rate": "rates", "pitch_rate": "rates", "yaw_rate": "rates", # Older style?
            # Filters
            "gyro_hardware_lpf": "filters", "gyro_lpf": "filters", "gyro_lowpass_type": "filters",
            "gyro_lowpass_hz": "filters", "gyro_lowpass2_type": "filters", "gyro_lowpass2_hz": "filters",
            "gyro_notch1_hz": "filters", "gyro_notch1_cutoff": "filters",
            "gyro_notch2_hz": "filters", "gyro_notch2_cutoff": "filters",
            "dterm_filter_type": "filters", "dterm_lpf_hz": "filters",
            "dterm_lpf_dyn_min_hz": "filters", "dterm_lpf_dyn_max_hz": "filters", # Dynamic DTerm LPF
            "dterm_filter2_type": "filters", "dterm_lpf2_hz": "filters",
            "dterm_notch_hz": "filters", "dterm_notch_cutoff": "filters", "dterm_dyn_notch_enable": "filters",
            "yaw_lpf_hz": "filters",
            "dyn_notch_enable": "filters", "dyn_notch_q": "filters", "dyn_notch_min_hz": "filters", "dyn_notch_max_hz": "filters",
            "rpm_filter_enable": "filters", "rpm_filter_harmonics": "filters", "rpm_filter_q": "filters", "rpm_filter_min_hz": "filters",
            # Features (often a single bitmask or list)
            "features": "features",
            # Other Settings
            "looptime": "other_settings", "pid_process_denom": "other_settings",
            "vbat_pid_gain": "other_settings", "use_integrated_yaw": "other_settings",
            "dshot_bidir": "other_settings", "motor_poles": "other_settings",
            "serialrx_provider": "other_settings", "blackbox_device": "other_settings",
            "blackbox_rate_num": "other_settings", "blackbox_rate_denom": "other_settings",
        }

        # --- Process Lines ---
        for line in metadata_lines:
            stripped_line = line.strip()
            if not stripped_line: continue

            # Store raw header lines
            if stripped_line.startswith('H '):
                metadata['raw_headers'].append(stripped_line)

            matched = False
            # Check general patterns first
            for key, pattern in general_patterns.items():
                match = pattern.match(stripped_line)
                if match:
                    try:
                        if len(match.groups()) == 1:
                            value = match.group(1).strip()
                            # Simple type conversion for known numeric fields
                            if key in ['log_rate_hz', 'acc_1g', 'gyro_scale']:
                                value = float(value) if '.' in value else int(value)
                            metadata['firmware'][key] = value # Store most under firmware/hardware
                        elif len(match.groups()) == 2: # e.g., motorOutput range
                             metadata['firmware'][key] = [int(match.group(1)), int(match.group(2))]
                        matched = True
                        break # Pattern matched
                    except ValueError:
                         logger.warning(f"Could not convert value for metadata key '{key}' in line: {stripped_line}")
                         metadata['firmware'][key] = match.group(1).strip() # Store as string
                         matched = True
                         break
                    except Exception as e:
                         logger.warning(f"Error processing metadata key '{key}' from line '{stripped_line}': {e}")
                         matched = True
                         break
            if matched: continue

            # Check for key-value setting patterns
            match = setting_pattern.match(stripped_line)
            if match:
                key = match.group(1).strip().lower()
                value_str = match.group(2).strip()
                category = known_settings_map.get(key, 'other_settings') # Default to other_settings

                # Attempt to parse value (handle lists, numbers, strings)
                parsed_value: Any
                if ',' in value_str: # Potential list
                    try:
                        # Try converting items to float, fallback to string
                        parsed_value = [float(v.strip()) if '.' in v.strip() or 'e' in v.strip().lower() else int(v.strip())
                                        for v in value_str.split(',') if v.strip()]
                    except ValueError:
                        parsed_value = [v.strip() for v in value_str.split(',') if v.strip()] # Keep as strings
                    # If list has only one item, store it directly
                    if len(parsed_value) == 1:
                        parsed_value = parsed_value[0]
                else: # Single value
                    try:
                        # Try int first, then float, then keep as string
                        parsed_value = int(value_str)
                    except ValueError:
                        try:
                            parsed_value = float(value_str)
                        except ValueError:
                            parsed_value = value_str # Keep original string

                # Handle 'features' specifically (can be bitmask or list)
                if category == 'features':
                    if isinstance(parsed_value, list):
                         metadata[category].extend(parsed_value) # Append feature strings
                    else:
                         metadata[category].append(parsed_value) # Append single feature/bitmask
                elif category:
                    metadata[category][key] = parsed_value
                matched = True

            # if not matched: # Optional: Log lines that didn't match any pattern
            #     logger.debug(f"Metadata line not matched by patterns: {stripped_line}")


        # Post-processing and cleanup
        # - Consolidate features into unique list
        if isinstance(metadata.get('features'), list):
             try:
                  # Attempt to convert numeric feature masks/strings to int if possible
                  processed_features = []
                  for f in metadata['features']:
                       try: processed_features.append(int(f))
                       except (ValueError, TypeError): processed_features.append(str(f)) # Keep as string if not int
                  metadata['features'] = sorted(list(set(processed_features)))
             except Exception:
                  logger.warning("Could not process features list, keeping raw.")
                  metadata['features'] = list(set(metadata['features'])) # Simple unique strings

        # - Convert scales/rates if present
        try:
            if 'gyro_scale' in metadata.get('firmware', {}):
                 metadata['firmware']['gyro_scale'] = float(metadata['firmware']['gyro_scale'])
            if 'log_rate_hz' in metadata.get('firmware', {}):
                 metadata['firmware']['log_rate_hz'] = int(metadata['firmware']['log_rate_hz'])
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Could not convert known numeric metadata fields: {e}")

        # Add analysis timestamp
        metadata['analysis_info']['analysis_timestamp'] = datetime.now().isoformat()
        logger.info(f"Metadata parsing complete. Found keys in categories: { {k: len(v) for k, v in metadata.items() if isinstance(v, (dict, list))} }")
        return metadata


    # --- Data Parsing and Preparation ---

    def parse_data(self, header_line: str, data_lines: List[str]) -> pd.DataFrame:
        """
        Parses the data section of the log into a Pandas DataFrame.

        Args:
            header_line (str): The cleaned, comma-separated header string.
            data_lines (List[str]): A list of strings, each representing a row of data.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed log data.

        Raises:
            ValueError: If parsing fails or the resulting DataFrame is empty.
        """
        logger.debug(f"Parsing {len(data_lines)} data lines using header: '{header_line[:100]}...'")
        if not data_lines:
            raise ValueError("No data lines provided for parsing.")

        # Combine header and data lines into a CSV-like string
        # Ensure header has a newline, handle potential newlines in data_lines
        csv_content = header_line.strip() + "\n" + "".join(line for line in data_lines)
        csv_file = io.StringIO(csv_content)

        try:
            # Use pandas read_csv for robust parsing
            df = pd.read_csv(
                csv_file,
                header=0,             # First line is the header
                skipinitialspace=True,# Handle potential spaces after commas
                on_bad_lines='warn',  # Warn about problematic lines but try to continue
                quotechar='"',        # Standard quote character
                low_memory=False      # Recommended for potentially mixed types / large files
            )
            logger.debug(f"Pandas read_csv successful. Initial shape: {df.shape}")

            if df.empty:
                 raise ValueError("Parsing resulted in an empty DataFrame. Check log format or content.")

            # --- Column Cleaning ---
            # 1. Remove potential leading/trailing whitespace from column names
            df.columns = df.columns.str.strip()
            # 2. Remove potential quotes from column names
            df.columns = df.columns.str.replace('"', '', regex=False).str.replace("'", '', regex=False)
            # 3. Handle potential duplicate columns (e.g., if header had issues)
            if df.columns.has_duplicates:
                logger.warning(f"Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}. Keeping first occurrence.")
                df = df.loc[:, ~df.columns.duplicated(keep='first')]

            logger.debug(f"Cleaned columns: {df.columns.tolist()}")
            return df

        except pd.errors.EmptyDataError:
            logger.error("Pandas read_csv failed: No columns to parse from file content.")
            raise ValueError("Log content appears empty or lacks a valid header/data structure.") from None
        except Exception as e:
            logger.error(f"Error parsing data with pandas: {e}", exc_info=True)
            logger.error(f"Header used: {header_line[:100]}...")
            logger.error(f"First data line: {data_lines[0][:100] if data_lines else 'N/A'}...")
            raise ValueError(f"Failed to parse log data into DataFrame: {e}") from e

    def prepare_data(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Cleans, prepares, and validates the DataFrame for analysis.

        - Handles the 'time' column (identification, units, conversion).
        - Converts data columns to numeric types.
        - Handles missing values (NaNs).
        - Sets and validates the time index.

        Args:
            df (pd.DataFrame): The raw DataFrame parsed from the log.
            metadata (Dict[str, Any]): The parsed metadata dictionary. This dictionary
                                       will be updated with 'time_unit' information.

        Returns:
            pd.DataFrame: The prepared DataFrame with a time-based index and numeric columns.

        Raises:
            ValueError: If essential time information is missing or invalid,
                        or if the DataFrame becomes empty after preparation.
        """
        logger.debug(f"Starting data preparation. Initial shape: {df.shape}, Columns: {df.columns.tolist()}")
        df_prepared = df.copy() # Work on a copy

        # --- 1. Handle Time Column ---
        time_col_name = None
        time_unit = 'unknown'

        # Find potential time columns (case-insensitive)
        potential_time_cols = [col for col in df_prepared.columns if col.lower() == 'time']
        if potential_time_cols:
            time_col_name = potential_time_cols[0]
            logger.debug(f"Found time column: '{time_col_name}'")
            # Attempt to convert to numeric, coercing errors
            df_prepared[time_col_name] = pd.to_numeric(df_prepared[time_col_name], errors='coerce')
            if df_prepared[time_col_name].isnull().all():
                logger.warning(f"Time column '{time_col_name}' contains only non-numeric values.")
                time_col_name = None # Treat as missing if all NaN after conversion
            else:
                 # Assume microseconds if it's numeric and wasn't explicitly set otherwise
                 time_unit = metadata.get('analysis_info', {}).get('time_unit', 'us')
                 logger.info(f"Using existing numeric time column '{time_col_name}'. Assuming units: {time_unit}.")
        else:
            # Fallback: Check for 'loopIteration'
            potential_loop_cols = [col for col in df_prepared.columns if col.lower() == 'loopiteration']
            if potential_loop_cols:
                time_col_name = potential_loop_cols[0]
                logger.info(f"Using '{time_col_name}' as time column.")
                df_prepared[time_col_name] = pd.to_numeric(df_prepared[time_col_name], errors='coerce')
                if df_prepared[time_col_name].isnull().all():
                     logger.warning(f"LoopIteration column '{time_col_name}' contains only non-numeric values.")
                     time_col_name = None
                else:
                     time_unit = 'us' # loopIteration is typically microseconds

        # If no time column found, raise error
        if time_col_name is None:
            raise ValueError("Missing essential time information ('time' or 'loopIteration' column). Cannot proceed.")

        # Store identified time unit in metadata
        if 'analysis_info' not in metadata: metadata['analysis_info'] = {}
        metadata['analysis_info']['time_unit'] = time_unit
        metadata['analysis_info']['time_column_used'] = time_col_name

        # --- 2. Convert Data Columns to Numeric ---
        numeric_cols = []
        non_numeric_issues = []
        for col in df_prepared.columns:
            if col == time_col_name: continue # Skip the time column itself

            if not pd.api.types.is_numeric_dtype(df_prepared[col]):
                original_dtype = df_prepared[col].dtype
                original_nan_count = df_prepared[col].isnull().sum()
                # Attempt conversion, coercing errors to NaN
                df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
                new_nan_count = df_prepared[col].isnull().sum()
                nan_increase = new_nan_count - original_nan_count

                if nan_increase > 0:
                    logger.debug(f"Column '{col}' (dtype {original_dtype}): Coerced {nan_increase} non-numeric values to NaN.")
                    # Log a warning if a significant portion became NaN
                    if nan_increase > 0.1 * len(df_prepared): # More than 10% became NaN
                        non_numeric_issues.append(f"{col} ({nan_increase} coerced)")

            # Add to list if now numeric
            if pd.api.types.is_numeric_dtype(df_prepared[col]):
                numeric_cols.append(col)
            elif col not in [time_col_name]: # Log if still not numeric (and not time col)
                 logger.warning(f"Column '{col}' could not be converted to a numeric type.")

        if non_numeric_issues:
            logger.warning(f"High NaN count after numeric conversion for columns: {', '.join(non_numeric_issues)}.")

        # --- 3. Handle Missing Values (NaNs) ---
        initial_rows = len(df_prepared)

        # Drop rows where essential data (e.g., gyro) is missing entirely
        essential_gyro_cols = [c for c in df_prepared.columns if c.lower().startswith('gyroadc[') or c.lower() in ['gyroroll', 'gyropitch', 'gyroyaw']]
        essential_cols_present = [c for c in essential_gyro_cols if c in df_prepared.columns and c in numeric_cols]

        if essential_cols_present:
            # Also check the time column for NaNs before dropping
            essential_check_cols = essential_cols_present + [time_col_name]
            df_prepared.dropna(subset=essential_check_cols, how='any', inplace=True)
            rows_dropped = initial_rows - len(df_prepared)
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} rows due to missing essential gyro or time data.")
                initial_rows = len(df_prepared) # Update row count
        else:
             logger.warning("No essential gyro columns found or they are not numeric. Skipping NaN row drop based on gyro.")
             # Still drop rows with NaN in the time column
             df_prepared.dropna(subset=[time_col_name], how='any', inplace=True)
             rows_dropped = initial_rows - len(df_prepared)
             if rows_dropped > 0:
                  logger.warning(f"Dropped {rows_dropped} rows due to missing time data.")
                  initial_rows = len(df_prepared) # Update row count


        # Fill remaining NaNs in *numeric* columns using ffill/bfill
        numeric_cols_to_fill = [c for c in numeric_cols if c in df_prepared.columns] # Re-check columns exist after dropna
        if numeric_cols_to_fill:
            nan_before_fill = df_prepared[numeric_cols_to_fill].isnull().sum().sum()
            if nan_before_fill > 0:
                df_prepared[numeric_cols_to_fill] = df_prepared[numeric_cols_to_fill].ffill().bfill()
                nan_after_fill = df_prepared[numeric_cols_to_fill].isnull().sum().sum()
                filled_count = nan_before_fill - nan_after_fill
                logger.debug(f"Filled {filled_count} missing values in numeric columns using ffill/bfill.")
                if nan_after_fill > 0:
                     logger.warning(f"{nan_after_fill} NaNs remain in numeric columns after ffill/bfill (likely entire columns were NaN).")
        else:
             logger.debug("No numeric columns found to fill NaNs.")


        # --- 4. Set Time Index ---
        if df_prepared.empty:
             raise ValueError("DataFrame became empty after handling missing values.")

        # Check for duplicate timestamps before setting index
        if df_prepared[time_col_name].duplicated().any():
            duplicates_count = df_prepared[time_col_name].duplicated().sum()
            logger.warning(f"Found {duplicates_count} duplicate timestamps in column '{time_col_name}'. Keeping the last occurrence.")
            df_prepared = df_prepared.drop_duplicates(subset=[time_col_name], keep='last')

        # Set index
        try:
            df_prepared.set_index(time_col_name, inplace=True)
            if not df_prepared.index.is_monotonic_increasing:
                logger.debug("Time index is not monotonic increasing. Sorting...")
                df_prepared.sort_index(inplace=True)
            logger.debug(f"Successfully set '{time_col_name}' as index.")
        except KeyError:
             # This shouldn't happen if time_col_name was validated earlier
             logger.error(f"Failed to set index: Column '{time_col_name}' not found.")
             raise ValueError(f"Internal error: Time column '{time_col_name}' lost during preparation.")
        except Exception as e:
             logger.error(f"Error setting '{time_col_name}' as index: {e}", exc_info=True)
             # Decide if this is critical - analysis might still work with default index
             # raise ValueError(f"Could not set time index: {e}") from e
             logger.warning("Proceeding without time-based index due to error.")


        if df_prepared.empty:
            raise ValueError("DataFrame became empty after final preparation steps.")

        logger.info(f"Data preparation finished. Final shape: {df_prepared.shape}")
        return df_prepared

    # --- Analysis Methods ---

    def diagnose_data_quality(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assesses the quality and integrity of the prepared log data.

        Checks for missing data, unusual values, sampling issues, and sensor flatlines.

        Args:
            df (pd.DataFrame): The prepared DataFrame (should have time index).
            metadata (Dict[str, Any]): Parsed log metadata.

        Returns:
            Dict[str, Any]: A dictionary containing diagnostic results, including:
                - 'missing_data': Info on missing columns or high NaN percentages.
                - 'unusual_values': List of columns with potential outliers or flatlines.
                - 'sampling_issues': Info on irregular sampling or data gaps.
                - 'quality_score': A score from 0.0 (poor) to 1.0 (excellent).
                - 'summary': A brief text summary of the quality.
                - 'diagnosis': A list of specific issues found.
        """
        logger.debug("Diagnosing data quality...")
        diagnostics: Dict[str, Any] = {
            "missing_data": {},
            "unusual_values": [],
            "sampling_issues": {},
            "quality_score": 1.0, # Start with perfect score
            "summary": "Data quality checks passed.", # Default summary
            "diagnosis": [] # List of specific issues found
        }
        total_rows = len(df)

        if total_rows == 0:
            diagnostics["summary"] = "No data rows available for quality diagnosis."
            diagnostics["quality_score"] = 0.0
            diagnostics["diagnosis"].append("Empty DataFrame after preparation.")
            return diagnostics

        # --- Check for essential columns ---
        # Essential for core flight analysis
        essential_columns = [
            'gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]', # Prefer array notation
            'gyroRoll', 'gyroPitch', 'gyroYaw',      # Allow legacy names
            'motor[0]', 'motor[1]', 'motor[2]', 'motor[3]'
        ]
        present_columns = df.columns.str.lower().tolist()
        missing_essentials = []
        has_any_gyro = False
        has_any_motor = False

        # Check for at least one gyro axis
        if not any(c.lower() in present_columns for c in ['gyroadc[0]', 'gyroroll']):
             missing_essentials.append("Gyro Roll")
        else: has_any_gyro = True
        if not any(c.lower() in present_columns for c in ['gyroadc[1]', 'gyropitch']):
             missing_essentials.append("Gyro Pitch")
        else: has_any_gyro = True
        if not any(c.lower() in present_columns for c in ['gyroadc[2]', 'gyroyaw']):
             missing_essentials.append("Gyro Yaw")
        else: has_any_gyro = True

        # Check for at least one motor output
        if not any(c.lower().startswith('motor[') for c in present_columns):
             missing_essentials.append("Motor Outputs")
        else: has_any_motor = True

        if not has_any_gyro:
            diagnostics["missing_data"]["essential_columns"] = "Gyro data (Roll/Pitch/Yaw) missing."
            diagnostics["quality_score"] -= 0.4 # Significant penalty
            diagnostics["diagnosis"].append("Missing essential gyro columns.")
        elif missing_essentials and "Motor Outputs" not in missing_essentials: # Some gyro axes missing
             diagnostics["missing_data"]["missing_gyro_axes"] = [m for m in missing_essentials if "Gyro" in m]
             diagnostics["quality_score"] -= 0.1 # Minor penalty if some gyro exists
             diagnostics["diagnosis"].append("Missing some gyro axis data.")

        if not has_any_motor:
             diagnostics["missing_data"]["essential_columns"] = diagnostics["missing_data"].get("essential_columns","") + " Motor data missing."
             diagnostics["quality_score"] -= 0.2 # Significant penalty
             diagnostics["diagnosis"].append("Missing essential motor columns.")


        # --- Check NaN percentages (post-preparation) ---
        # Should be low after prepare_data, but check anyway
        nan_percentages = (df.isnull().sum() / total_rows * 100)
        high_nan_cols = nan_percentages[nan_percentages > 5] # Lower threshold now
        if not high_nan_cols.empty:
            diagnostics["missing_data"]["high_nan_percentages (>5%)"] = high_nan_cols.round(1).to_dict()
            diagnostics["quality_score"] -= 0.1 # Minor penalty, as filling was attempted
            diagnostics["diagnosis"].append(f"High percentage of remaining NaNs in columns: {list(high_nan_cols.index)}")

        # --- Check time sampling regularity ---
        time_unit = metadata.get('analysis_info', {}).get('time_unit', 'unknown')
        if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)) or np.issubdtype(df.index.dtype, np.number):
            time_diffs = pd.Series(df.index).diff().dropna()
            if len(time_diffs) > 1:
                # Convert diffs to seconds for consistent comparison
                median_diff_val = time_diffs.median()
                std_diff_val = time_diffs.std()

                time_diff_median_sec = 0.0
                time_diff_std_sec = 0.0

                if time_unit == 'us':
                    time_diff_median_sec = median_diff_val / 1_000_000.0
                    time_diff_std_sec = std_diff_val / 1_000_000.0
                elif isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
                     # Assuming Timedelta results from diff()
                     time_diff_median_sec = median_diff_val.total_seconds()
                     time_diff_std_sec = std_diff_val.total_seconds()
                else: # Assume seconds if numeric index and unit unknown/not 'us'
                    time_diff_median_sec = median_diff_val
                    time_diff_std_sec = std_diff_val

                diagnostics["sampling_issues"]["median_interval_ms"] = round(time_diff_median_sec * 1000, 3)
                diagnostics["sampling_issues"]["std_dev_interval_ms"] = round(time_diff_std_sec * 1000, 3)

                # Check for irregular sampling intervals (high std dev relative to median)
                # Threshold: Std dev > 20% of median interval
                if time_diff_median_sec > 1e-9 and time_diff_std_sec > 0.2 * time_diff_median_sec:
                    diagnostics["sampling_issues"]["irregular_sampling"] = True
                    diagnostics["quality_score"] -= 0.15
                    diagnostics["diagnosis"].append("Irregular sampling intervals detected (check log rate consistency).")

                # Check for large gaps (e.g., > 10x median interval or > 0.1 seconds)
                gap_threshold_val = 0
                if time_unit == 'us': gap_threshold_val = max(10 * median_diff_val, 0.1 * 1_000_000)
                elif isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)): gap_threshold_val = max(10 * median_diff_val, pd.Timedelta(seconds=0.1))
                else: gap_threshold_val = max(10 * median_diff_val, 0.1) # Assume seconds

                large_gaps = time_diffs[time_diffs > gap_threshold_val]
                if not large_gaps.empty:
                    max_gap_val = large_gaps.max()
                    max_gap_sec = 0.0
                    if time_unit == 'us': max_gap_sec = max_gap_val / 1_000_000.0
                    elif isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)): max_gap_sec = max_gap_val.total_seconds()
                    else: max_gap_sec = max_gap_val # Assume seconds

                    diagnostics["sampling_issues"]["data_gaps_count"] = len(large_gaps)
                    diagnostics["sampling_issues"]["max_gap_s"] = round(max_gap_sec, 3)
                    diagnostics["quality_score"] -= 0.1
                    diagnostics["diagnosis"].append(f"Found {len(large_gaps)} significant gaps in time data (max: {max_gap_sec:.3f}s).")
            else:
                 logger.debug("Not enough time differences to analyze sampling regularity.")
        else:
            logger.warning("Time index is not numeric or time-based. Cannot analyze sampling regularity.")
            diagnostics["sampling_issues"]["error"] = "Cannot analyze sampling (invalid index type)."


        # --- Check for unusual value ranges and flatlines ---
        numeric_df = df.select_dtypes(include=np.number) # Select only numeric columns for checks
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if col_data.empty: continue

            # Check Gyro ranges (assuming deg/s)
            if col.lower().startswith('gyroadc[') or col.lower() in ['gyroroll', 'gyropitch', 'gyroyaw']:
                # Check for extreme values (e.g., > 2500 deg/s, common limit)
                extreme_threshold = 2500
                extreme_values_pct = (col_data.abs() > extreme_threshold).mean() * 100
                if extreme_values_pct > 1.0: # More than 1% extreme values
                    diagnostics["unusual_values"].append({
                        "column": col, "issue": f"Frequent extreme values (> +/-{extreme_threshold} deg/s)",
                        "percentage": round(extreme_values_pct, 1)
                    })
                    diagnostics["quality_score"] -= 0.05
                    diagnostics["diagnosis"].append(f"Frequent high values detected in {col}.")

            # Check Motor ranges (handle different scales)
            elif col.lower().startswith('motor['):
                 min_val, max_val = col_data.min(), col_data.max()
                 # Check typical PWM/DShot range (1000-2000) with some tolerance
                 if not ((min_val >= 950 and max_val <= 2050) or \
                         (min_val >= -5 and max_val <= 1.05)): # Allow for normalized range 0-1
                      # Check if range seems reasonable but outside standard scale
                      if min_val >= 0 and max_val <= 2200: # Allow slightly wider range
                           logger.debug(f"Motor column {col} has range [{min_val:.1f}, {max_val:.1f}], outside standard but plausible.")
                      else:
                           diagnostics["unusual_values"].append({
                                "column": col, "issue": "Values outside typical range (0-1 or 1000-2000)",
                                "range": [round(min_val,1), round(max_val,1)]
                           })
                           diagnostics["quality_score"] -= 0.1
                           diagnostics["diagnosis"].append(f"Motor values ({col}) outside expected range.")

            # Check for flatlined data (low standard deviation over rolling window)
            # Apply only to sensors like gyro, accel where variation is expected
            if col.lower().startswith(('gyroadc[', 'accsmooth[', 'gyroroll', 'gyropitch', 'gyroyaw')):
                if len(col_data) > 100: # Need enough data for rolling window
                    try:
                        # Use a window size relative to sampling rate if possible, default 50
                        sampling_rate = 1.0 / time_diff_median_sec if 'time_diff_median_sec' in locals() and time_diff_median_sec > 1e-9 else 1000 # Default 1kHz
                        window_size = max(20, min(100, int(sampling_rate * 0.05))) # e.g., 50ms window

                        rolling_std = col_data.rolling(window=window_size, center=True, min_periods=window_size//2).std()
                        # Define 'flat' threshold based on overall signal std dev
                        overall_std = col_data.std()
                        flatline_threshold = max(0.01, overall_std * 0.01) if overall_std > 1e-6 else 0.01 # 1% of std dev or minimum 0.01
                        flatline_pct = (rolling_std.dropna() < flatline_threshold).mean() * 100

                        if flatline_pct > 15.0: # More than 15% of time looks flat
                            diagnostics["unusual_values"].append({
                                "column": col, "issue": "Potential flatlined data (low variation)",
                                "flatline_percentage": round(flatline_pct, 1),
                                "threshold_used": round(flatline_threshold, 4)
                            })
                            diagnostics["quality_score"] -= 0.1
                            diagnostics["diagnosis"].append(f"Potential flatlined sensor data detected in {col}.")
                    except Exception as e_flat:
                         logger.warning(f"Could not perform flatline check for {col}: {e_flat}")


        # --- Final Score and Summary ---
        score = max(0.0, min(1.0, diagnostics["quality_score"])) # Clamp score
        diagnostics["quality_score"] = round(score, 2)

        if score < 0.4:
            diagnostics["summary"] = "Poor data quality. Analysis reliability may be very low due to significant issues."
        elif score < 0.6:
            diagnostics["summary"] = "Fair data quality. Some analysis results may be affected by identified issues."
        elif score < 0.8:
            diagnostics["summary"] = "Good data quality, but minor issues detected that might affect analysis."
        else: # Score >= 0.8
            if not diagnostics["diagnosis"]:
                diagnostics["summary"] = "Excellent data quality. No significant issues detected."
            else:
                diagnostics["summary"] = f"Good/Excellent data quality, but note potential minor issues: {diagnostics['diagnosis'][0]}"

        logger.info(f"Data Quality Score: {diagnostics['quality_score']:.2f}. Summary: {diagnostics['summary']}")
        return diagnostics

# --- Continuing BetaflightLogAnalyzer Class ---

    def _find_col(self, df: pd.DataFrame, potential_names: List[str]) -> Optional[str]:
        """Helper to find the first matching column name (case-insensitive)."""
        df_cols_lower = [c.lower() for c in df.columns]
        for name in potential_names:
            try:
                idx = df_cols_lower.index(name.lower())
                return df.columns[idx] # Return original case name
            except ValueError:
                continue
        return None

    def analyze_pid_performance(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes PID tracking error and step response characteristics.

        Handles multiple common naming formats for gyro and setpoint columns.

        Args:
            df (pd.DataFrame): The prepared DataFrame with a time index.
            metadata (Dict[str, Any]): Parsed log metadata.

        Returns:
            Dict[str, Any]: Dictionary containing PID analysis results under the 'pid' key.
                            Includes errors if essential data is missing.
        """
        logger.debug("Analyzing PID performance...")
        results: Dict[str, Any] = {} # Results specific to this analysis step

        # Define potential column names for each axis
        gyro_potential = {
            'roll': ['gyroADC[0]', 'gyroRoll', 'gyro_roll'],
            'pitch': ['gyroADC[1]', 'gyroPitch', 'gyro_pitch'],
            'yaw': ['gyroADC[2]', 'gyroYaw', 'gyro_yaw']
        }
        setpoint_potential = {
            'roll': ['setpoint[0]', 'setpointRoll', 'setpoint_roll'],
            'pitch': ['setpoint[1]', 'setpointPitch', 'setpoint_pitch'],
            'yaw': ['setpoint[2]', 'setpointYaw', 'setpoint_yaw']
        }

        # Find actual column names used in the DataFrame
        gyro_cols = {axis: self._find_col(df, names) for axis, names in gyro_potential.items()}
        setpoint_cols = {axis: self._find_col(df, names) for axis, names in setpoint_potential.items()}

        has_any_gyro = any(gyro_cols.values())
        has_any_setpoint = any(setpoint_cols.values())

        if not has_any_gyro:
            results["error_gyro"] = "No gyro data columns found (checked common names like gyroADC[0], gyroRoll)."
            logger.error(results["error_gyro"])
            return {"pid": results} # Return wrapped in 'pid' key

        analysis_performed_on_axis = []
        for axis in ['roll', 'pitch', 'yaw']:
            gyro_col = gyro_cols.get(axis)
            setpoint_col = setpoint_cols.get(axis)

            axis_results: Dict[str, Any] = {}

            # --- Analyze Gyro Data (Always perform if available) ---
            if gyro_col and gyro_col in df.columns and pd.api.types.is_numeric_dtype(df[gyro_col]):
                gyro_data = df[gyro_col].dropna()
                if not gyro_data.empty and len(gyro_data) > 1:
                    axis_results[f"{axis}_gyro_mean"] = gyro_data.mean()
                    axis_results[f"{axis}_gyro_std"] = gyro_data.std()
                    axis_results[f"{axis}_gyro_range"] = gyro_data.max() - gyro_data.min()
                    # Basic noise metric (std dev of differences)
                    gyro_diff = gyro_data.diff().dropna()
                    if not gyro_diff.empty:
                         axis_results[f"{axis}_gyro_noise_metric_std"] = gyro_diff.std()

                    # --- Analyze Tracking Error & Step Response (If Setpoint exists) ---
                    if setpoint_col and setpoint_col in df.columns and pd.api.types.is_numeric_dtype(df[setpoint_col]):
                        setpoint_data = df[setpoint_col].dropna()
                        if not setpoint_data.empty:
                            # Align gyro and setpoint data on their index (time)
                            aligned_gyro, aligned_setpoint = gyro_data.align(setpoint_data, join='inner')

                            if not aligned_gyro.empty and len(aligned_gyro) > 10: # Need some data points
                                # Calculate Tracking Error
                                error = aligned_setpoint - aligned_gyro
                                axis_results[f"{axis}_tracking_error_mean"] = error.mean()
                                axis_results[f"{axis}_tracking_error_std"] = error.std()
                                axis_results[f"{axis}_tracking_error_mae"] = error.abs().mean() # Mean Absolute Error
                                axis_results[f"{axis}_tracking_error_max_abs"] = error.abs().max()

                                # Calculate Step Response if enough data
                                min_points_for_step = 50 # Require more points for reliable step analysis
                                if len(aligned_gyro) > min_points_for_step:
                                    logger.debug(f"Calculating step response for {axis}...")
                                    # Pass aligned data and metadata (for time unit)
                                    step_metrics = self.calculate_step_response(aligned_setpoint, aligned_gyro, metadata)
                                    axis_results[f"{axis}_step_response"] = step_metrics
                                else:
                                    logger.debug(f"Skipping step response for {axis}: Insufficient aligned data points ({len(aligned_gyro)} <= {min_points_for_step}).")
                                    axis_results[f"{axis}_step_response"] = {"status": f"Insufficient data ({len(aligned_gyro)} points)"}
                            else:
                                logger.warning(f"Could not align gyro/setpoint or not enough aligned data for {axis} tracking analysis.")
                                axis_results[f"{axis}_tracking_error_warning"] = "Alignment failed or insufficient data."
                                axis_results[f"{axis}_step_response"] = {"status": "Alignment failed or insufficient data."}
                        else:
                             logger.warning(f"Setpoint column '{setpoint_col}' for {axis} is empty after dropping NaNs.")
                             axis_results[f"{axis}_tracking_error_warning"] = "Setpoint data empty."
                             axis_results[f"{axis}_step_response"] = {"status": "Setpoint data empty."}
                    else:
                         # Gyro exists but Setpoint doesn't
                         logger.debug(f"Gyro data found for {axis}, but no corresponding setpoint data for tracking analysis.")
                         axis_results[f"{axis}_tracking_error_warning"] = "Setpoint data missing or non-numeric."
                         axis_results[f"{axis}_step_response"] = {"status": "Setpoint data missing."}

                    # If any gyro analysis was done for this axis, add results
                    if axis_results:
                         results.update(axis_results)
                         analysis_performed_on_axis.append(axis)
                else:
                     logger.warning(f"Gyro column '{gyro_col}' for {axis} is empty or has insufficient data after dropping NaNs.")
            else:
                 logger.debug(f"Gyro column not found or not numeric for {axis}.")


        if not analysis_performed_on_axis:
            results["error_overall"] = "Could not perform PID/Gyro analysis on any axis."
        elif not has_any_setpoint:
            results["warning_setpoint"] = "No setpoint data found – performed gyro stability analysis only."

        logger.debug(f"PID analysis completed. Analyzed axes: {analysis_performed_on_axis}")
        return {"pid": results} # Return results wrapped in 'pid' key


    def calculate_step_response(self, setpoint: pd.Series, actual: pd.Series, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates step response metrics (rise time, overshoot, settling time)
        from aligned setpoint and actual (gyro) data.

        Args:
            setpoint (pd.Series): Aligned setpoint data with time index.
            actual (pd.Series): Aligned actual (gyro) data with time index.
            metadata (Dict[str, Any]): Parsed metadata containing time unit info.

        Returns:
            Dict[str, Any]: Dictionary containing median step response metrics:
                - 'status': Description of the calculation outcome.
                - 'rise_time_ms': Median rise time (10-90%) in milliseconds.
                - 'overshoot_percent': Median overshoot percentage.
                - 'settling_time_ms': Median settling time (to +/- 5% band) in milliseconds.
                - 'analyzed_steps_count': Number of valid steps analyzed.
        """
        metrics_list: List[Dict[str, Optional[float]]] = []
        status = "Calculation started"

        # --- Validate Input and Index ---
        if not isinstance(setpoint.index, pd.Index) or not isinstance(actual.index, pd.Index):
             return {"status": "Input Series must have a Pandas Index.", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None, "analyzed_steps_count": 0}
        if not setpoint.index.equals(actual.index):
             # This should not happen if align was used correctly before calling
             logger.error("Setpoint and Actual indices do not match in calculate_step_response.")
             return {"status": "Input indices do not match.", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None, "analyzed_steps_count": 0}

        index = setpoint.index
        time_unit = metadata.get('analysis_info', {}).get('time_unit', 'us') # Default to 'us'

        is_time_index = isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex))
        is_numeric_index = np.issubdtype(index.dtype, np.number)

        if not (is_time_index or is_numeric_index):
            logger.warning("Step response requires a numeric or time-based index.")
            return {"status": "Time index required", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None, "analyzed_steps_count": 0}

        # --- Detect Steps ---
        setpoint_diff = setpoint.diff().abs().dropna()
        if len(setpoint_diff) < 5:
            return {"status": "Not enough data points for step detection", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None, "analyzed_steps_count": 0}

        # Calculate dynamic threshold based on diff stats and overall range
        diff_median = setpoint_diff.median()
        diff_std = setpoint_diff.std()
        setpoint_range = setpoint.max() - setpoint.min()
        # Threshold: robust measure (median + 3*std) or % of range, whichever is larger
        threshold = max(diff_median + 3 * diff_std, 0.1 * setpoint_range if setpoint_range > 1e-6 else 1.0)
        threshold = max(threshold, 1e-3) # Ensure a minimum threshold to avoid noise triggers
        logger.debug(f"Step detection threshold: {threshold:.4f}")

        step_indices_raw = setpoint_diff[setpoint_diff > threshold].index

        # --- Filter Steps (Minimum Interval) ---
        min_step_interval_sec = 0.05 # Minimum 50ms between analyzed steps
        filtered_step_indices = []
        if not step_indices_raw.empty:
            last_step_time = step_indices_raw[0]
            filtered_step_indices.append(last_step_time)
            for current_step_time in step_indices_raw[1:]:
                try:
                    time_diff_val = current_step_time - last_step_time
                    time_diff_sec = 0.0
                    if is_time_index: time_diff_sec = time_diff_val.total_seconds()
                    elif is_numeric_index and time_unit == 'us': time_diff_sec = time_diff_val / 1_000_000.0
                    elif is_numeric_index: time_diff_sec = time_diff_val # Assume seconds if numeric but not 'us'

                    if time_diff_sec >= min_step_interval_sec:
                        filtered_step_indices.append(current_step_time)
                        last_step_time = current_step_time
                except Exception as e_filter:
                    logger.error(f"Error filtering step times: {e_filter}")
                    continue

        if not filtered_step_indices:
            return {"status": "No significant steps found after filtering", "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None, "analyzed_steps_count": 0}

        logger.debug(f"Found {len(filtered_step_indices)} potential steps to analyze.")

        # --- Analyze Each Step ---
        analysis_window_sec = 0.25 # How long after the step to analyze (increased slightly)
        max_steps_to_analyze = 30 # Limit analysis to avoid excessive computation

        for step_time in filtered_step_indices[:max_steps_to_analyze]:
            try:
                # Define analysis window end time based on index type
                end_time = None
                if is_time_index:
                    end_time = step_time + pd.Timedelta(seconds=analysis_window_sec)
                elif is_numeric_index and time_unit == 'us':
                    end_time = step_time + int(analysis_window_sec * 1_000_000)
                elif is_numeric_index: # Assume seconds
                    end_time = step_time + analysis_window_sec

                # Select data within the window, handle potential boundary issues
                try:
                    # Get index location for slicing robustly
                    start_idx_loc = setpoint.index.get_loc(step_time)
                    # Find end index location (might be beyond data range)
                    try:
                         end_idx_loc = setpoint.index.get_loc(end_time, method='bfill') # Find next available index
                    except KeyError:
                         end_idx_loc = len(setpoint) # Use end if end_time is past last index

                    window_setpoint = setpoint.iloc[start_idx_loc:end_idx_loc]
                    window_actual = actual.iloc[start_idx_loc:end_idx_loc]

                except KeyError:
                    logger.debug(f"Skipping step at {step_time}: window start/end time not found precisely in index.")
                    continue

                if len(window_setpoint) < 10 or len(window_actual) < 10:
                     logger.debug(f"Skipping step at {step_time}: insufficient data points ({len(window_setpoint)}) in window.")
                     continue

                # Determine step characteristics
                # Use median of first/last few points for robustness against noise
                initial_setpoint = np.median(window_setpoint.iloc[:min(5, len(window_setpoint))])
                final_setpoint = np.median(window_setpoint.iloc[-min(5, len(window_setpoint)):])
                initial_actual = np.median(window_actual.iloc[:min(5, len(window_actual))])
                setpoint_change = final_setpoint - initial_setpoint

                # Skip if setpoint change is too small (close to noise threshold)
                if abs(setpoint_change) < threshold * 0.5:
                     logger.debug(f"Skipping step at {step_time}: setpoint change {setpoint_change:.3f} too small.")
                     continue

                # --- Calculate Metrics for this step ---
                rise_time_ms = None
                overshoot_percent = None
                settling_time_ms = None

                # Rise Time (10% to 90% of step *response*)
                try:
                    response_change = final_setpoint - initial_actual # Use actual start for response range
                    if abs(response_change) > 1e-6:
                        target_10pct = initial_actual + 0.1 * response_change
                        target_90pct = initial_actual + 0.9 * response_change

                        if response_change >= 0: # Rising step response
                            time_10pct_indices = window_actual[window_actual >= target_10pct].index
                            time_90pct_indices = window_actual[window_actual >= target_90pct].index
                        else: # Falling step response
                            time_10pct_indices = window_actual[window_actual <= target_10pct].index
                            time_90pct_indices = window_actual[window_actual <= target_90pct].index

                        if not time_10pct_indices.empty and not time_90pct_indices.empty:
                            time_10 = time_10pct_indices[0]
                            time_90 = time_90pct_indices[0]
                            if time_90 >= time_10: # Ensure 90% time is after 10% time
                                rise_time_val = time_90 - time_10
                                # Convert to milliseconds
                                if is_time_index: rise_time_ms = rise_time_val.total_seconds() * 1000
                                elif is_numeric_index and time_unit == 'us': rise_time_ms = rise_time_val / 1000.0
                                elif is_numeric_index: rise_time_ms = rise_time_val * 1000.0 # Assume seconds
                except Exception as e_rise:
                    logger.debug(f"Rise time calculation error for step at {step_time}: {e_rise}")

                # Overshoot (relative to setpoint change)
                try:
                    if abs(setpoint_change) > 1e-6:
                        if setpoint_change > 0: # Rising step
                            peak_value = window_actual.max()
                            overshoot = peak_value - final_setpoint
                        else: # Falling step
                            peak_value = window_actual.min()
                            overshoot = final_setpoint - peak_value # How far it went below target

                        overshoot_percent = max(0.0, (overshoot / abs(setpoint_change)) * 100.0)
                except Exception as e_over:
                    logger.debug(f"Overshoot calculation error for step at {step_time}: {e_over}")

                # Settling Time (within +/- 5% band of final setpoint)
                try:
                    settling_band_half_width = max(0.05 * abs(setpoint_change), 1e-3) # 5% band or minimum value
                    lower_bound = final_setpoint - settling_band_half_width
                    upper_bound = final_setpoint + settling_band_half_width

                    # Find the last time the signal was *outside* the settling band within the window
                    outside_band_indices = window_actual[(window_actual < lower_bound) | (window_actual > upper_bound)].index

                    if outside_band_indices.empty:
                        # Settled immediately or within the first few points
                        settling_time_ms = 0.0
                    else:
                        last_unsettled_time = outside_band_indices[-1]
                        # Settling time is time from step start to last exit from band
                        settling_time_val = last_unsettled_time - step_time
                         # Convert to milliseconds
                        if is_time_index: settling_time_ms = settling_time_val.total_seconds() * 1000
                        elif is_numeric_index and time_unit == 'us': settling_time_ms = settling_time_val / 1000.0
                        elif is_numeric_index: settling_time_ms = settling_time_val * 1000.0 # Assume seconds
                        settling_time_ms = max(0.0, settling_time_ms) # Ensure non-negative

                except Exception as e_settle:
                    logger.debug(f"Settling time calculation error for step at {step_time}: {e_settle}")

                # Append calculated metrics for this step
                if rise_time_ms is not None or overshoot_percent is not None or settling_time_ms is not None:
                     metrics_list.append({
                         "rise_time_ms": rise_time_ms,
                         "overshoot_percent": overshoot_percent,
                         "settling_time_ms": settling_time_ms
                     })

            except Exception as e_outer:
                logger.error(f"Error processing step response at index {step_time}: {e_outer}", exc_info=True)
                continue # Skip to next step if error occurs

        # --- Calculate Median Metrics ---
        analyzed_steps_count = len(metrics_list)
        if analyzed_steps_count == 0:
            status = "Could not calculate metrics for any step" if filtered_step_indices else "No significant steps found"
            return {"status": status, "rise_time_ms": None, "overshoot_percent": None, "settling_time_ms": None, "analyzed_steps_count": 0}

        avg_metrics: Dict[str, Any] = {"analyzed_steps_count": analyzed_steps_count}
        for key in ["rise_time_ms", "overshoot_percent", "settling_time_ms"]:
            valid_values = [m[key] for m in metrics_list if m.get(key) is not None and np.isfinite(m[key])]
            if valid_values:
                 # Use nanmedian for robustness against outliers
                 avg_metrics[key] = round(np.nanmedian(valid_values), 2)
            else:
                 avg_metrics[key] = None

        avg_metrics["status"] = f"Median from {analyzed_steps_count} analyzed step(s)"
        logger.debug(f"Step response calculation result: {avg_metrics}")
        return avg_metrics


    def analyze_motors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes motor output levels, saturation, balance, and throttle usage.

        Args:
            df (pd.DataFrame): The prepared DataFrame.

        Returns:
            Dict[str, Any]: Dictionary containing motor analysis results under the 'motors' key.
        """
        logger.debug("Analyzing motors...")
        results: Dict[str, Any] = {}

        # Identify motor columns (expecting motor[0] to motor[N-1])
        motor_cols = sorted([col for col in df.columns if col.lower().startswith('motor[') and col[-2:-1].isdigit()])

        if not motor_cols:
            results["error_motors"] = "No motor data columns found (e.g., 'motor[0]')."
            logger.warning(results["error_motors"])
            return {"motors": results}

        try:
            motor_data = df[motor_cols].copy() # Work with a copy

            # Convert to numeric if needed (should be done in prepare_data, but double-check)
            for col in motor_cols:
                 if not pd.api.types.is_numeric_dtype(motor_data[col]):
                      motor_data[col] = pd.to_numeric(motor_data[col], errors='coerce')

            motor_data.dropna(how='all', inplace=True) # Drop rows where ALL motor values are NaN

            if motor_data.empty:
                results["error_motors"] = "Motor columns exist but contain only NaN or no valid data."
                logger.warning(results["error_motors"])
                return {"motors": results}

            # --- Detect Motor Output Range ---
            # Check across all motor columns simultaneously
            max_value = motor_data.max().max()
            min_value = motor_data.min().min()

            motor_range_min: float = 0.0
            motor_range_max: float = 0.0
            motor_value_type: str = "Unknown"

            # Use tolerances for range detection
            if 950 <= min_value and max_value <= 2050:
                motor_range_min, motor_range_max = 1000.0, 2000.0
                motor_value_type = "PWM/DShot (1000-2000)"
            elif -0.05 <= min_value and max_value <= 1.05:
                motor_range_min, motor_range_max = 0.0, 1.0
                motor_value_type = "Normalized (0-1)"
            elif min_value >= 0 and max_value <= 100: # Possible percentage?
                 motor_range_min, motor_range_max = 0.0, 100.0
                 motor_value_type = "Percentage? (0-100)"
            else:
                # Fallback: Use actual min/max, might indicate issues
                motor_range_min, motor_range_max = min_value, max_value
                motor_value_type = f"Custom/Unknown ({motor_range_min:.1f}-{motor_range_max:.1f})"
                logger.warning(f"Detected unusual motor range: {motor_value_type}")

            results["motor_value_type"] = motor_value_type
            results["motor_range_detected"] = [round(min_value, 2), round(max_value, 2)]
            results["motor_range_assumed"] = [motor_range_min, motor_range_max] # Assumed range for saturation

            # --- Motor Saturation ---
            # Use assumed max range for calculation, add tolerance (e.g., 1%)
            saturation_threshold = motor_range_max * 0.99
            saturated_points = (motor_data >= saturation_threshold).sum() # Sums boolean True counts per column
            total_points = len(motor_data) # Number of rows with valid motor data

            results["motor_saturation_pct_per_motor"] = (saturated_points / total_points * 100).round(2).to_dict()
            results["motor_saturation_pct_overall"] = round((saturated_points.sum() / motor_data.size) * 100, 2) # / total elements

            # --- Motor Averages and Balance ---
            avg_motors = motor_data.mean()
            results["motor_averages"] = avg_motors.round(2).to_dict()

            if len(avg_motors) > 1: # Need at least 2 motors for imbalance
                motor_imbalance_std = avg_motors.std()
                results["motor_imbalance_std_dev"] = round(motor_imbalance_std, 2)
                # Relative imbalance (Std Dev as % of Mean) - more meaningful
                overall_avg_motor = avg_motors.mean()
                if abs(overall_avg_motor) > 1e-6 * motor_range_max: # Avoid division by near-zero avg
                    results["motor_imbalance_pct_of_avg"] = round((motor_imbalance_std / overall_avg_motor) * 100, 1)
                else:
                     results["motor_imbalance_pct_of_avg"] = None # Cannot calculate relative imbalance

            # --- Throttle Analysis (if available) ---
            throttle_col = self._find_col(df, ['rcCommand[3]', 'rcCommands[3]', 'rcCommandThrottle'])
            if throttle_col and throttle_col in df.columns and pd.api.types.is_numeric_dtype(df[throttle_col]):
                throttle_data = df[throttle_col].dropna()
                if not throttle_data.empty:
                    results["avg_throttle_command"] = round(throttle_data.mean(), 1)
                    results["throttle_command_range"] = [round(throttle_data.min(), 1), round(throttle_data.max(), 1)]

                    # Normalize throttle based on its own range (e.g., 1000-2000 -> 0-1)
                    t_min, t_max = throttle_data.min(), throttle_data.max()
                    if t_max > t_min: # Avoid division by zero
                        normalized_throttle = ((throttle_data - t_min) / (t_max - t_min)).clip(0, 1)

                        # Throttle distribution (e.g., quartiles)
                        try:
                            throttle_bins = [0, 0.25, 0.5, 0.75, 1.0]
                            labels = [f"{int(throttle_bins[i]*100)}-{int(throttle_bins[i+1]*100)}%" for i in range(len(throttle_bins)-1)]
                            throttle_dist = pd.cut(normalized_throttle, bins=throttle_bins, labels=labels, include_lowest=True, right=True)
                            dist_counts = throttle_dist.value_counts(normalize=True, sort=False) # Keep order
                            results["throttle_distribution_pct"] = (dist_counts * 100).round(1).to_dict()
                        except Exception as e_dist:
                            logger.warning(f"Could not calculate throttle distribution: {e_dist}")
                            results["throttle_distribution_error"] = str(e_dist)
                else:
                     logger.debug("Throttle command column found but empty after dropping NaNs.")
            else:
                 logger.debug("Throttle command column (rcCommand[3]) not found or not numeric.")

            logger.debug("Motor analysis complete.")
            return {"motors": results}

        except Exception as e:
            logger.error(f"Error analyzing motor data: {e}", exc_info=True)
            results["error_motors"] = f"Unexpected error: {e}"
            return {"motors": results}

    def analyze_gyro_accel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes gyroscope and accelerometer data for basic characteristics,
        noise estimation, and potential issues like flatlines or anomalies.

        Args:
            df (pd.DataFrame): The prepared DataFrame.

        Returns:
            Dict[str, Any]: Dictionary containing gyro/accel analysis results under the 'gyro_accel' key.
        """
        logger.debug("Analyzing Gyro and Accelerometer data...")
        results: Dict[str, Any] = {}
        sensors_analyzed = []

        # --- Define potential columns ---
        sensor_potential = {
            'gyro_roll': ['gyroADC[0]', 'gyroRoll', 'gyro_roll'],
            'gyro_pitch': ['gyroADC[1]', 'gyroPitch', 'gyro_pitch'],
            'gyro_yaw': ['gyroADC[2]', 'gyroYaw', 'gyro_yaw'],
            'acc_roll': ['accSmooth[0]', 'accRoll', 'acc_roll'],
            'acc_pitch': ['accSmooth[1]', 'accPitch', 'acc_pitch'],
            'acc_z': ['accSmooth[2]', 'accZ', 'acc_z'] # Note: Z axis often named differently
        }

        # --- Analyze each sensor axis ---
        for sensor_axis_name, potential_names in sensor_potential.items():
            col = self._find_col(df, potential_names)
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                sensor_data = df[col].dropna()
                if not sensor_data.empty and len(sensor_data) > 1:
                    sensors_analyzed.append(sensor_axis_name)
                    axis_results: Dict[str, Any] = {}

                    # Basic stats
                    axis_results['mean'] = round(sensor_data.mean(), 3)
                    axis_results['std_dev'] = round(sensor_data.std(), 3)
                    axis_results['min'] = round(sensor_data.min(), 3)
                    axis_results['max'] = round(sensor_data.max(), 3)

                    # Noise/Variability (Std Dev of Differences)
                    diff_data = sensor_data.diff().dropna()
                    if not diff_data.empty:
                        axis_results['noise_metric_std_diff'] = round(diff_data.std(), 3)
                        axis_results['max_abs_diff'] = round(diff_data.abs().max(), 3)

                    # Anomaly Detection (Points > 3 * Std Dev from Mean)
                    if axis_results['std_dev'] > 1e-6: # Avoid issues if std dev is zero
                        anomaly_threshold = 3.0 * axis_results['std_dev']
                        anomalies = sensor_data[np.abs(sensor_data - axis_results['mean']) > anomaly_threshold]
                        anomaly_count = len(anomalies)
                        if anomaly_count > 0:
                            axis_results['anomaly_count_3std'] = anomaly_count
                            axis_results['anomaly_percentage_3std'] = round(anomaly_count / len(sensor_data) * 100, 2)

                    # Flatline Check (already done in diagnose_data_quality, but can add summary here)
                    # Retrieve from diagnostics if needed, or recalculate briefly
                    # Example: Check if std dev is extremely low
                    if axis_results['std_dev'] < 0.01: # Arbitrary low threshold
                         axis_results['potential_flatline_low_std'] = True

                    results[sensor_axis_name] = axis_results # Store results for this specific sensor axis
                else:
                     logger.debug(f"Sensor column '{col}' for {sensor_axis_name} is empty or has insufficient data after dropping NaNs.")
            else:
                 logger.debug(f"Sensor column not found or not numeric for {sensor_axis_name}.")


        if not sensors_analyzed:
            results['error'] = "No valid gyro or accelerometer data found for analysis."
            logger.warning(results['error'])
        else:
             logger.debug(f"Gyro/Accel analysis complete. Analyzed sensors: {sensors_analyzed}")

        return {"gyro_accel": results} # Return results wrapped in 'gyro_accel' key

# --- Continuing BetaflightLogAnalyzer Class ---

    def analyze_altitude_power(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes altitude (barometer) and power (voltage, current) data from the log.

        Args:
            df (pd.DataFrame): The prepared DataFrame.

        Returns:
            Dict[str, Any]: Dictionary containing altitude and power analysis results
                            under the 'alt_power' key.
        """
        logger.debug("Analyzing Altitude and Power...")
        results: Dict[str, Any] = {}
        analysis_performed = False

        # --- Find relevant columns ---
        alt_col = self._find_col(df, ['baroAlt', 'altitudeBaro', 'alt']) # Common names for baro altitude
        voltage_col = self._find_col(df, ['vbatLatest', 'voltage', 'vbat'])
        current_col = self._find_col(df, ['amperageLatest', 'current', 'amperage'])

        # --- Altitude Analysis ---
        if alt_col and alt_col in df.columns and pd.api.types.is_numeric_dtype(df[alt_col]):
            # Assume baroAlt is often in cm, check typical range or metadata if possible
            # For now, assume cm if name contains 'baro' and max value > 1000
            alt_data_raw = df[alt_col].dropna()
            if not alt_data_raw.empty:
                scale_factor = 1.0
                unit = "m" # Assume meters by default
                if 'baro' in alt_col.lower() and alt_data_raw.max() > 1000:
                     scale_factor = 100.0 # Convert cm to m
                     unit = "m (converted from cm)"
                     logger.debug(f"Assuming altitude column '{alt_col}' is in cm, converting to m.")

                alt_data = alt_data_raw / scale_factor
                results['altitude_unit'] = unit
                results['altitude_mean'] = round(alt_data.mean(), 2)
                results['altitude_max'] = round(alt_data.max(), 2)
                results['altitude_min'] = round(alt_data.min(), 2)
                results['altitude_std'] = round(alt_data.std(), 2)
                analysis_performed = True

                # Climb/Descend Rates (requires time index)
                if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.RangeIndex)) or np.issubdtype(df.index.dtype, np.number):
                    time_diff_sec = pd.Series(df.index).diff()
                    # Convert time diff to seconds
                    if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
                         time_diff_sec = time_diff_sec.dt.total_seconds()
                    elif np.issubdtype(df.index.dtype, np.number):
                         time_unit = self.column_descriptions.get('time', 'us') # Get unit if available
                         if 'us' in time_unit: time_diff_sec = time_diff_sec / 1_000_000.0

                    alt_diff = alt_data.diff()
                    vertical_speed = (alt_diff / time_diff_sec).dropna() # Speed in m/s
                    vertical_speed = vertical_speed[np.isfinite(vertical_speed)] # Remove inf/-inf

                    if not vertical_speed.empty:
                         results['max_climb_rate_mps'] = round(vertical_speed[vertical_speed > 0].max(), 2) if any(vertical_speed > 0) else 0.0
                         results['max_descend_rate_mps'] = round(vertical_speed[vertical_speed < 0].min(), 2) if any(vertical_speed < 0) else 0.0
                         results['avg_vertical_speed_mps'] = round(vertical_speed.mean(), 2)
                else:
                     logger.warning("Cannot calculate vertical speed without a valid time index.")

        # --- Power Analysis ---
        voltage_data = None
        current_data = None

        if voltage_col and voltage_col in df.columns and pd.api.types.is_numeric_dtype(df[voltage_col]):
            v_scale_factor = 1.0
            v_unit = "V"
            voltage_data_raw = df[voltage_col].dropna()
            # Assume vbatLatest is often cV (centivolts) if max > 50 (typical V range)
            if 'latest' in voltage_col.lower() and voltage_data_raw.max() > 50:
                 v_scale_factor = 100.0
                 v_unit = "V (converted from cV)"
                 logger.debug(f"Assuming voltage column '{voltage_col}' is in cV, converting to V.")

            voltage_data = voltage_data_raw / v_scale_factor
            if not voltage_data.empty:
                results['voltage_unit'] = v_unit
                results['voltage_mean'] = round(voltage_data.mean(), 2)
                results['voltage_min'] = round(voltage_data.min(), 2)
                results['voltage_max'] = round(voltage_data.max(), 2)
                voltage_sag = voltage_data.iloc[0] - voltage_data.min() if len(voltage_data)>0 else 0
                results['voltage_sag'] = round(voltage_sag, 2)
                analysis_performed = True

        if current_col and current_col in df.columns and pd.api.types.is_numeric_dtype(df[current_col]):
            c_scale_factor = 1.0
            c_unit = "A"
            current_data_raw = df[current_col].dropna()
            # Assume amperageLatest is often cA (centiamps) if max > 200 (typical A range)
            if 'latest' in current_col.lower() and current_data_raw.max() > 200:
                 c_scale_factor = 100.0
                 c_unit = "A (converted from cA)"
                 logger.debug(f"Assuming current column '{current_col}' is in cA, converting to A.")

            current_data = current_data_raw / c_scale_factor
            if not current_data.empty:
                results['current_unit'] = c_unit
                results['current_mean'] = round(current_data.mean(), 2)
                results['current_max'] = round(current_data.max(), 2)
                analysis_performed = True

        # Calculated Power (Watts)
        if voltage_data is not None and current_data is not None:
            # Align before multiplying
            aligned_v, aligned_c = voltage_data.align(current_data, join='inner')
            if not aligned_v.empty:
                power_data = aligned_v * aligned_c
                results['power_mean_w'] = round(power_data.mean(), 2)
                results['power_max_w'] = round(power_data.max(), 2)
                analysis_performed = True

        if not analysis_performed:
            results['error'] = "No valid altitude or power data found."
            logger.warning(results['error'])

        return {"alt_power": results}


    def analyze_rc_commands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes RC command characteristics and estimates pilot input style.

        Args:
            df (pd.DataFrame): The prepared DataFrame.

        Returns:
            Dict[str, Any]: Dictionary containing RC command analysis results
                            under the 'rc_commands' key.
        """
        logger.debug("Analyzing RC Commands...")
        results: Dict[str, Any] = {}
        analysis_performed = False

        rc_potential = {
            'roll': ['rcCommand[0]', 'rcCommands[0]', 'rcCommandRoll'],
            'pitch': ['rcCommand[1]', 'rcCommands[1]', 'rcCommandPitch'],
            'yaw': ['rcCommand[2]', 'rcCommands[2]', 'rcCommandYaw'],
            'throttle': ['rcCommand[3]', 'rcCommands[3]', 'rcCommandThrottle']
        }

        rc_cols = {axis: self._find_col(df, names) for axis, names in rc_potential.items()}
        axis_data: Dict[str, pd.Series] = {} # Store valid data series

        # --- Analyze each RC axis ---
        for axis, col in rc_cols.items():
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                rc_data = df[col].dropna()
                if not rc_data.empty and len(rc_data) > 1:
                    axis_data[axis] = rc_data # Store for later use
                    axis_results: Dict[str, Any] = {}
                    analysis_performed = True

                    # Basic statistics
                    axis_results['mean'] = round(rc_data.mean(), 1)
                    axis_results['std_dev'] = round(rc_data.std(), 1)
                    axis_results['min'] = round(rc_data.min(), 1)
                    axis_results['max'] = round(rc_data.max(), 1)
                    axis_results['range'] = round(axis_results['max'] - axis_results['min'], 1)

                    # Rate of change analysis
                    rc_diff = rc_data.diff().dropna()
                    if not rc_diff.empty:
                        axis_results['rate_of_change_mean_abs'] = round(rc_diff.abs().mean(), 2)
                        axis_results['rate_of_change_std'] = round(rc_diff.std(), 2)
                        axis_results['rate_of_change_max_abs'] = round(rc_diff.abs().max(), 2)

                    results[f'{axis}_rc'] = axis_results # Store results for this axis

        if not analysis_performed:
            results['error'] = "No valid RC command columns found."
            logger.warning(results['error'])
            return {"rc_commands": results}

        # --- Pilot Style Assessment (Roll & Pitch) ---
        roll_data = axis_data.get('roll')
        pitch_data = axis_data.get('pitch')

        if roll_data is not None and pitch_data is not None:
            # Combine roll and pitch rate of change (absolute values)
            roll_diff = roll_data.diff().abs().dropna()
            pitch_diff = pitch_data.diff().abs().dropna()
            combined_diff = pd.concat([roll_diff, pitch_diff])

            if not combined_diff.empty:
                # Smoothness assessment (lower std of rate of change indicates smoother)
                smoothness_std = combined_diff.std()
                results['pilot_smoothness_metric'] = round(smoothness_std, 2)
                if smoothness_std < 5: assessment = 'Very Smooth'
                elif smoothness_std < 15: assessment = 'Smooth'
                elif smoothness_std < 30: assessment = 'Moderate'
                else: assessment = 'Aggressive / Twitchy'
                results['pilot_smoothness_assessment'] = assessment

                # Aggression assessment (95th percentile of absolute rate of change)
                aggression_95 = combined_diff.quantile(0.95)
                results['pilot_aggression_metric'] = round(aggression_95, 2)
                if aggression_95 < 20: assessment = 'Low'
                elif aggression_95 < 50: assessment = 'Moderate'
                else: assessment = 'High'
                results['pilot_aggression_assessment'] = assessment

            # Center Focus (time spent near center - assumes center is near 0 or 1500)
            # Determine center based on range
            roll_center = 1500 if roll_data.min() > 1000 else 0
            pitch_center = 1500 if pitch_data.min() > 1000 else 0
            roll_range = roll_data.max() - roll_data.min()
            pitch_range = pitch_data.max() - pitch_data.min()
            center_threshold_pct = 0.05 # Within 5% of range from center

            roll_center_focus_pct = (np.abs(roll_data - roll_center) < (center_threshold_pct * roll_range)).mean() * 100 if roll_range > 0 else 100.0
            pitch_center_focus_pct = (np.abs(pitch_data - pitch_center) < (center_threshold_pct * pitch_range)).mean() * 100 if pitch_range > 0 else 100.0
            results['pilot_center_focus_pct_avg'] = round((roll_center_focus_pct + pitch_center_focus_pct) / 2.0, 1)

        logger.debug("RC command analysis complete.")
        return {"rc_commands": results}


    def analyze_rc_vs_gyro(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes RC command vs Gyro response latency using cross-correlation.

        Args:
            df (pd.DataFrame): The prepared DataFrame.
            metadata (Dict[str, Any]): Parsed metadata (used for sampling rate).

        Returns:
            Dict[str, Any]: Dictionary containing latency analysis results
                            under the 'rc_gyro_latency' key.
        """
        logger.debug("Analyzing RC Command vs Gyro Latency...")
        results: Dict[str, Any] = {}
        analysis_performed = False

        # --- Get Sampling Rate ---
        sampling_rate = metadata.get('analysis_info', {}).get('actual_sampling_rate_used')
        if sampling_rate is None or sampling_rate <= 0:
             # Attempt to estimate again if missing from metadata (e.g., if spectral analysis failed)
             time_unit = metadata.get('analysis_info', {}).get('time_unit', 'us')
             if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)) or np.issubdtype(df.index.dtype, np.number):
                  time_diffs = pd.Series(df.index).diff().dropna()
                  if len(time_diffs) > 0:
                       median_diff = time_diffs.median()
                       if median_diff > 1e-9:
                            if time_unit == 'us': sampling_rate = 1_000_000.0 / median_diff
                            elif isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)): sampling_rate = 1.0 / median_diff.total_seconds()
                            else: sampling_rate = 1.0 / median_diff # Assume seconds
             if sampling_rate is None or sampling_rate <= 0:
                  results['error'] = "Could not determine valid sampling rate for latency analysis."
                  logger.error(results['error'])
                  return {"rc_gyro_latency": results}
             else:
                  logger.warning(f"Sampling rate missing in metadata, estimated: {sampling_rate:.1f} Hz")
                  results['estimated_sampling_rate_hz'] = round(sampling_rate, 1)
        else:
             results['sampling_rate_used_hz'] = round(sampling_rate, 1)


        # --- Find RC and Gyro columns ---
        rc_potential = {
            'roll': ['rcCommand[0]', 'rcCommands[0]'], 'pitch': ['rcCommand[1]', 'rcCommands[1]'], 'yaw': ['rcCommand[2]', 'rcCommands[2]']
        }
        gyro_potential = {
            'roll': ['gyroADC[0]', 'gyroRoll'], 'pitch': ['gyroADC[1]', 'gyroPitch'], 'yaw': ['gyroADC[2]', 'gyroYaw']
        }
        rc_cols = {axis: self._find_col(df, names) for axis, names in rc_potential.items()}
        gyro_cols = {axis: self._find_col(df, names) for axis, names in gyro_potential.items()}

        # --- Analyze Latency for each axis ---
        for axis in ['roll', 'pitch', 'yaw']:
            rc_col = rc_cols.get(axis)
            gyro_col = gyro_cols.get(axis)

            if rc_col and gyro_col and rc_col in df.columns and gyro_col in df.columns:
                # Select numeric data only
                rc_data_raw = df[rc_col] if pd.api.types.is_numeric_dtype(df[rc_col]) else pd.to_numeric(df[rc_col], errors='coerce')
                gyro_data_raw = df[gyro_col] if pd.api.types.is_numeric_dtype(df[gyro_col]) else pd.to_numeric(df[gyro_col], errors='coerce')

                rc_data = rc_data_raw.dropna()
                gyro_data = gyro_data_raw.dropna()

                # Align data
                aligned_rc, aligned_gyro = rc_data.align(gyro_data, join='inner')

                if len(aligned_rc) > 100: # Need sufficient data for correlation
                    analysis_performed = True
                    try:
                        # Use rates of change for correlation (more sensitive to latency)
                        rc_rate = aligned_rc.diff().dropna()
                        gyro_rate = aligned_gyro.diff().dropna()

                        # Align rates again
                        aligned_rc_rate, aligned_gyro_rate = rc_rate.align(gyro_rate, join='inner')

                        if len(aligned_rc_rate) > 50:
                            # Normalize signals (important for cross-correlation amplitude)
                            rc_rate_norm = (aligned_rc_rate - aligned_rc_rate.mean()) / aligned_rc_rate.std()
                            gyro_rate_norm = (aligned_gyro_rate - aligned_gyro_rate.mean()) / aligned_gyro_rate.std()

                            # Handle potential NaNs or Infs after normalization
                            rc_rate_norm.fillna(0, inplace=True)
                            gyro_rate_norm.fillna(0, inplace=True)
                            rc_rate_norm.replace([np.inf, -np.inf], 0, inplace=True)
                            gyro_rate_norm.replace([np.inf, -np.inf], 0, inplace=True)

                            # Compute cross-correlation
                            correlation = signal.correlate(gyro_rate_norm.values, rc_rate_norm.values, mode='full', method='fft') # Correlate gyro vs rc
                            lags = signal.correlation_lags(len(gyro_rate_norm), len(rc_rate_norm), mode='full')

                            # Find lag corresponding to the peak correlation
                            # The peak indicates the shift of rc_rate needed to best match gyro_rate
                            peak_corr_index = np.argmax(correlation)
                            peak_lag_samples = lags[peak_corr_index]

                            # Convert lag (in samples) to milliseconds
                            # Positive lag means gyro lags behind RC
                            lag_ms = (peak_lag_samples / sampling_rate) * 1000.0
                            results[f'{axis}_lag_ms'] = round(lag_ms, 1)

                            # Also calculate simple Pearson correlation for reference
                            corr_coeff, _ = np.polynomial.polynomial.polyfit(aligned_rc_rate.values, aligned_gyro_rate.values, 1, full=False) # Using polyfit to get coefficient
                            # corr_coeff = aligned_rc_rate.corr(aligned_gyro_rate) # Pearson correlation
                            results[f'{axis}_rc_gyro_rate_correlation'] = round(corr_coeff, 3) # Use corr_coeff from polyfit or standard corr()

                        else:
                             logger.debug(f"Skipping latency for {axis}: Not enough data points after diff and align.")
                             results[f'{axis}_lag_ms'] = "N/A (Insufficient Data)"

                    except Exception as e:
                        logger.warning(f"Latency/correlation calculation failed for {axis} axis: {e}", exc_info=True)
                        results[f'{axis}_lag_ms'] = f"Error: {e}"
                else:
                     logger.debug(f"Skipping latency for {axis}: Not enough aligned data points ({len(aligned_rc)}).")

        if not analysis_performed:
             results['info'] = "Latency analysis requires sufficient RC and Gyro data."

        logger.debug("RC vs Gyro latency analysis complete.")
        return {"rc_gyro_latency": results}


    def perform_spectral_analysis(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs FFT spectral analysis on gyro data to identify noise frequencies.

        Args:
            df (pd.DataFrame): The prepared DataFrame.
            metadata (Dict[str, Any]): Parsed metadata (used for sampling rate).

        Returns:
            Dict[str, Any]: Dictionary containing spectral analysis results under the 'spectral' key.
                            Includes FFT frequencies, magnitudes, peak frequencies, and band energy.
        """
        logger.debug("Starting spectral analysis...")
        results: Dict[str, Any] = {'spectra': {}}
        analysis_info = metadata.get('analysis_info', {})

        # --- Determine Sampling Rate ---
        sampling_rate = analysis_info.get('actual_sampling_rate_used') # Check if already calculated
        if sampling_rate is None:
             # Try estimating from index if not already done
             time_unit = analysis_info.get('time_unit', 'us')
             if isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)) or np.issubdtype(df.index.dtype, np.number):
                  time_diffs = pd.Series(df.index).diff().dropna()
                  if len(time_diffs) > 0:
                       median_diff = time_diffs.median()
                       if median_diff > 1e-9:
                            if time_unit == 'us': sampling_rate = 1_000_000.0 / median_diff
                            elif isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)): sampling_rate = 1.0 / median_diff.total_seconds()
                            else: sampling_rate = 1.0 / median_diff # Assume seconds
             # Fallback to metadata looptime
             if sampling_rate is None or sampling_rate <= 0:
                  looptime_us = metadata.get('other_settings', {}).get('looptime')
                  if looptime_us and looptime_us > 0:
                       sampling_rate = 1_000_000.0 / looptime_us
                       results['estimated_sampling_rate_hz'] = f"{round(sampling_rate)} (from looptime)"
                       logger.info(f"Using sampling rate from metadata looptime: {sampling_rate:.1f} Hz")

        if sampling_rate is None or sampling_rate <= 0:
            results['error'] = "Could not determine valid sampling rate for spectral analysis."
            logger.error(results['error'])
            return {"spectral": results}

        # Store the rate used, whether estimated or confirmed
        results['actual_sampling_rate_used'] = sampling_rate
        analysis_info['actual_sampling_rate_used'] = sampling_rate # Update metadata too
        logger.info(f"Using sampling rate for FFT: {sampling_rate:.2f} Hz")

        # --- Identify Gyro Columns ---
        gyro_potential = {
            'gyro_roll': ['gyroADC[0]', 'gyroRoll'], 'gyro_pitch': ['gyroADC[1]', 'gyroPitch'], 'gyro_yaw': ['gyroADC[2]', 'gyroYaw']
        }
        gyro_cols = {name: self._find_col(df, p_names) for name, p_names in gyro_potential.items()}

        # --- Perform FFT for each axis ---
        for axis_name, col in gyro_cols.items():
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                signal_data = df[col].dropna()
                n = len(signal_data)
                if n < 50: # Need a reasonable number of points for FFT
                    results['spectra'][axis_name] = {"error": f"Not enough data points ({n} < 50)."}
                    logger.warning(f"Skipping FFT for {axis_name}: Insufficient data points ({n}).")
                    continue

                try:
                    # Apply Hann window to reduce spectral leakage
                    window = signal.windows.hann(n)
                    windowed_signal = signal_data.values * window

                    # Compute FFT
                    fft_complex = fft(windowed_signal)
                    # Compute frequencies (only positive half needed due to symmetry)
                    freqs = fftfreq(n, 1 / sampling_rate)[:n // 2]

                    # Compute magnitude (one-sided spectrum)
                    # Take absolute value, scale by 2/N (except DC and Nyquist)
                    fft_magnitude = np.abs(fft_complex[:n // 2]) * (2.0 / n)
                    fft_magnitude[0] = np.abs(fft_complex[0]) / n # DC component (0 Hz)
                    # If n is even, Nyquist frequency component is also scaled by 1/N
                    # if n % 2 == 0: fft_magnitude[-1] = np.abs(fft_complex[n // 2]) / n # Usually ignored

                    axis_fft_results: Dict[str, Any] = {}

                    # Find dominant peaks (ignoring low frequencies < 5 Hz)
                    min_freq_for_peaks = 5 # Hz
                    peak_indices, properties = signal.find_peaks(
                        fft_magnitude[freqs >= min_freq_for_peaks], # Search only above min freq
                        height=np.percentile(fft_magnitude[freqs >= min_freq_for_peaks], 75), # Height threshold (e.g., 75th percentile)
                        distance=int(5 / (sampling_rate / n)) if sampling_rate > 0 else 5 # Min distance (e.g., 5 Hz separation)
                    )
                    # Adjust indices back to original frequency array
                    peak_indices += np.sum(freqs < min_freq_for_peaks)

                    if peak_indices.size > 0:
                        peak_freqs = freqs[peak_indices]
                        peak_mags = fft_magnitude[peak_indices]
                        # Sort peaks by magnitude descending
                        sorted_indices = np.argsort(peak_mags)[::-1]
                        top_n = 5
                        axis_fft_results["dominant_peaks_hz_mag"] = list(zip(
                            np.round(peak_freqs[sorted_indices][:top_n], 1),
                            np.round(peak_mags[sorted_indices][:top_n], 4)
                        ))
                    else:
                        axis_fft_results["dominant_peaks_hz_mag"] = []

                    # Calculate average magnitude in frequency bands
                    bands = {
                        "prop_wash_(<30Hz)": (0, 30),
                        "low_motor_(30-80Hz)": (30, 80),
                        "mid_motor_(80-200Hz)": (80, 200),
                        "high_freq_(200-500Hz)": (200, 500),
                        "noise_floor_(>500Hz)": (500, sampling_rate / 2)
                    }
                    band_avg_magnitude = {}
                    if freqs.size > 0:
                        for name, (low, high) in bands.items():
                            high = min(high, sampling_rate / 2 - 1e-6) # Ensure high freq doesn't exceed Nyquist
                            if low >= high: continue
                            mask = (freqs >= low) & (freqs < high)
                            if np.any(mask):
                                band_avg_magnitude[name] = round(np.mean(fft_magnitude[mask]), 4)
                            else:
                                band_avg_magnitude[name] = 0.0
                    axis_fft_results["band_avg_magnitude"] = band_avg_magnitude

                    # Downsample data for storage/plotting if needed (e.g., > 1000 points)
                    max_points = 1000
                    if len(freqs) > max_points:
                        indices = np.linspace(0, len(freqs) - 1, max_points, dtype=int)
                        axis_fft_results["frequencies_hz"] = np.round(freqs[indices], 2).tolist()
                        axis_fft_results["magnitude"] = np.round(fft_magnitude[indices], 5).tolist()
                    elif len(freqs) > 0:
                        axis_fft_results["frequencies_hz"] = np.round(freqs, 2).tolist()
                        axis_fft_results["magnitude"] = np.round(fft_magnitude, 5).tolist()
                    else:
                        axis_fft_results["frequencies_hz"] = []
                        axis_fft_results["magnitude"] = []

                    results['spectra'][axis_name] = axis_fft_results

                except Exception as fft_err:
                    logger.error(f"Error during FFT processing for {axis_name} ({col}): {fft_err}", exc_info=True)
                    results['spectra'][axis_name] = {"error": f"FFT calculation failed: {fft_err}"}
            else:
                 logger.debug(f"Gyro column not found or not numeric for {axis_name}.")


        if not results['spectra']:
             results['error'] = "No valid gyro data found for spectral analysis."
             logger.warning(results['error'])

        return {"spectral": results}
    
# --- Continuing BetaflightLogAnalyzer Class ---
# << Previous methods (__init__ to perform_spectral_analysis) from Parts 2-5 go here >>

    # --- Interpretation, Recommendations, and History ---

    def identify_problem_patterns(self, analysis_results: Dict[str, Any], metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Identifies potential tuning and flight performance issues from analysis results.

        Uses thresholds and heuristics to flag common problems related to PID performance,
        motor output, and spectral noise. Includes basic PID optimization suggestions.

        Args:
            analysis_results (Dict[str, Any]): The combined dictionary of results from various analysis methods.
            metadata (Dict[str, Any]): Parsed metadata for the current log (used for current PID values).

        Returns:
            List[Tuple[str, Dict[str, Any]]]: A list of tuples, where each tuple contains:
                - problem_name (str): A short description of the identified problem.
                - details_dict (Dict[str, Any]): A dictionary containing:
                    - 'recommendation': Suggested action.
                    - 'explanation': Details about why the problem was flagged.
                    - 'severity': A numerical score (0-10) indicating importance.
                    - 'category': Type of issue (e.g., PID Tuning, Vibration, Mechanical).
                    - 'commands': Optional list of suggested CLI commands.
                    - 'simulated': Optional dictionary with simulation results (for PID).
        """
        logger.debug("Identifying problem patterns from analysis results...")
        problem_patterns: List[Tuple[str, Dict[str, Any]]] = []

        # --- PID Performance Issues ---
        pid_results = analysis_results.get('pid', {})
        if pid_results and not pid_results.get('error_overall'):
            for axis in ['roll', 'pitch', 'yaw']:
                mae_key = f'{axis}_tracking_error_mae'
                step_key = f'{axis}_step_response'
                mae = pid_results.get(mae_key)
                step_response = pid_results.get(step_key, {}) if isinstance(pid_results.get(step_key), dict) else {}
                overshoot = step_response.get('overshoot_percent')
                rise_time = step_response.get('rise_time_ms')
                settling_time = step_response.get('settling_time_ms')

                # 1. High Tracking Error (MAE)
                mae_threshold = 15.0 # degrees/sec - adjust as needed
                if mae is not None and mae > mae_threshold:
                    # Attempt optimization only if MAE is high
                    current_P, current_D = None, None
                    # Try finding current PIDs from metadata (handle different key formats)
                    pid_meta = metadata.get("pids", {})
                    p_keys = [f'{axis}pid', f'p_{axis}'] # e.g., rollpid, p_roll
                    d_keys = [f'{axis}pid', f'd_{axis}'] # e.g., rollpid, d_roll

                    # Extract P value
                    for pk in p_keys:
                         val = pid_meta.get(pk)
                         if isinstance(val, list) and len(val) > 0: current_P = val[0] # First element is P
                         elif isinstance(val, (int, float)): current_P = val
                         if current_P is not None: break
                    # Extract D value
                    for dk in d_keys:
                         val = pid_meta.get(dk)
                         if isinstance(val, list) and len(val) > 2: current_D = val[2] # Third element is D
                         elif isinstance(val, (int, float)) and dk.startswith('d_'): current_D = val # Specific d_axis key
                         if current_D is not None: break

                    rec_P, rec_D, opt_score = None, None, None
                    simulated_data = None
                    if current_P is not None and current_D is not None:
                         rec_P, rec_D, opt_score = optimize_pid_for_axis(current_P, current_D)
                         if rec_P is not None and rec_D is not None:
                              simulated_data = {"current": {"P": current_P, "D": current_D},
                                                "recommended": {"P": rec_P, "D": rec_D},
                                                "score": opt_score}

                    recommendation = f"Reduce {axis.capitalize()} P/D gains or check filters/vibrations."
                    explanation = f"High Mean Absolute Error ({mae:.1f}°/s) indicates poor setpoint tracking."
                    commands = [f"# High {axis} MAE - Review PID & Filters"]
                    if simulated_data:
                         recommendation = f"Adjust {axis.capitalize()} PID terms towards P={rec_P:.1f}, D={rec_D:.1f}."
                         explanation += f" Current P={current_P}, D={current_D}. Simulation suggests P={rec_P:.1f}, D={rec_D:.1f} may improve response (score: {opt_score:.2f}). Verify with test flights."
                         commands = [f"set p_{axis} = {rec_P:.1f}", f"set d_{axis} = {rec_D:.1f}", "save"]

                    problem_patterns.append((f"High {axis.capitalize()} Axis Tracking Error", {
                        "recommendation": recommendation,
                        "explanation": explanation,
                        "severity": 7.0 + min(2.0, (mae - mae_threshold) / mae_threshold), # Scale severity
                        "category": "PID Tuning",
                        "simulated": simulated_data,
                        "commands": commands
                    }))

                # 2. High Overshoot
                overshoot_threshold = 20.0 # percent
                if overshoot is not None and overshoot > overshoot_threshold:
                    problem_patterns.append((f"High {axis.capitalize()} Axis Overshoot", {
                        "recommendation": f"Reduce {axis.capitalize()} P gain, consider increasing D gain slightly.",
                        "explanation": f"Step response overshoot ({overshoot:.1f}%) exceeds target. Indicates P gain might be too high or D gain too low.",
                        "severity": 6.0 + min(3.0, (overshoot - overshoot_threshold) / 10.0),
                        "category": "PID Tuning",
                        "commands": [f"# High {axis} overshoot - Reduce P, check D"]
                    }))

                # 3. Slow Rise / Settling Time (relative to typical values)
                rise_time_threshold = 150 # ms
                settling_time_threshold = 250 # ms
                if rise_time is not None and rise_time > rise_time_threshold:
                     problem_patterns.append((f"Slow {axis.capitalize()} Axis Rise Time", {
                        "recommendation": f"Increase {axis.capitalize()} P gain, potentially adjust F gain.",
                        "explanation": f"Step response rise time ({rise_time:.0f}ms) is slow. Indicates P gain might be too low.",
                        "severity": 5.0 + min(3.0, (rise_time - rise_time_threshold) / 50.0),
                        "category": "PID Tuning",
                        "commands": [f"# Slow {axis} rise time - Increase P, check F"]
                     }))
                elif settling_time is not None and settling_time > settling_time_threshold: # Check only if rise time is okay
                     problem_patterns.append((f"Slow {axis.capitalize()} Axis Settling Time", {
                        "recommendation": f"Review {axis.capitalize()} I and D gains, check for oscillations.",
                        "explanation": f"Step response settling time ({settling_time:.0f}ms) is slow, even if rise time is acceptable. May indicate oscillations or incorrect I/D balance.",
                        "severity": 5.0 + min(3.0, (settling_time - settling_time_threshold) / 100.0),
                        "category": "PID Tuning",
                        "commands": [f"# Slow {axis} settling - Review I/D, check oscillations"]
                     }))


        # --- Motor Performance Issues ---
        motors_results = analysis_results.get('motors', {})
        if motors_results and not motors_results.get('error_motors'):
            # 1. High Motor Saturation
            saturation = motors_results.get('motor_saturation_pct_overall', 0)
            sat_threshold = 15.0 # percent overall
            if saturation is not None and saturation > sat_threshold:
                problem_patterns.append(("High Motor Saturation", {
                    "recommendation": "Reduce overall PID gains (especially P and D), check motor limits, or consider more powerful motors/props.",
                    "explanation": f"Overall motor saturation ({saturation:.1f}%) is high. Indicates motors are frequently hitting their maximum output, limiting control authority. Can be caused by overly aggressive PIDs, high D gain, or demanding flight maneuvers for the hardware.",
                    "severity": 7.0 + min(3.0, (saturation - sat_threshold) / 10.0),
                    "category": "Motor Performance / PID Tuning",
                    "commands": ["# High motor saturation - Reduce P/D gains, check motor limits/hardware"]
                }))

            # 2. Motor Imbalance
            imbalance = motors_results.get('motor_imbalance_pct_of_avg') # Std dev as % of mean
            imb_threshold = 15.0 # percent
            if imbalance is not None and imbalance > imb_threshold:
                 avg_outputs = motors_results.get('motor_averages', {})
                 explanation = f"Significant motor output imbalance detected (StdDev is {imbalance:.1f}% of average output)."
                 if avg_outputs:
                      explanation += f" Avg outputs: {avg_outputs}. Check motors/props corresponding to higher average outputs."

                 problem_patterns.append(("Motor Imbalance", {
                    "recommendation": "Check physical condition: motor mounting, prop balance/damage, frame resonance.",
                    "explanation": explanation,
                    "severity": 6.0 + min(2.0, (imbalance - imb_threshold) / 5.0),
                    "category": "Mechanical / Vibration",
                    "commands": ["# Motor imbalance - Check props, motors, frame"]
                }))

        # --- Spectral Noise Issues ---
        spectral_results = analysis_results.get('spectral', {}).get('spectra', {})
        if spectral_results:
            noise_bands = ["mid_motor_(80-200Hz)", "high_freq_(200-500Hz)", "noise_floor_(>500Hz)"]
            noise_threshold = 0.1 # Adjust based on typical clean logs
            for axis_name, spec_data in spectral_results.items():
                 if isinstance(spec_data, dict) and 'band_avg_magnitude' in spec_data:
                      band_mags = spec_data['band_avg_magnitude']
                      total_noise_mag = sum(band_mags.get(band, 0) for band in noise_bands if band_mags.get(band) is not None)

                      if total_noise_mag > noise_threshold * len(noise_bands): # Average magnitude threshold
                           dominant_noise_band = max((band for band in noise_bands if band_mags.get(band) is not None),
                                                      key=lambda b: band_mags.get(b, 0), default="N/A")
                           peaks = spec_data.get("dominant_peaks_hz_mag", [])
                           peak_info = f" Dominant peaks around: {peaks[0][0]} Hz" if peaks else ""

                           problem_patterns.append((f"High Noise Level ({axis_name.replace('_',' ').title()})", {
                               "recommendation": "Investigate vibration sources (props, motors, frame) or adjust filtering.",
                               "explanation": f"Elevated noise detected in {axis_name.replace('_',' ').title()} gyro signal. Average magnitude in noise bands ({total_noise_mag:.3f}) is high. Primarily in '{dominant_noise_band}'.{peak_info}",
                               "severity": 5.0 + min(4.0, total_noise_mag * 5), # Scale severity with noise magnitude
                               "category": "Vibration / Filtering",
                               "commands": ["# High gyro noise - Check hardware vibrations, review filter settings"]
                           }))


        # --- Data Quality Issues ---
        dq_results = analysis_results.get('data_quality', {})
        if dq_results and dq_results.get('quality_score', 1.0) < 0.6:
             problem_patterns.append(("Poor Data Quality", {
                 "recommendation": "Review data quality diagnostics. Results may be unreliable.",
                 "explanation": f"Log quality score is low ({dq_results.get('quality_score', 0.0):.2f}). Issues like data gaps, irregular sampling, or sensor anomalies were detected. {dq_results.get('summary', '')}",
                 "severity": 8.0, # High severity as it impacts all other analyses
                 "category": "Data Quality",
                 "commands": ["# Poor data quality - Check logging setup, SD card, sensor health"]
             }))


        logger.debug(f"Identified {len(problem_patterns)} problem patterns.")
        # Sort patterns by severity (descending)
        problem_patterns.sort(key=lambda p: p[1].get('severity', 0), reverse=True)
        return problem_patterns


    def generate_tuning_recommendations(self, analysis_results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates comprehensive tuning recommendations and an overall flight assessment.

        Args:
            analysis_results (Dict[str, Any]): Combined dictionary of analysis results.
            metadata (Dict[str, Any]): Parsed metadata for the current log.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'flight_assessment': Overall quality score, summary, strengths, weaknesses.
                - 'problem_patterns': List of identified issues with details (from identify_problem_patterns).
        """
        logger.debug("Generating tuning recommendations...")

        # 1. Identify specific problems
        problem_patterns = self.identify_problem_patterns(analysis_results, metadata)

        # 2. Calculate Overall Flight Quality Score
        # Start with data quality score, then deduct based on problem severity
        quality_score = analysis_results.get('data_quality', {}).get('quality_score', 1.0)
        max_penalty = 0.0
        penalty_factor_sum = 0.0

        for _, details in problem_patterns:
            # Don't double-penalize for data quality itself
            if details.get('category') == "Data Quality": continue
            severity = details.get('severity', 0) / 10.0 # Normalize severity 0-1
            # Apply penalty based on severity (e.g., quadratic for higher impact)
            penalty = severity**2 * 0.5 # Max penalty of 0.5 per severe issue
            max_penalty = max(max_penalty, penalty)
            penalty_factor_sum += penalty

        # Apply combined penalty, ensuring score doesn't drop excessively from many minor issues
        # Use a mix of max penalty and sum, capped
        final_penalty = min(0.8, max_penalty + penalty_factor_sum * 0.2)
        final_quality_score = max(0.0, quality_score - final_penalty) # Ensure score >= 0

        # 3. Generate Assessment Summary
        summary = "No major issues detected. Minor tuning might optimize performance."
        strengths = ["Generally stable flight characteristics."]
        weaknesses = []

        if final_quality_score >= 0.9:
            summary = "Excellent flight performance. Tune appears well-optimized."
            strengths.append("Good PID tracking and low noise.")
        elif final_quality_score >= 0.75:
            summary = "Good flight performance. Minor issues detected."
            strengths.append("Acceptable PID tracking.")
        elif final_quality_score >= 0.5:
            summary = "Fair flight performance. Tuning improvements recommended."
            weaknesses.append("Review PID tuning and potential vibration sources.")
        else:
            summary = "Poor flight performance. Significant tuning or hardware checks required."
            weaknesses = ["PID tuning likely needs significant adjustment.", "Check for mechanical issues or excessive vibration."]

        # Add specific weaknesses from identified problems
        for name, details in problem_patterns:
            cat = details.get('category', 'Issue')
            weaknesses.append(f"{cat}: {name}")

        # Remove default strength/weakness if specific ones were added
        if len(weaknesses) > 1 and "Review PID tuning..." in weaknesses: weaknesses.remove("Review PID tuning and potential vibration sources.")
        if len(problem_patterns) > 0 and "Generally stable..." in strengths: strengths.remove("Generally stable flight characteristics.")
        if final_quality_score < 0.8 and "Good PID tracking..." in strengths: strengths.remove("Good PID tracking and low noise.")


        flight_assessment = {
            "flight_quality_score": round(final_quality_score, 2),
            "summary": summary,
            "strengths": list(set(strengths)), # Unique strengths
            "weaknesses": list(set(weaknesses)) # Unique weaknesses
        }

        # 4. Compile final recommendations dictionary
        recommendations = {
            "flight_assessment": flight_assessment,
            "problem_patterns": problem_patterns # Include the detailed list
        }

        logger.info(f"Generated recommendations. Quality Score: {flight_assessment['flight_quality_score']:.2f}")
        return recommendations

    def _extract_performance_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Extracts a condensed set of key performance indicators (KPIs)
        from the full analysis results for storage in history and comparison.

        Args:
            analysis_results (Dict[str, Any]): The full analysis results dictionary.

        Returns:
            Dict[str, Optional[float]]: A flat dictionary of key performance metrics (numeric).
                                        Returns None for metrics that couldn't be calculated.
        """
        perf: Dict[str, Optional[float]] = {}
        pid_results = analysis_results.get('pid', {})
        motors_results = analysis_results.get('motors', {})
        spectral_results = analysis_results.get('spectral', {}).get('spectra', {})
        rc_gyro_results = analysis_results.get('rc_gyro_latency', {})

        # PID Tracking Errors (MAE)
        for axis in ['roll', 'pitch', 'yaw']:
            perf[f'mae_{axis}'] = pid_results.get(f'{axis}_tracking_error_mae')

        # Step Response Metrics (Median)
        for axis in ['roll', 'pitch']: # Usually focus on roll/pitch for step response
             step_res = pid_results.get(f"{axis}_step_response", {}) if isinstance(pid_results.get(f"{axis}_step_response"), dict) else {}
             perf[f'overshoot_{axis}_pct'] = step_res.get('overshoot_percent')
             perf[f'rise_time_{axis}_ms'] = step_res.get('rise_time_ms')
             perf[f'settling_time_{axis}_ms'] = step_res.get('settling_time_ms')

        # Motor Performance
        perf['motor_saturation_overall_pct'] = motors_results.get('motor_saturation_pct_overall')
        perf['motor_imbalance_std_dev'] = motors_results.get('motor_imbalance_std_dev')
        perf['motor_imbalance_pct_of_avg'] = motors_results.get('motor_imbalance_pct_of_avg')

        # Spectral Noise (Average magnitude in high-frequency bands)
        for axis_name, spec_data in spectral_results.items():
            if isinstance(spec_data, dict) and 'band_avg_magnitude' in spec_data:
                 bands = spec_data['band_avg_magnitude']
                 # Combine high freq bands for a single noise metric per axis
                 noise_mag = sum(bands.get(b, 0) for b in bands if b.startswith(('high_freq', 'noise_floor')))
                 perf[f'noise_level_{axis_name}'] = noise_mag

        # RC vs Gyro Latency
        for axis in ['roll', 'pitch', 'yaw']:
             perf[f'rc_gyro_lag_{axis}_ms'] = rc_gyro_results.get(f'{axis}_lag_ms')

        # Convert all values to float or None
        for key, value in perf.items():
            try:
                perf[key] = float(value) if value is not None else None
            except (ValueError, TypeError):
                perf[key] = None # Ensure None if conversion fails

        return perf


    def save_log_analysis(self, log_id: str, metadata: Dict[str, Any], analysis_results: Dict[str, Any], recommendations: Dict[str, Any]) -> bool:
        """
        Saves a summary of the log analysis results to the JSON history database.

        Stores key identifiers, crucial settings (PIDs, Filters, Rates),
        extracted performance metrics (KPIs), and the overall assessment/recommendations.
        Avoids storing the full raw analysis or DataFrame to keep the file size manageable.

        Args:
            log_id (str): A unique identifier for this log analysis session.
            metadata (Dict[str, Any]): Parsed metadata.
            analysis_results (Dict[str, Any]): Full analysis results dictionary.
            recommendations (Dict[str, Any]): Generated recommendations dictionary.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        logger.info(f"Saving analysis summary for log_id: {log_id}")
        try:
            # 1. Extract Key Information to Save
            performance_kpis = self._extract_performance_metrics(analysis_results)
            key_settings = {
                "pids": metadata.get("pids", {}),
                "filters": metadata.get("filters", {}),
                "rates": metadata.get("rates", {})
            }
            firmware_info = {
                 "bf_version": metadata.get("firmware", {}).get("bf_version"),
                 "target": metadata.get("firmware", {}).get("target"),
            }

            # Create the entry for the history file
            entry = {
                'log_id': log_id,
                'timestamp': datetime.now().isoformat(), # Analysis timestamp
                'filename': metadata.get('filename', 'Unknown'),
                'firmware': firmware_info,
                'settings_snapshot': key_settings, # Save snapshot of key settings
                'performance_kpis': performance_kpis, # Save extracted KPIs
                'flight_assessment': recommendations.get("flight_assessment", {}), # Save assessment summary
                # Optionally save a summary of problem patterns, not the full details
                'problem_summary': [p[0] for p in recommendations.get("problem_patterns", [])]
            }

            # 2. Read existing history
            try:
                 with self.history_db_path.open('r', encoding='utf-8') as f:
                      db_content = f.read()
                      if not db_content: # Handle empty file
                           history_db = {"log_history": []}
                      else:
                           history_db = json.loads(db_content)
                 # Ensure the expected structure exists
                 if "log_history" not in history_db or not isinstance(history_db["log_history"], list):
                      logger.warning(f"History DB {self.history_db_path} has invalid format. Reinitializing.")
                      history_db = {"log_history": []}
            except (FileNotFoundError, json.JSONDecodeError) as e:
                 logger.warning(f"Could not read or decode history DB {self.history_db_path}: {e}. Starting new history.")
                 history_db = {"log_history": []}

            history = history_db["log_history"]

            # 3. Add new entry and limit history size
            history.append(entry)
            max_history_entries = 100 # Keep the last 100 analyses
            if len(history) > max_history_entries:
                history = history[-max_history_entries:]
            history_db["log_history"] = history # Put updated list back

            # 4. Write back to file using make_serializable for safety
            with self.history_db_path.open('w', encoding='utf-8') as f:
                json.dump(history_db, f, indent=2, default=make_serializable) # Use helper

            logger.info(f"Successfully saved analysis summary for {log_id} to {self.history_db_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving log analysis summary for {log_id}: {e}", exc_info=True)
            return False


    def compare_logs(self, log_id1: str, log_id2: str) -> Dict[str, Any]:
        """
        Compares metadata and key performance metrics between two saved logs from history.

        Args:
            log_id1 (str): The unique ID of the first log to compare.
            log_id2 (str): The unique ID of the second log to compare.

        Returns:
            Dict[str, Any]: A dictionary containing the comparison results, including
                            setting changes, performance changes, and an overall assessment.
                            Returns {'error': ...} if comparison fails.
        """
        logger.info(f"Comparing logs: {log_id1} vs {log_id2}")
        try:
            # 1. Load History Data
            history = self.get_tuning_history()
            if not history: # get_tuning_history returns [] on error/not found
                 return {"error": f"Tuning history is empty or could not be loaded from {self.history_db_path}."}

            # 2. Find Log Entries
            log1_entry = next((entry for entry in history if entry.get('log_id') == log_id1), None)
            log2_entry = next((entry for entry in history if entry.get('log_id') == log_id2), None)

            if not log1_entry or not log2_entry:
                missing = [lid for lid, entry in zip([log_id1, log_id2], [log1_entry, log2_entry]) if not entry]
                msg = f"Log ID(s) not found in history: {', '.join(missing)}"
                logger.warning(msg)
                return {"error": msg}

            # 3. Determine Order (Older vs Newer) based on analysis timestamp
            ts1 = pd.to_datetime(log1_entry.get("timestamp"), errors='coerce')
            ts2 = pd.to_datetime(log2_entry.get("timestamp"), errors='coerce')

            if ts1 is not pd.NaT and ts2 is not pd.NaT and ts1 > ts2:
                log1_entry, log2_entry = log2_entry, log1_entry # Swap entries
                log_id1, log_id2 = log_id2, log_id1 # Swap IDs for reporting
                ts1, ts2 = ts2, ts1
                logger.debug("Swapped logs based on timestamp for comparison.")
            elif ts1 is pd.NaT or ts2 is pd.NaT:
                 logger.warning("Could not determine log order reliably due to missing/invalid timestamps.")

            comparison: Dict[str, Any] = {
                "log1_id": log_id1,
                "log2_id": log_id2,
                "log1_timestamp": log1_entry.get("timestamp"),
                "log2_timestamp": log2_entry.get("timestamp"),
                "log1_filename": log1_entry.get("filename"),
                "log2_filename": log2_entry.get("filename"),
                "setting_changes": {},
                "performance_changes": {},
                "overall_assessment": {}
            }

            # 4. Compare Settings Snapshot
            settings1 = log1_entry.get("settings_snapshot", {})
            settings2 = log2_entry.get("settings_snapshot", {})
            all_setting_keys = set(settings1.keys()) | set(settings2.keys())
            setting_changes_dict = {}
            for cat_key in all_setting_keys: # Iterate through 'pids', 'filters', 'rates'
                 cat1 = settings1.get(cat_key, {})
                 cat2 = settings2.get(cat_key, {})
                 all_keys_in_cat = set(cat1.keys()) | set(cat2.keys())
                 for key in all_keys_in_cat:
                      val1 = cat1.get(key)
                      val2 = cat2.get(key)
                      # Use make_serializable to handle potential type differences (list vs tuple)
                      if make_serializable(val1) != make_serializable(val2):
                           setting_changes_dict[f"{cat_key}.{key}"] = {"old": val1, "new": val2}
            comparison["setting_changes"] = setting_changes_dict

            # 5. Compare Performance KPIs
            perf1 = log1_entry.get("performance_kpis", {})
            perf2 = log2_entry.get("performance_kpis", {})
            all_perf_keys = set(perf1.keys()) | set(perf2.keys())
            perf_changes_dict = {}
            improvements = 0
            regressions = 0

            # Define which metrics are better when lower
            lower_is_better_metrics = [
                 'mae_', 'overshoot_', 'rise_time_', 'settling_time_', # PID/Step Response errors
                 'motor_saturation', 'motor_imbalance', # Motor issues
                 'noise_level', # Spectral noise
                 'rc_gyro_lag' # Latency
            ]

            for key in sorted(list(all_perf_keys)):
                 old_val = perf1.get(key)
                 new_val = perf2.get(key)
                 change_info: Dict[str, Any] = {"old": old_val, "new": new_val}

                 # Calculate change only if both values are valid numbers
                 if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)) and np.isfinite(old_val) and np.isfinite(new_val):
                     change = new_val - old_val
                     change_info["change"] = round(change, 3)
                     pct_change = None
                     if abs(old_val) > 1e-9: # Avoid division by zero
                         pct_change = (change / abs(old_val)) * 100
                     elif new_val != 0: # Handle change from zero
                         pct_change = float('inf') * np.sign(change)
                     change_info["percent_change"] = round(pct_change, 1) if pct_change is not None else None

                     # Determine improvement/regression (ignore tiny changes)
                     improvement_flag = None
                     rel_change_threshold = 0.01 # 1% relative change threshold
                     abs_change_threshold = 1e-3 # Small absolute change threshold
                     if abs(change) > abs_change_threshold and (abs(old_val) < 1e-9 or abs(change / old_val) > rel_change_threshold):
                          lower_is_better = any(k in key for k in lower_is_better_metrics)
                          if (lower_is_better and change < 0) or (not lower_is_better and change > 0):
                               improvement_flag = True
                               improvements += 1
                          else:
                               improvement_flag = False
                               regressions += 1
                     change_info["improvement"] = improvement_flag
                 else:
                      change_info["change"] = None
                      change_info["percent_change"] = None
                      change_info["improvement"] = None

                 perf_changes_dict[key] = change_info
            comparison["performance_changes"] = perf_changes_dict

            # 6. Overall Assessment Verdict
            verdict = "Mixed/Unchanged"
            if improvements > regressions: verdict = "Improved"
            elif regressions > improvements: verdict = "Regressed"
            comparison["overall_assessment"] = {"improvements": improvements, "regressions": regressions, "verdict": verdict}

            logger.info(f"Comparison complete: {log_id1} vs {log_id2}. Verdict: {verdict}")
            return comparison

        except Exception as e:
            logger.error(f"Error comparing logs {log_id1} and {log_id2}: {e}", exc_info=True)
            return {"error": f"Error during comparison: {e}"}
# --- Continuing BetaflightLogAnalyzer Class ---
    # << Previous methods (__init__ to compare_logs) from Parts 2-6 go here >>

    # --- Plotting Functions ---

    def _get_time_axis(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[pd.Index | np.ndarray, str]:
        """Helper to get the time axis data and appropriate label."""
        time_axis = df.index
        time_unit = metadata.get('analysis_info', {}).get('time_unit', 'us')
        xaxis_title = "Time"

        if isinstance(time_axis, (pd.TimedeltaIndex, pd.DatetimeIndex)):
            # Already a time-based index
            xaxis_title = "Time"
            time_axis_display = time_axis
        elif np.issubdtype(time_axis.dtype, np.number):
            # Convert numeric index (likely microseconds) to seconds
            if time_unit == 'us':
                time_axis_display = time_axis.to_numpy() / 1_000_000.0
                xaxis_title = "Time (s)"
            else: # Assume seconds if numeric but not 'us'
                time_axis_display = time_axis.to_numpy()
                xaxis_title = "Time (s)"
        else:
            # Fallback for unknown index types
            time_axis_display = time_axis.to_numpy() # Convert to numpy array
            xaxis_title = f"Index ({time_axis.dtype})"

        return time_axis_display, xaxis_title

    def plot_pid_tracking(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates PID tracking plot (Gyro vs Setpoint)."""
        logger.debug("Generating PID tracking plot...")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Roll Axis", "Pitch Axis", "Yaw Axis"))

        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        plot_generated = False

        gyro_potential = {'Roll': ['gyroADC[0]', 'gyroRoll'], 'Pitch': ['gyroADC[1]', 'gyroPitch'], 'Yaw': ['gyroADC[2]', 'gyroYaw']}
        setpoint_potential = {'Roll': ['setpoint[0]', 'setpointRoll'], 'Pitch': ['setpoint[1]', 'setpointPitch'], 'Yaw': ['setpoint[2]', 'setpointYaw']}

        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            gyro_col = self._find_col(df, gyro_potential[axis])
            setpoint_col = self._find_col(df, setpoint_potential[axis])

            if gyro_col:
                fig.add_trace(go.Scatter(x=time_axis_display, y=df[gyro_col],
                                         mode='lines', name=f'Gyro {axis}',
                                         line=dict(width=1)),
                              row=i+1, col=1)
                plot_generated = True
            if setpoint_col:
                fig.add_trace(go.Scatter(x=time_axis_display, y=df[setpoint_col],
                                         mode='lines', name=f'Setpoint {axis}',
                                         line=dict(dash='dash', width=1.5)),
                              row=i+1, col=1)
                plot_generated = True

            fig.update_yaxes(title_text="Rate (°/s)", row=i+1, col=1)

        fig.update_layout(title="PID Tracking: Gyro vs Setpoint", height=600, legend_title_text='Trace')
        fig.update_xaxes(title_text=xaxis_title, row=3, col=1)

        if not plot_generated:
             fig.add_annotation(text="No Gyro or Setpoint data found for plotting.", showarrow=False)
             logger.warning("PID Tracking plot: No relevant data found.")

        return fig

    def plot_motor_output(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates motor output plot."""
        logger.debug("Generating motor output plot...")
        motor_cols = sorted([col for col in df.columns if col.lower().startswith('motor[') and col[-2:-1].isdigit()])

        if not motor_cols:
            return go.Figure().update_layout(title="Motor Output (No Motor Data Found)")

        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        fig = go.Figure()
        for col in motor_cols:
            fig.add_trace(go.Scatter(x=time_axis_display, y=df[col], mode='lines',
                                     name=col, line=dict(width=1)))

        fig.update_layout(title="Motor Outputs", xaxis_title=xaxis_title,
                          yaxis_title="Motor Output Value", height=400, legend_title_text='Motor')
        return fig

    def plot_motor_saturation(self, motor_results: Dict[str, Any]) -> go.Figure:
        """Generates a bar chart showing per-motor saturation percentage."""
        logger.debug("Generating Motor Saturation plot...")
        saturation_per_motor = motor_results.get('motor_saturation_pct_per_motor', {})
        saturation_overall = motor_results.get('motor_saturation_pct_overall')

        if not saturation_per_motor or not isinstance(saturation_per_motor, dict) or not any(saturation_per_motor.values()):
            return go.Figure().update_layout(title="Per-Motor Saturation (No Saturation Data)")

        # Filter out potential non-numeric keys/values if errors occurred during analysis
        motors = [k for k, v in saturation_per_motor.items() if isinstance(v, (int, float))]
        sat_values = [v for v in saturation_per_motor.values() if isinstance(v, (int, float))]

        if not motors:
             return go.Figure().update_layout(title="Per-Motor Saturation (Invalid Data Format)")

        fig = go.Figure(data=[go.Bar(x=motors, y=sat_values, name='Saturation %')])
        title_text = "Per-Motor Saturation (%)"
        if saturation_overall is not None:
            title_text += f"<br><sup>Overall Saturation: {saturation_overall:.2f}%</sup>"

        fig.update_layout(title=title_text, xaxis_title="Motor",
                          yaxis_title="Saturation Percentage (%)", yaxis_range=[0, 100],
                          height=350, bargap=0.2)
        return fig

    def plot_throttle_distribution(self, motor_results: Dict[str, Any]) -> go.Figure:
        """Generates a pie chart showing throttle distribution."""
        logger.debug("Generating Throttle Distribution plot...")
        distribution = motor_results.get('throttle_distribution_pct', {})

        if not distribution or not isinstance(distribution, dict) or not any(distribution.values()):
            return go.Figure().update_layout(title="Throttle Distribution (No Throttle Data)")

        # Filter out potential non-numeric values
        labels = [k for k, v in distribution.items() if isinstance(v, (int, float))]
        values = [v for v in distribution.values() if isinstance(v, (int, float))]

        if not labels:
             return go.Figure().update_layout(title="Throttle Distribution (Invalid Data Format)")

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                     pull=[0.05 if l=='0-25%' else 0 for l in labels], # Pull lowest slice
                                     sort=False)]) # Keep order from analysis

        title_text = "Throttle Distribution (% Time)"
        avg_throttle = motor_results.get('avg_throttle_command')
        if avg_throttle is not None:
            title_text += f"<br><sup>Average Throttle Command: {avg_throttle:.1f}</sup>"

        fig.update_layout(title=title_text, height=350, showlegend=True, legend_title_text="Throttle Range")
        return fig

    def plot_motor_balance(self, motor_results: Dict[str, Any]) -> go.Figure:
        """Generates a bar chart showing average motor outputs and imbalance."""
        logger.debug("Generating Motor Balance plot...")
        averages = motor_results.get('motor_averages', {})
        imbalance_std = motor_results.get('motor_imbalance_std_dev')
        imbalance_pct = motor_results.get('motor_imbalance_pct_of_avg')

        if not averages or not isinstance(averages, dict) or not any(isinstance(v, (int,float)) for v in averages.values()):
            return go.Figure().update_layout(title="Motor Average Output (No Data)")

        motors = [k for k, v in averages.items() if isinstance(v, (int, float))]
        avg_values = [v for v in averages.values() if isinstance(v, (int, float))]

        if not motors:
             return go.Figure().update_layout(title="Motor Average Output (Invalid Data Format)")

        fig = go.Figure(data=[go.Bar(x=motors, y=avg_values, name='Average Output')])
        title_text = "Motor Average Output"
        subtitle_parts = []
        if imbalance_std is not None: subtitle_parts.append(f"StdDev: {imbalance_std:.2f}")
        if imbalance_pct is not None: subtitle_parts.append(f"({imbalance_pct:.1f}% of Avg)")
        if subtitle_parts: title_text += f"<br><sup>Imbalance: {' '.join(subtitle_parts)}</sup>"

        fig.update_layout(title=title_text, xaxis_title="Motor", yaxis_title="Average Output Value",
                          height=350, bargap=0.2)
        return fig

    def plot_spectral_analysis(self, spectral_results: Dict[str, Any]) -> go.Figure:
        """Generates spectral analysis FFT plot."""
        logger.debug("Generating spectral analysis plot...")
        spectra = spectral_results.get('spectra', {})
        sampling_rate = spectral_results.get('actual_sampling_rate_used', 1000) # Default if not found

        if not spectra or all(isinstance(v, dict) and 'error' in v for v in spectra.values()):
            return go.Figure().update_layout(title="Spectral Analysis (No Data or Errors)")

        valid_axes_data = {k.replace('_',' ').title(): v for k, v in spectra.items()
                           if isinstance(v, dict) and 'error' not in v and v.get("frequencies_hz")}
        num_axes = len(valid_axes_data)
        if num_axes == 0:
            return go.Figure().update_layout(title="Spectral Analysis (No Valid Spectrum Data)")

        fig = make_subplots(rows=num_axes, cols=1, shared_xaxes=True,
                            subplot_titles=list(valid_axes_data.keys()))
        max_freq_plotted = 0
        plot_row = 1
        for axis_title, axis_data in valid_axes_data.items():
            freqs = axis_data.get("frequencies_hz", [])
            mags = axis_data.get("magnitude", [])
            if freqs and mags:
                fig.add_trace(go.Scatter(x=freqs, y=mags, mode='lines', name=axis_title, line=dict(width=1)),
                              row=plot_row, col=1)
                fig.update_yaxes(title_text="Magnitude", type="log", row=plot_row, col=1) # Use log scale
                max_freq_plotted = max(max_freq_plotted, freqs[-1])
                plot_row += 1

        fig.update_layout(title="Gyro Spectral Analysis (FFT)", height=250 * num_axes, showlegend=False)
        # Set reasonable default x-axis range, e.g., up to 500 Hz or Nyquist/2
        max_plot_freq = min(500, sampling_rate / 2, max_freq_plotted if max_freq_plotted > 0 else 500)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max_plot_freq], row=num_axes, col=1)
        return fig

    def plot_throttle_freq_heatmap(self, spectral_results: Dict[str, Any]) -> go.Figure:
        """Generates throttle vs frequency heatmap (requires specific pre-computed data)."""
        logger.debug("Generating throttle vs frequency heatmap...")
        # NOTE: This plot requires the 'perform_spectral_analysis' to be modified
        #       to calculate FFTs segmented by throttle level. This is complex
        #       and not implemented in the current analysis pipeline.
        #       This function serves as a placeholder.
        heatmap_data = spectral_results.get("throttle_freq_heatmap") # Check for the specific key

        if not heatmap_data or not all(k in heatmap_data for k in ["magnitude_matrix", "frequency_bins_hz", "throttle_bins"]):
            logger.warning("Throttle vs Frequency heatmap data not found or incomplete in spectral results.")
            fig = go.Figure().update_layout(
                title="Throttle vs Frequency Heatmap (Data Not Available)",
                xaxis_title="Frequency (Hz)", yaxis_title="Normalized Throttle", height=400
            )
            fig.add_annotation(text="Heatmap requires FFT analysis segmented by throttle level (not implemented).", showarrow=False)
            return fig
        try:
            # If data *was* available, plot it:
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
            logger.error(f"Error creating heatmap from available data: {e}")
            return go.Figure().update_layout(title=f"Throttle vs Frequency Heatmap (Error: {e})")

    def plot_3d_flight(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates a 3D flight path plot using GPS or Altitude data."""
        logger.debug("Generating 3D flight tracking plot...")

        x, y, z = None, None, None
        color_data, colorbar_title = None, 'Altitude (m)'
        title = "3D Flight Path"
        scene = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data') # Use data aspect ratio

        # --- Try GPS Cartesian Coordinates ---
        gps_x_col = self._find_col(df, ['gpsCartesianCoords[0]'])
        gps_y_col = self._find_col(df, ['gpsCartesianCoords[1]'])
        gps_z_col = self._find_col(df, ['gpsCartesianCoords[2]'])

        if gps_x_col and gps_y_col and gps_z_col:
            x_raw = df[gps_x_col].dropna()
            y_raw = df[gps_y_col].dropna()
            z_raw = df[gps_z_col].dropna()
            common_index = x_raw.index.intersection(y_raw.index).intersection(z_raw.index)
            if not common_index.empty and len(common_index) > 1:
                x, y, z = x_raw.loc[common_index], y_raw.loc[common_index], z_raw.loc[common_index]
                title = "3D Flight Path (GPS Cartesian)"
                scene = dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode='data')
                color_data = z # Color by Z altitude
                logger.info("Using GPS Cartesian coordinates for 3D plot.")

        # --- Fallback: Altitude vs Time (Pseudo-3D) ---
        if x is None:
            alt_col = self._find_col(df, ['baroAlt', 'altitudeBaro', 'alt'])
            if alt_col:
                alt_data_raw = df[alt_col].dropna()
                if not alt_data_raw.empty:
                    scale_factor = 100.0 if 'baro' in alt_col.lower() and alt_data_raw.max() > 1000 else 1.0
                    z_alt = alt_data_raw / scale_factor # Use altitude for Z

                    time_axis_display, time_title = self._get_time_axis(df.loc[z_alt.index], metadata) # Get time for valid altitude points

                    x = time_axis_display # Time on X
                    y = pd.Series(0, index=z_alt.index) # Use 0 for Y
                    z = z_alt
                    title = "Altitude vs Time"
                    scene = dict(xaxis_title=time_title, yaxis_title="", zaxis_title="Altitude (m)", aspectmode='cube') # Cube aspect ratio
                    color_data = z # Color by altitude
                    logger.info("Using Altitude vs Time for pseudo-3D plot.")

        # --- Handle No Data ---
        if x is None or len(x) < 2:
            logger.warning("No suitable GPS or Altitude data found for 3D plot.")
            return go.Figure().update_layout(title="3D Flight Path (No Position Data)")

        # --- Create Plot ---
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines', # Use lines only for path clarity
            marker=dict(size=2, color=color_data, colorscale='Viridis', opacity=0.8,
                        colorbar=dict(title=colorbar_title) if color_data is not None else None),
            line=dict(width=3, color=color_data, colorscale='Viridis' if color_data is not None else None) # Color line by altitude too
        )])
        fig.update_layout(title=title, scene=scene, height=600, margin=dict(l=0, r=0, b=0, t=40))
        return fig

    def plot_3d_coords_over_time(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates plots of X, Y, Z GPS coordinates over time."""
        logger.debug("Generating 3D Coordinates over Time plot...")
        coord_cols = {
            'X': self._find_col(df, ['gpsCartesianCoords[0]']),
            'Y': self._find_col(df, ['gpsCartesianCoords[1]']),
            'Z': self._find_col(df, ['gpsCartesianCoords[2]'])
        }
        available_cols = {axis: col for axis, col in coord_cols.items() if col}
        if not available_cols:
            return go.Figure().update_layout(title="GPS Coordinates Over Time (No Data)")

        num_axes = len(available_cols)
        fig = make_subplots(rows=num_axes, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{axis} Coordinate" for axis in available_cols.keys()])
        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        current_row = 1
        for axis, col_name in available_cols.items():
            fig.add_trace(go.Scatter(x=time_axis_display, y=df[col_name], mode='lines',
                                     name=f'{axis} Coordinate', line=dict(width=1)),
                          row=current_row, col=1)
            fig.update_yaxes(title_text=f"{axis} (m)", row=current_row, col=1)
            current_row += 1

        fig.update_layout(title="GPS Cartesian Coordinates Over Time", height=200 * num_axes, showlegend=False)
        fig.update_xaxes(title_text=xaxis_title, row=num_axes, col=1)
        return fig

    def plot_gyro_analysis(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates plot showing gyro data characteristics and anomalies."""
        logger.debug("Generating Gyro Analysis plot...")
        gyro_potential = {'Roll': ['gyroADC[0]', 'gyroRoll'], 'Pitch': ['gyroADC[1]', 'gyroPitch'], 'Yaw': ['gyroADC[2]', 'gyroYaw']}
        gyro_cols = {axis: self._find_col(df, names) for axis, names in gyro_potential.items()}
        available_axes = [axis for axis, col in gyro_cols.items() if col]
        if not available_axes:
             return go.Figure().update_layout(title="Gyro Analysis (No Gyro Data)")

        fig = make_subplots(rows=len(available_axes), cols=1, shared_xaxes=True,
                            subplot_titles=[f"{axis} Axis Gyro" for axis in available_axes])
        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        plot_row = 1
        for axis in available_axes:
            col = gyro_cols[axis]
            gyro_data = df[col].dropna()
            if not gyro_data.empty:
                # Main gyro trace
                fig.add_trace(go.Scatter(x=time_axis_display, y=gyro_data, mode='lines',
                                         name=f'{axis} Gyro', line=dict(width=1)),
                              row=plot_row, col=1)
                # Highlight anomalies (> 3 std dev)
                mean = gyro_data.mean()
                std = gyro_data.std()
                if std > 1e-6:
                     anomaly_indices = gyro_data[np.abs(gyro_data - mean) > (3 * std)].index
                     if not anomaly_indices.empty:
                          # Get corresponding time values for anomalies
                          anomaly_times = time_axis_display[df.index.get_indexer(anomaly_indices)]
                          fig.add_trace(go.Scatter(x=anomaly_times, y=gyro_data.loc[anomaly_indices], mode='markers',
                                                  name=f'{axis} Anomalies', marker=dict(color='red', size=4)),
                                        row=plot_row, col=1)
                fig.update_yaxes(title_text="Rate (°/s)", row=plot_row, col=1)
                plot_row += 1

        fig.update_layout(title="Gyro Data Analysis with Anomalies (>3σ)", height=200 * len(available_axes), showlegend=True)
        fig.update_xaxes(title_text=xaxis_title, row=len(available_axes), col=1)
        return fig

    def plot_gyro_heatmap(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates a heatmap showing rolling standard deviation of gyro/RC data."""
        logger.debug("Generating Gyro/RC Variation Heatmap...")
        sensors_to_plot = {
            'Gyro Roll': self._find_col(df, ['gyroADC[0]', 'gyroRoll']),
            'Gyro Pitch': self._find_col(df, ['gyroADC[1]', 'gyroPitch']),
            'Gyro Yaw': self._find_col(df, ['gyroADC[2]', 'gyroYaw']),
            'RC Roll': self._find_col(df, ['rcCommand[0]', 'rcCommands[0]']),
            'RC Pitch': self._find_col(df, ['rcCommand[1]', 'rcCommands[1]']),
            'RC Yaw': self._find_col(df, ['rcCommand[2]', 'rcCommands[2]'])
        }
        available_sensors = {name: col for name, col in sensors_to_plot.items() if col}
        if not available_sensors:
            return go.Figure().update_layout(title="Gyro/RC Variation Heatmap (No Data)")

        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        heatmap_matrix = []
        labels = []
        # Estimate window size based on sampling rate (e.g., 50ms window)
        sampling_rate = metadata.get('analysis_info', {}).get('actual_sampling_rate_used', 1000)
        window_size = max(20, min(100, int(sampling_rate * 0.05)))

        for name, col in available_sensors.items():
            sensor_data = df[col].dropna()
            if len(sensor_data) > window_size:
                rolling_std = sensor_data.rolling(window=window_size, center=True, min_periods=window_size//2).std().fillna(0)
                # Downsample for plotting if too many points
                max_heatmap_cols = 1000
                if len(rolling_std) > max_heatmap_cols:
                     indices = np.linspace(0, len(rolling_std)-1, max_heatmap_cols, dtype=int)
                     heatmap_matrix.append(rolling_std.iloc[indices].values)
                     # Adjust time axis if downsampling
                     if len(labels)==0: # Only adjust time axis once
                          time_axis_display = time_axis_display[indices]
                else:
                     heatmap_matrix.append(rolling_std.values)
                labels.append(name)

        if not heatmap_matrix:
            return go.Figure().update_layout(title="Gyro/RC Variation Heatmap (Not Enough Data)")

        fig = go.Figure(data=go.Heatmap(
            z=np.array(heatmap_matrix),
            x=time_axis_display,
            y=labels,
            colorscale='Viridis',
            colorbar=dict(title='Rolling Std Dev')
        ))
        fig.update_layout(title='Gyro & RC Command Variation (Rolling Std Dev)',
                          xaxis_title=xaxis_title, yaxis_title='Sensor/Command', height=400)
        return fig

    def plot_power_altitude(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates Power and Altitude plot."""
        logger.debug("Generating Power & Altitude plot...")
        alt_col = self._find_col(df, ['baroAlt', 'altitudeBaro', 'alt'])
        voltage_col = self._find_col(df, ['vbatLatest', 'voltage', 'vbat'])
        current_col = self._find_col(df, ['amperageLatest', 'current', 'amperage'])
        plot_data = {} # Store data and units

        if alt_col:
            alt_data_raw = df[alt_col].dropna()
            if not alt_data_raw.empty:
                 scale = 100.0 if 'baro' in alt_col.lower() and alt_data_raw.max() > 1000 else 1.0
                 plot_data['Altitude (m)'] = alt_data_raw / scale
        if voltage_col:
             v_data_raw = df[voltage_col].dropna()
             if not v_data_raw.empty:
                  scale = 100.0 if 'latest' in voltage_col.lower() and v_data_raw.max() > 50 else 1.0
                  plot_data['Voltage (V)'] = v_data_raw / scale
        if current_col:
             c_data_raw = df[current_col].dropna()
             if not c_data_raw.empty:
                  scale = 100.0 if 'latest' in current_col.lower() and c_data_raw.max() > 200 else 1.0
                  plot_data['Current (A)'] = c_data_raw / scale

        if not plot_data:
            return go.Figure().update_layout(title="Power & Altitude (No Data Found)")

        plot_count = len(plot_data)
        fig = make_subplots(rows=plot_count, cols=1, shared_xaxes=True,
                            subplot_titles=list(plot_data.keys()))
        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        current_row = 1
        for name, data in plot_data.items():
            fig.add_trace(go.Scatter(x=time_axis_display, y=data, mode='lines', name=name, line=dict(width=1)),
                          row=current_row, col=1)
            fig.update_yaxes(title_text=name, row=current_row, col=1)
            current_row += 1

        fig.update_layout(title="Power & Altitude Data", height=200 * plot_count, showlegend=False)
        fig.update_xaxes(title_text=xaxis_title, row=plot_count, col=1)
        return fig

    def plot_rc_vs_gyro_response(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> go.Figure:
        """Generates plot comparing RC commands and gyro response."""
        logger.debug("Generating RC Command vs Gyro Response plot...")
        rc_potential = {'Roll': ['rcCommand[0]'], 'Pitch': ['rcCommand[1]'], 'Yaw': ['rcCommand[2]']}
        gyro_potential = {'Roll': ['gyroADC[0]', 'gyroRoll'], 'Pitch': ['gyroADC[1]', 'gyroPitch'], 'Yaw': ['gyroADC[2]', 'gyroYaw']}
        rc_cols = {axis: self._find_col(df, names) for axis, names in rc_potential.items()}
        gyro_cols = {axis: self._find_col(df, names) for axis, names in gyro_potential.items()}
        available_axes = [axis for axis in ['Roll', 'Pitch', 'Yaw'] if rc_cols.get(axis) and gyro_cols.get(axis)]

        if not available_axes:
             return go.Figure().update_layout(title="RC Command vs Gyro Response (Missing Data)")

        fig = make_subplots(rows=len(available_axes), cols=1, shared_xaxes=True,
                            subplot_titles=[f"{axis} Axis RC vs Gyro" for axis in available_axes])
        time_axis_display, xaxis_title = self._get_time_axis(df, metadata)
        plot_row = 1
        for axis in available_axes:
            rc_col = rc_cols[axis]
            gyro_col = gyro_cols[axis]
            # Plot RC Command
            fig.add_trace(go.Scatter(x=time_axis_display, y=df[rc_col], mode='lines',
                                     name=f'{axis} RC Cmd', line=dict(width=1, color='blue')),
                          row=plot_row, col=1)
            # Plot Gyro Response (potentially on secondary y-axis if scales differ significantly)
            fig.add_trace(go.Scatter(x=time_axis_display, y=df[gyro_col], mode='lines',
                                     name=f'{axis} Gyro', line=dict(width=1.5, color='red', dash='dot')),
                          row=plot_row, col=1)
            fig.update_yaxes(title_text="Value", row=plot_row, col=1) # Generic Y axis label
            plot_row += 1

        fig.update_layout(title="RC Command vs Gyro Response", height=250 * len(available_axes), showlegend=True)
        fig.update_xaxes(title_text=xaxis_title, row=len(available_axes), col=1)
        return fig

    # << Previous methods (__init__ to plot_rc_vs_gyro_response) from Parts 2-7 go here >>


# --- Inside the BetaflightLogAnalyzer class ---

    def full_log_analysis(self, file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Performs the complete analysis workflow for a single log file.

        Reads, parses, prepares, analyzes, and saves results for the given log file.
        This revised version proceeds with analysis even if data quality score is low,
        but includes quality diagnostics in the output.

        Args:
            file_path (Union[str, pathlib.Path]): The path to the log file.

        Returns:
            Dict[str, Any]: A dictionary containing the results:
                - 'log_id': Unique identifier for this analysis run.
                - 'metadata': Parsed metadata from the log header.
                - 'analysis_results': Dictionary containing results from all analysis steps.
                - 'recommendations': Dictionary with flight assessment and tuning suggestions.
                - 'data_quality': Results from the data quality diagnosis.
                - 'df': The prepared Pandas DataFrame used for analysis (or None if failed).
                - 'error': Error message if the analysis failed at any critical stage.
        """
        filepath = pathlib.Path(file_path)
        filename = filepath.name
        logger.info(f"--- Starting Full Analysis for {filename} ---")

        # Initialize return structure
        output: Dict[str, Any] = {
            "log_id": None,
            "metadata": {"filename": filename}, # Start with filename
            "analysis_results": {},
            "recommendations": {},
            "data_quality": {}, # Initialize data quality dict
            "df": None,
            "error": None
        }
        df_prepared: Optional[pd.DataFrame] = None # Keep track of the prepared df
        metadata: Dict[str, Any] = {"filename": filename} # Initialize metadata

        try:
            # === Stage 1: Reading and Parsing Header/Metadata ===
            logger.debug("Stage 1: Reading and Parsing Header/Metadata")
            lines = self._read_log_file(filepath)
            metadata_lines, header_line, data_start_index = self._find_header_and_data(lines)
            # Parse metadata - ensure filename is included/updated
            parsed_meta = self.parse_metadata(metadata_lines)
            metadata.update(parsed_meta) # Merge parsed metadata
            output["metadata"] = metadata # Update output metadata immediately

            # === Stage 2: Parsing and Preparing Data ===
            logger.debug("Stage 2: Parsing and Preparing Data")
            df_raw = self.parse_data(header_line, lines[data_start_index:])
            # Pass metadata to prepare_data so it can access/update analysis_info
            df_prepared = self.prepare_data(df_raw, metadata)
            output["df"] = df_prepared # Store prepared df
            # Metadata might have been updated by prepare_data (e.g., time_unit), so update again
            output["metadata"] = metadata

            # === Stage 3: Data Quality Check ===
            logger.debug("Stage 3: Data Quality Check")
            # Ensure diagnose_data_quality exists and call it
            if not hasattr(self, 'diagnose_data_quality'):
                raise AttributeError("'BetaflightLogAnalyzer' object has no attribute 'diagnose_data_quality'")
            data_quality = self.diagnose_data_quality(df_prepared, metadata) # Pass df and metadata
            output["data_quality"] = data_quality # Store quality results

            # --- MODIFIED Quality Check ---
            # Log a warning if quality is low, but DO NOT abort the analysis based on score alone.
            # Analysis will only abort now if a critical error occurs in subsequent steps.
            min_quality_threshold = 0.3 # Define threshold for warning
            current_quality_score = data_quality.get("quality_score", 0.0)
            if current_quality_score < min_quality_threshold:
                logger.warning(f"Data quality score ({current_quality_score:.2f}) is below threshold ({min_quality_threshold}), but proceeding with full analysis.")
            # --- End Modification ---

            # === Stage 4: Perform Detailed Analyses ===
            logger.debug("Stage 4: Performing Detailed Analyses")
            analysis_results: Dict[str, Any] = {}
            # Include data_quality results within analysis_results for easier access later
            analysis_results['data_quality'] = data_quality

            # Run each analysis method, passing df and metadata where needed
            # Use update to merge results, handling potential errors within each method
            try: analysis_results.update(self.analyze_pid_performance(df_prepared, metadata))
            except Exception as e: logger.error("Error during PID analysis", exc_info=True); analysis_results['pid'] = {'error': str(e)}

            try: analysis_results.update(self.analyze_motors(df_prepared))
            except Exception as e: logger.error("Error during Motor analysis", exc_info=True); analysis_results['motors'] = {'error': str(e)}

            try: analysis_results.update(self.perform_spectral_analysis(df_prepared, metadata=metadata))
            except Exception as e: logger.error("Error during Spectral analysis", exc_info=True); analysis_results['spectral'] = {'error': str(e)}

            try: analysis_results.update(self.analyze_gyro_accel(df_prepared)) # Updated to return wrapped dict
            except Exception as e: logger.error("Error during Gyro/Accel analysis", exc_info=True); analysis_results['gyro_accel'] = {'error': str(e)}

            try: analysis_results.update(self.analyze_rc_commands(df_prepared)) # Updated to return wrapped dict
            except Exception as e: logger.error("Error during RC Command analysis", exc_info=True); analysis_results['rc_commands'] = {'error': str(e)}

            try: analysis_results.update(self.analyze_altitude_power(df_prepared)) # Updated to return wrapped dict
            except Exception as e: logger.error("Error during Alt/Power analysis", exc_info=True); analysis_results['alt_power'] = {'error': str(e)}

            try: analysis_results.update(self.analyze_rc_vs_gyro(df_prepared, metadata)) # Updated to return wrapped dict
            except Exception as e: logger.error("Error during RC/Gyro Latency analysis", exc_info=True); analysis_results['rc_gyro_latency'] = {'error': str(e)}

            output["analysis_results"] = analysis_results # Store potentially partial results if errors occurred

            # === Stage 5: Generate Recommendations ===
            logger.debug("Stage 5: Generating Recommendations")
            # Ensure dependent methods exist before calling
            if not hasattr(self, 'generate_tuning_recommendations'):
                raise AttributeError("'BetaflightLogAnalyzer' object has no attribute 'generate_tuning_recommendations'")
            if not hasattr(self, 'identify_problem_patterns'):
                 raise AttributeError("'BetaflightLogAnalyzer' object has no attribute 'identify_problem_patterns'")

            try:
                recommendations = self.generate_tuning_recommendations(analysis_results, metadata)
                output["recommendations"] = recommendations
            except Exception as e:
                 logger.error("Error generating recommendations", exc_info=True)
                 output["recommendations"] = {'error': f"Failed to generate recommendations: {e}"}


            # === Stage 6: Save Results Summary ===
            logger.debug("Stage 6: Saving Analysis Summary")
            # Generate a unique ID for this analysis run
            timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
            # Sanitize filename for use in ID if needed
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            log_id = f"{timestamp_str}_{safe_filename}"
            output["log_id"] = log_id

            try:
                save_success = self.save_log_analysis(log_id, metadata, analysis_results, output["recommendations"]) # Pass generated recommendations
                if not save_success:
                    logger.warning(f"Failed to save analysis summary for log {log_id}.")
                    # Don't treat this as a fatal error for the overall analysis return
            except Exception as e:
                 logger.error(f"Error during save_log_analysis for log {log_id}", exc_info=True)
                 # Log the error but don't stop the function from returning results


            logger.info(f"--- Finished Full Analysis for {filename} ---")
            # No top-level error set, return the full results object
            output["error"] = None # Explicitly set error to None on success
            return output

        # --- Error Handling for Critical Stages (Parsing, Preparation) ---
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            logger.error(error_msg, exc_info=False) # No need for full traceback for FileNotFoundError
            output["error"] = error_msg
            # Keep metadata if available
            output["metadata"] = metadata if metadata else output["metadata"]
            return output
        except ValueError as e: # Catches header/data parsing errors, empty df errors, etc.
            error_msg = f"Data processing error: {e}"
            logger.error(error_msg, exc_info=True)
            output["error"] = error_msg
            # Include df and metadata if prepared before error
            output["df"] = df_prepared if df_prepared is not None else None
            output["metadata"] = metadata if metadata else output["metadata"]
            return output
        except IOError as e:
            error_msg = f"File reading error: {e}"
            logger.error(error_msg, exc_info=True)
            output["error"] = error_msg
            output["metadata"] = metadata if metadata else output["metadata"]
            return output
        except AttributeError as e: # Catch issues like missing methods during development
            error_msg = f"Internal code error (AttributeError): {e}. Check class implementation."
            logger.error(error_msg, exc_info=True)
            output["error"] = error_msg
            output["df"] = df_prepared if df_prepared is not None else None
            output["metadata"] = metadata if metadata else output["metadata"]
            return output
        except Exception as e: # Catch any other unexpected critical errors
            error_msg = f"An unexpected critical error occurred during analysis: {type(e).__name__} - {e}"
            logger.critical(f"Unexpected critical error during full_log_analysis for {filename}:", exc_info=True)
            output["error"] = error_msg
            # Return df and metadata if successfully prepared before the error
            output["df"] = df_prepared if df_prepared is not None else None
            output["metadata"] = metadata if metadata else output["metadata"]
            return output
    
# --- UI Display Helper Functions (with added error handling & debug) ---

def display_welcome():
    """Displays the initial welcome message."""
    st.markdown("## Welcome to the Advanced Betaflight Log Analyzer!")
    st.info(
        """
        👈 **Getting Started:**

        1.  Use the sidebar to **upload a single log file** or select one from a **local directory**.
        2.  Click **'Analyze Selected Log'**.
        3.  Explore the results in the tabs that appear.
        4.  Analyze multiple logs to view **'History'** trends or **'Compare'** two specific logs.
        """
    )
    st.markdown("---")
    st.markdown("Developed with Python, Pandas, Plotly, and Streamlit.")


def display_summary_tab(results: Dict[str, Any]):
    """Displays the content for the Summary & Recommendations tab."""
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    rec_data = results.get("recommendations", {})
    if not rec_data or not isinstance(rec_data, dict):
        st.warning("Recommendation data is missing or invalid.")
        if debug_enabled: st.json({"DEBUG_recommendations_data_received": make_serializable(rec_data)})
        return

    # --- Flight Assessment ---
    st.subheader("📊 Flight Assessment")
    assessment = rec_data.get("flight_assessment", {})
    if assessment and isinstance(assessment, dict):
        quality_score = assessment.get("flight_quality_score", assessment.get("flight_quality")) # Check both keys
        if quality_score is None: quality_score = 0.0
        if not isinstance(quality_score, (int, float)): quality_score = 0.0

        # Determine color based on score
        if quality_score >= 0.8: color = "green"
        elif quality_score >= 0.6: color = "orange"
        else: color = "red"
        st.markdown(f"**Overall Quality Score:** <span style='color:{color}; font-size: 1.2em; font-weight:bold;'>{quality_score:.2f} / 1.0</span>", unsafe_allow_html=True)
        st.progress(float(quality_score))
        st.markdown(f"**Summary:** {assessment.get('summary', 'N/A')}")

        col1, col2 = st.columns(2)
        strengths = assessment.get("strengths", [])
        weaknesses = assessment.get("weaknesses", [])
        col1.markdown("**👍 Strengths:**\n" + ("\n".join([f"- {s}" for s in strengths]) if strengths else "_None identified_"))
        col2.markdown("**👎 Weaknesses:**\n" + ("\n".join([f"- {w}" for w in weaknesses]) if weaknesses else "_None identified_"))
    else:
        st.warning("Flight assessment data missing or invalid.")
        logger.warning(f"Flight assessment data missing or invalid: {type(assessment)}")
        if debug_enabled: st.json({"DEBUG_flight_assessment_data_received": make_serializable(assessment)})


    st.divider()

    # --- Tuning Suggestions ---
    st.subheader("🔧 Tuning Suggestions & Diagnosis")
    problem_patterns_list = rec_data.get("problem_patterns", [])
    if problem_patterns_list and isinstance(problem_patterns_list, list):
        for i, pattern_tuple in enumerate(problem_patterns_list):
            if isinstance(pattern_tuple, (list, tuple)) and len(pattern_tuple) == 2:
                problem_name, details = pattern_tuple
                if isinstance(details, dict):
                    cat = f" ({details.get('category', '')})" if details.get('category') else ""
                    sev = details.get('severity', 0)
                    if sev >= 8.0: sev_color = "red"
                    elif sev >= 6.0: sev_color = "orange"
                    else: sev_color = "blue"
                    sev_text = f"<span style='color:{sev_color};'>**Severity: {sev:.1f}/10**</span>"
                    expander_label = details.get('recommendation', problem_name)
                    if len(expander_label) > 80: expander_label = expander_label[:77] + "..."

                    with st.expander(f"Suggestion {i+1}: {expander_label} - {sev_text}", expanded=(i < 2)):
                        st.markdown(f"**Issue:** {problem_name}")
                        st.markdown(f"**Category:** {details.get('category', 'N/A')}")
                        st.markdown(f"**Severity:** {sev:.1f}/10")
                        st.markdown(f"**Explanation:** {details.get('explanation', 'N/A')}")
                        commands = details.get('commands')
                        if commands and isinstance(commands, list):
                            st.markdown("**Suggested CLI Commands:**")
                            st.code("\n".join(commands), language="bash")
                        sim = details.get('simulated')
                        if sim:
                            st.markdown("**PID Simulation:**")
                            st.json(make_serializable(sim))
                else: st.warning(f"Invalid format for problem details at index {i}: {details}")
            else: st.warning(f"Invalid format for problem pattern tuple at index {i}: {pattern_tuple}")
        if debug_enabled:
             with st.expander("DEBUG: Raw Problem Patterns List"):
                  st.json(make_serializable(problem_patterns_list))
    else:
        st.info("No specific problems detected requiring tuning changes based on current thresholds.")

    st.divider()

    # --- Data Quality ---
    st.subheader("🩺 Data Quality Check")
    data_quality = results.get("data_quality", results.get("analysis_results", {}).get('data_quality', {}))
    if data_quality and isinstance(data_quality, dict):
        dq_score = data_quality.get('quality_score', 0.0)
        if not isinstance(dq_score, (int, float)): dq_score = 0.0
        st.metric("Data Quality Score", f"{dq_score:.2f} / 1.0")
        st.progress(float(dq_score))
        st.markdown(f"**Diagnosis:** {data_quality.get('summary', 'N/A')}")
        diagnosis_list = data_quality.get("diagnosis", [])
        if diagnosis_list and isinstance(diagnosis_list, list):
            with st.expander("Data Quality Issues Found"):
                for issue in diagnosis_list: st.warning(f"- {issue}")
        if debug_enabled:
             with st.expander("DEBUG: Raw Data Quality Dict"):
                  st.json(make_serializable(data_quality))
    else:
        st.warning("Data quality information not available.")
        logger.warning(f"Data quality info missing or invalid type: {type(data_quality)}")
        if debug_enabled: st.json({"DEBUG_data_quality_received": make_serializable(data_quality)})


def display_metadata_tab(results: Dict[str, Any]):
    """Displays the content for the Metadata tab."""
    st.subheader("📝 Log File Metadata")
    metadata = results.get("metadata", {})
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not metadata or not isinstance(metadata, dict):
        st.warning("Metadata not available or invalid.")
        if debug_enabled: st.json({"DEBUG_metadata_received": make_serializable(metadata)})
        return

    try:
        # ... (metadata display logic from previous version - seems okay) ...
        st.markdown(f"**Filename:** `{metadata.get('filename', 'N/A')}`")
        fw_info = metadata.get('firmware', {})
        if not isinstance(fw_info, dict): fw_info = {}
        an_info = metadata.get('analysis_info', {})
        if not isinstance(an_info, dict): an_info = {}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**BF Version:** {fw_info.get('betaflight_version', fw_info.get('bf_version', 'N/A'))}")
            st.markdown(f"**Target:** {fw_info.get('firmware_target', fw_info.get('target','N/A'))}")
            st.markdown(f"**Board:** {fw_info.get('board_name', 'N/A')}")
            st.markdown(f"**Craft Name:** {fw_info.get('craft_name', 'N/A')}")
        with col2:
            st.markdown(f"**Analyzed:** {an_info.get('analysis_timestamp', 'N/A')}")
            st.markdown(f"**Time Unit:** `{an_info.get('time_unit', 'N/A')}`")
            rate = fw_info.get('log_rate_hz')
            looptime = metadata.get('other_settings', {}).get('looptime')
            if rate: st.markdown(f"**Log Rate:** {rate} Hz")
            elif looptime: st.markdown(f"**Looptime:** {looptime} µs")
            debug_mode_fw = fw_info.get('debug_mode')
            debug_mode_other = metadata.get('other_settings', {}).get('debug_mode')
            debug_mode = debug_mode_fw if debug_mode_fw is not None else debug_mode_other
            if debug_mode is not None: st.markdown(f"**Debug Mode:** {debug_mode}")

        st.divider()
        st.markdown("**Configuration Details:**")
        for cat_key, cat_name in [('pids', 'PID Settings'), ('rates', 'Rate Settings'),
                                  ('filters', 'Filter Settings'), ('features', 'Features'),
                                  ('other_settings', 'Other Settings'), ('firmware', 'Firmware/Board Details'),
                                  ('hardware','Hardware Details'), ('raw_headers','Raw Header Lines')]:
            cat_data = metadata.get(cat_key)
            if cat_data:
                if isinstance(cat_data, (dict, list)) and not cat_data and cat_key != 'raw_headers': continue
                with st.expander(f"{cat_name} ({len(cat_data) if isinstance(cat_data, (list, dict)) else 1})"):
                    st.json(make_serializable(cat_data))
    except Exception as meta_e:
        st.error(f"Error displaying metadata: {meta_e}")
        logger.error("Error displaying metadata tab", exc_info=True)
    finally: # Add debug output even if display fails
        if debug_enabled:
             with st.expander("DEBUG: Raw Metadata Dict"):
                  st.json(make_serializable(metadata))


def display_pid_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    """Displays the content for the PID Performance tab."""
    st.subheader("📉 PID Tracking Performance")
    pid_results = results.get("analysis_results", {}).get('pid', {})
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    # Check more thoroughly for valid pid_results structure
    if not isinstance(pid_results, dict) or pid_results.get("error_gyro") or pid_results.get("error_overall"):
        error_msg = "No valid PID analysis data."
        if isinstance(pid_results, dict):
             error_msg = pid_results.get('error_gyro') or pid_results.get('error_overall') or 'No data or invalid format'
        st.warning(f"Could not display PID performance. Reason: {error_msg}")
        if isinstance(pid_results, dict):
             st.json(make_serializable(pid_results)) # Show details on error
             if debug_enabled:
                 with st.expander("DEBUG: Raw PID Analysis Dict (on error)"): st.json(make_serializable(pid_results))
        return

    # Plotting with error handling
    if isinstance(df_current, pd.DataFrame):
        st.markdown("**Gyro vs Setpoint**")
        try:
            # Ensure the plot function actually exists on the analyzer instance
            if not hasattr(analyzer, 'plot_pid_tracking'):
                 st.error("Plotting function 'plot_pid_tracking' is missing from the analyzer.")
                 raise NotImplementedError("plot_pid_tracking not implemented")

            fig_pid = analyzer.plot_pid_tracking(df_current, results['metadata'])
            st.plotly_chart(fig_pid, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error generating PID tracking plot: {e}")
            logger.error("Error in plot_pid_tracking call or display", exc_info=True)
            if debug_enabled: # Show relevant data for debugging plot errors
                 st.write("Debug Info: Data columns available for plotting:")
                 st.write(df_current.columns.tolist())
    else:
        st.warning("Processed DataFrame not available for PID plot.")

    # Metrics display (seems okay, keep as is)
    st.markdown("**Tracking Error Metrics (Mean Absolute Error):**")
    cols = st.columns(3)
    has_mae = False
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        mae = pid_results.get(f"{axis}_tracking_error_mae")
        cols[i].metric(f"{axis.capitalize()} MAE (°/s)", f"{mae:.2f}" if isinstance(mae, (int, float)) else "N/A")
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

    # Debug Output
    if debug_enabled:
         with st.expander("DEBUG: Raw PID Analysis Dict"):
              st.json(make_serializable(pid_results))


def display_motors_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    """Displays the content for the Motors tab."""
    st.subheader("⚙️ Motor Output Analysis")
    motor_results = results.get("analysis_results", {}).get('motors', {})
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not isinstance(motor_results, dict) or motor_results.get("error_motors"):
        error_msg = motor_results.get('error_motors', 'No data or invalid format') if isinstance(motor_results, dict) else 'Motor results invalid'
        st.warning(f"Could not display motor analysis. Error: {error_msg}")
        if isinstance(motor_results, dict):
            st.json(make_serializable(motor_results)) # Show details on error
            if debug_enabled:
                with st.expander("DEBUG: Raw Motor Analysis Dict (on error)"): st.json(make_serializable(motor_results))
        return

    # Plotting with error handling
    if isinstance(df_current, pd.DataFrame):
        st.markdown("#### Motor Output Over Time")
        try:
            if not hasattr(analyzer, 'plot_motor_output'): raise NotImplementedError("plot_motor_output missing")
            fig_motor = analyzer.plot_motor_output(df_current, results['metadata'])
            st.plotly_chart(fig_motor, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error generating Motor Output plot: {e}")
            logger.error("Error in plot_motor_output call", exc_info=True)
            if debug_enabled:
                motor_cols_present = [c for c in df_current.columns if c.lower().startswith('motor[')]
                st.write(f"Debug Info: Motor columns found: {motor_cols_present}")


        st.markdown("--- \n #### Motor Balance & Saturation")
        col1, col2 = st.columns(2)
        with col1:
            try:
                if not hasattr(analyzer, 'plot_motor_balance'): raise NotImplementedError("plot_motor_balance missing")
                fig_motor_balance = analyzer.plot_motor_balance(motor_results)
                st.plotly_chart(fig_motor_balance, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Error generating Motor Balance plot: {e}")
                logger.error("Error in plot_motor_balance call", exc_info=True)
        with col2:
            try:
                if not hasattr(analyzer, 'plot_motor_saturation'): raise NotImplementedError("plot_motor_saturation missing")
                fig_motor_sat = analyzer.plot_motor_saturation(motor_results)
                st.plotly_chart(fig_motor_sat, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Error generating Motor Saturation plot: {e}")
                logger.error("Error in plot_motor_saturation call", exc_info=True)

        st.markdown("--- \n #### Throttle Usage")
        try:
            if not hasattr(analyzer, 'plot_throttle_distribution'): raise NotImplementedError("plot_throttle_distribution missing")
            fig_throttle_dist = analyzer.plot_throttle_distribution(motor_results)
            if fig_throttle_dist.layout.title and "No Throttle Data" not in fig_throttle_dist.layout.title.text:
                st.plotly_chart(fig_throttle_dist, use_container_width=True)
            else:
                st.info("Throttle distribution data not available.")
        except Exception as e:
            st.error(f"⚠️ Error generating Throttle Distribution plot: {e}")
            logger.error("Error in plot_throttle_distribution call", exc_info=True)

    else:
        st.warning("Processed DataFrame not available for motor plots.")

    # Debug Output
    if debug_enabled:
         with st.expander("DEBUG: Raw Motor Analysis Dict"):
              st.json(make_serializable(motor_results))

# --- Implement the rest of the display_..._tab functions similarly ---
# (display_spectral_tab, display_gyro_accel_tab, display_rc_pilot_tab, etc.)
# Add try...except around plotting and add debug expanders for each.
# Make sure to check hasattr(analyzer, 'plot_function_name') before calling.

def display_spectral_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    st.subheader("🔊 Gyro Spectral Analysis (FFT)")
    spectral_results = results.get("analysis_results", {}).get('spectral', {})
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not isinstance(spectral_results, dict) or spectral_results.get("error"):
        error_msg = "No valid spectral analysis data."
        if isinstance(spectral_results, dict): error_msg = spectral_results.get('error', 'No data or invalid format')
        st.warning(f"Could not display spectral analysis. Reason: {error_msg}")
        if isinstance(spectral_results, dict):
             st.json(make_serializable(spectral_results))
             if debug_enabled:
                 with st.expander("DEBUG: Raw Spectral Analysis Dict (on error)"): st.json(make_serializable(spectral_results))
        return

    try:
        if not hasattr(analyzer, 'plot_spectral_analysis'): raise NotImplementedError("plot_spectral_analysis missing")
        fig_spec = analyzer.plot_spectral_analysis(spectral_results)
        st.plotly_chart(fig_spec, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error generating Spectral Analysis plot: {e}")
        logger.error("Error in plot_spectral_analysis call", exc_info=True)
        if debug_enabled: st.json({"DEBUG_spectral_input": make_serializable(spectral_results)})

    st.markdown("---")
    st.subheader("Throttle vs Frequency Heatmap")
    try:
        if not hasattr(analyzer, 'plot_throttle_freq_heatmap'): raise NotImplementedError("plot_throttle_freq_heatmap missing")
        fig_heatmap = analyzer.plot_throttle_freq_heatmap(spectral_results)
        if fig_heatmap.layout.title and ("Data Not Available" in fig_heatmap.layout.title.text or "Error" in fig_heatmap.layout.title.text):
             st.info("Throttle vs Frequency heatmap requires specific pre-calculation (currently not implemented).")
        elif fig_heatmap.data:
             st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
             st.info("Throttle vs Frequency heatmap data could not be generated.")
    except Exception as e:
        st.error(f"⚠️ Error generating Throttle Heatmap plot: {e}")
        logger.error("Error in plot_throttle_freq_heatmap call", exc_info=True)

    if debug_enabled:
         with st.expander("DEBUG: Raw Spectral Analysis Dict"):
              st.json(make_serializable(spectral_results))

def display_gyro_accel_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    st.subheader("⚖️ Gyro & Accelerometer Details")
    gyro_accel_results = results.get("analysis_results", {}).get('gyro_accel', {})
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    error_key = 'error_gyro' # Or 'error' if that's what your analysis func returns
    if not isinstance(gyro_accel_results, dict) or gyro_accel_results.get(error_key):
        error_msg = "No valid Gyro/Accel analysis data."
        if isinstance(gyro_accel_results, dict): error_msg = gyro_accel_results.get(error_key, 'No data or invalid format')
        st.warning(f"Could not display Gyro/Accel analysis. Reason: {error_msg}")
        if isinstance(gyro_accel_results, dict):
             st.json(make_serializable(gyro_accel_results))
             if debug_enabled:
                 with st.expander("DEBUG: Raw Gyro/Accel Dict (on error)"): st.json(make_serializable(gyro_accel_results))
        return

    if isinstance(df_current, pd.DataFrame):
        st.markdown("#### Gyro Data Over Time (with Anomalies)")
        try:
            if not hasattr(analyzer, 'plot_gyro_analysis'): raise NotImplementedError("plot_gyro_analysis missing")
            fig_gyro = analyzer.plot_gyro_analysis(df_current, results['metadata'])
            st.plotly_chart(fig_gyro, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error generating Gyro Analysis plot: {e}")
            logger.error("Error in plot_gyro_analysis call", exc_info=True)
            if debug_enabled: st.write("Debug Info: Gyro columns:", [c for c in df_current.columns if 'gyro' in c.lower()])
        # Optional: Add Accel plot here
    else:
        st.warning("Processed DataFrame not available for Gyro/Accel plots.")

    if debug_enabled:
        with st.expander("DEBUG: Raw Gyro/Accel Analysis Dict"):
            st.json(make_serializable(gyro_accel_results))

def display_rc_pilot_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    st.subheader("🎮 RC Command & Pilot Analysis")
    rc_results = results.get("analysis_results", {}).get('rc_commands', {})
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not isinstance(rc_results, dict) or rc_results.get("error"):
        error_msg = "No valid RC Command analysis data."
        if isinstance(rc_results, dict): error_msg = rc_results.get('error', 'No data or invalid format')
        st.warning(f"Could not display RC Command analysis. Reason: {error_msg}")
        if isinstance(rc_results, dict):
             st.json(make_serializable(rc_results))
             if debug_enabled:
                 with st.expander("DEBUG: Raw RC Command Dict (on error)"): st.json(make_serializable(rc_results))
        return

    st.subheader("Pilot Style Assessment")
    # ... (Metrics display seems okay) ...
    col1, col2, col3 = st.columns(3)
    smoothness = rc_results.get("pilot_smoothness_assessment", "N/A")
    aggression = rc_results.get("pilot_aggression_assessment", "N/A")
    center_focus = rc_results.get("pilot_center_focus_pct_avg") # Use the key from your analysis function
    col1.metric("Smoothness", smoothness, help="Based on standard deviation of stick rate of change.")
    col2.metric("Aggression", aggression, help="Based on 95th percentile of stick rate of change.")
    col3.metric("Center Focus (%)", f"{center_focus:.1f}%" if center_focus is not None else "N/A", help="Average percentage of time Roll/Pitch sticks are near center.")

    if isinstance(df_current, pd.DataFrame):
        try:
            st.markdown("---")
            st.markdown("#### RC Commands Over Time")
            # Reuse plotting logic from previous snippet - ensure _get_time_axis and _find_col exist
            fig_rc_time = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Roll Cmd", "Pitch Cmd", "Yaw Cmd", "Throttle Cmd"))
            time_axis_display, xaxis_title = analyzer._get_time_axis(df_current, results['metadata'])
            rc_cols_plot = {'Roll': ['rcCommand[0]','rcCommands[0]','rcCommandRoll'], 'Pitch': ['rcCommand[1]','rcCommands[1]','rcCommandPitch'], 'Yaw': ['rcCommand[2]','rcCommands[2]','rcCommandYaw'], 'Throttle': ['rcCommand[3]','rcCommands[3]','rcCommandThrottle']}
            plot_made = False
            for i, (axis, names) in enumerate(rc_cols_plot.items(), 1):
                col = analyzer._find_col(df_current, names)
                if col:
                    fig_rc_time.add_trace(go.Scatter(x=time_axis_display, y=df_current[col], mode='lines', name=f'{axis} Cmd', line=dict(width=1)), row=i, col=1)
                    plot_made = True
                fig_rc_time.update_yaxes(title_text="Command", row=i, col=1)
            if plot_made:
                fig_rc_time.update_layout(height=600, showlegend=False)
                fig_rc_time.update_xaxes(title_text=xaxis_title, row=4, col=1)
                st.plotly_chart(fig_rc_time, use_container_width=True)
            else: st.info("No RC Command data found to plot.")
        except Exception as e:
            st.error(f"⚠️ Error plotting RC commands: {e}")
            logger.error("Error plotting RC Commands", exc_info=True)
            if debug_enabled: st.write("Debug Info: RC columns:", [c for c in df_current.columns if 'rccommand' in c.lower()])

    else:
        st.warning("Processed DataFrame not available for RC Command plots.")

    if debug_enabled:
        with st.expander("DEBUG: Raw RC Command Analysis Dict"):
            st.json(make_serializable(rc_results))

def display_power_alt_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    st.subheader("⚡ Power & Altitude Analysis")
    alt_power_results = results.get("analysis_results", {}).get('alt_power', {})
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not isinstance(alt_power_results, dict) or alt_power_results.get('error'):
        # ... (error display) ...
        return
    if not any(k not in ['error'] for k in alt_power_results.keys()):
         st.info("No Power or Altitude data columns (e.g., vbatLatest, baroAlt) found.")
         return

    if isinstance(df_current, pd.DataFrame):
        st.markdown("#### Data Over Time")
        try:
            if not hasattr(analyzer, 'plot_power_altitude'): raise NotImplementedError("plot_power_altitude missing")
            fig_power_alt = analyzer.plot_power_altitude(df_current, results['metadata'])
            st.plotly_chart(fig_power_alt, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error generating Power/Altitude plot: {e}")
            logger.error("Error in plot_power_altitude call", exc_info=True)
            if debug_enabled: st.write("Debug Info: Power/Alt columns:", [c for c in df_current.columns if any(k in c.lower() for k in ['vbat','amperage','alt'])])

    else:
        st.warning("Processed DataFrame not available for Power/Altitude plot.")

    if debug_enabled:
        with st.expander("DEBUG: Raw Power/Altitude Analysis Dict"):
            st.json(make_serializable(alt_power_results))

def display_rc_latency_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    st.subheader("🛰️ RC Command vs Gyro Response Analysis")
    latency_results = results.get("analysis_results", {}).get('rc_gyro_latency', {})
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not isinstance(latency_results, dict) or latency_results.get("error"):
        # ... (error display) ...
         return

    st.markdown("#### Estimated Latency (RC to Gyro)")
    # ... (metrics display seems okay) ...
    cols = st.columns(3); has_latency_data = False
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        lag = latency_results.get(f"{axis}_lag_ms"); display_lag = f"{lag:.1f}" if isinstance(lag, (int, float)) else "N/A"
        if isinstance(lag, (int, float)): has_latency_data = True
        cols[i].metric(f"{axis.capitalize()} Lag (ms)", display_lag)
    if has_latency_data: st.caption("_Note: Estimated latency based on peak cross-correlation...")
    else: st.info("Latency could not be calculated...")

    st.markdown("#### RC/Gyro Rate Correlation")
    # ... (correlation display seems okay) ...
    cols_corr = st.columns(3); has_corr_data = False
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        corr = latency_results.get(f"{axis}_rc_gyro_rate_correlation"); display_corr = f"{corr:.3f}" if isinstance(corr, (int, float)) else "N/A"
        if isinstance(corr, (int, float)): has_corr_data = True
        cols_corr[i].metric(f"{axis.capitalize()} Correlation", display_corr)
    if not has_corr_data: st.info("Correlation could not be calculated.")

    if isinstance(df_current, pd.DataFrame):
        st.markdown("--- \n #### Visual Comparison")
        try:
            if not hasattr(analyzer, 'plot_rc_vs_gyro_response'): raise NotImplementedError("plot_rc_vs_gyro_response missing")
            fig_rc_gyro = analyzer.plot_rc_vs_gyro_response(df_current, results['metadata'])
            st.plotly_chart(fig_rc_gyro, use_container_width=True)
        except Exception as e:
             st.error(f"⚠️ Error plotting RC vs Gyro: {e}")
             logger.error("Error plotting RC vs Gyro", exc_info=True)
             if debug_enabled: st.write("Debug Info: RC/Gyro columns:", [c for c in df_current.columns if any(k in c.lower() for k in ['rccommand','gyro'])])

    else:
        st.warning("Processed DataFrame not available for RC vs Gyro plot.")

    if debug_enabled:
        with st.expander("DEBUG: Raw Latency/Correlation Data"):
            st.json(make_serializable(latency_results))

def display_3d_flight_tab(results: Dict[str, Any], analyzer: BetaflightLogAnalyzer):
    st.subheader("🛩️ 3D Flight Path & Coordinates")
    df_current = results.get("df")
    debug_enabled = st.session_state.get("debug_mode_checkbox", False)

    if not isinstance(df_current, pd.DataFrame):
        st.warning("Processed DataFrame not available for 3D plots.")
        return

    try:
        st.markdown("#### Flight Path (3D View)")
        if not hasattr(analyzer, 'plot_3d_flight'): raise NotImplementedError("plot_3d_flight missing")
        fig_3d_path = analyzer.plot_3d_flight(df_current, results['metadata'])
        if fig_3d_path.layout.title and "No Position Data" not in fig_3d_path.layout.title.text:
            st.plotly_chart(fig_3d_path, use_container_width=True)
        else:
            st.info("No suitable GPS Cartesian or Altitude data found for 3D path visualization.")
    except Exception as e:
        st.error(f"⚠️ Error generating 3D Flight Path plot: {e}")
        logger.error("Error generating 3D path plot", exc_info=True)
        if debug_enabled: st.write("Debug Info: Position columns:", [c for c in df_current.columns if any(k in c.lower() for k in ['gps','alt'])])


    try:
        st.markdown("---")
        st.markdown("#### Coordinates vs. Time (GPS Cartesian)")
        if not hasattr(analyzer, 'plot_3d_coords_over_time'): raise NotImplementedError("plot_3d_coords_over_time missing")
        fig_3d_time = analyzer.plot_3d_coords_over_time(df_current, results['metadata'])
        if fig_3d_time.layout.title and "No GPS Cartesian Data" not in fig_3d_time.layout.title.text and "No Data" not in fig_3d_time.layout.title.text :
            st.plotly_chart(fig_3d_time, use_container_width=True)
        else:
            st.info("GPS Cartesian coordinate data (gpsCartesianCoords[0/1/2]) not found for time-series plot.")
    except Exception as e:
        st.error(f"⚠️ Error displaying 3D Coords plot: {e}")
        logger.error("Error generating 3D coords plot", exc_info=True)
        if debug_enabled: st.write("Debug Info: GPS Cartesian columns:", [c for c in df_current.columns if 'gpscartesian' in c.lower()])

    if debug_enabled:
         # No specific analysis dict, just note that it uses the main DataFrame
         st.write("Debug Info: 3D plots use GPS/Altitude columns from the main DataFrame.")


def display_raw_data_tab(results: Dict[str, Any]):
    # ... (same as before, seems robust enough) ...
    df_current = results.get("df")
    # ... (rest of display and download button logic) ...


def display_history_tab(analyzer: BetaflightLogAnalyzer):
    # ... (same as before, seems reasonably robust) ...
    st.header("📊 Tuning History Trends")
    # ... (rest of history display logic) ...


def display_comparison_view(comparison_results: Dict[str, Any]):
    # ... (same as before, already made robust) ...
    st.header("📊 Log Comparison Results")
    # ... (rest of comparison display logic) ...


# --- Main App Logic (with corrected error message display) ---
def main():
    """Main function to run the Streamlit application."""
    # --- Session State Init ---
    if "current_file_path" not in st.session_state: st.session_state.current_file_path = None
    if "current_file_name" not in st.session_state: st.session_state.current_file_name = None
    if "analysis_output" not in st.session_state: st.session_state.analysis_output = None
    if "comparison_results" not in st.session_state: st.session_state.comparison_results = None
    if "batch_results" not in st.session_state: st.session_state.batch_results = []
    if "last_analyzed_file" not in st.session_state: st.session_state.last_analyzed_file = None
    if "selected_log_ids" not in st.session_state: st.session_state.selected_log_ids = []

    # --- Instantiate Analyzer ---
    @st.cache_resource
    def get_analyzer():
        try: script_dir = pathlib.Path(__file__).parent
        except NameError: script_dir = pathlib.Path.cwd()
        data_dir = script_dir / "bf_analyzer_data"
        logger.info(f"Using data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        # Pass base_dir to init if needed: BetaflightLogAnalyzer(base_dir=data_dir)
        return BetaflightLogAnalyzer()
    analyzer = get_analyzer()

    # --- Sidebar Logic ---
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/wiki/betaflight/betaflight/images/betaflight_logo_outline_blue.png", width=100)
        st.header("📁 Log File Selection")
        upload_method = st.radio("Select files from:", ["Upload Single File", "Local Directory"], key="upload_method_radio", index=0)
        log_files = []
        # Get path from state first, might be updated by widgets below
        selected_file_path_str = st.session_state.get("current_file_path")

        if upload_method == "Upload Single File":
            uploaded_file = st.file_uploader("Upload a Betaflight log", type=["bbl", "bfl", "csv", "log", "txt"], key="file_uploader_single")
            if uploaded_file is not None:
                temp_dir = pathlib.Path(tempfile.gettempdir()) / "streamlit_uploads_bf"
                temp_dir.mkdir(parents=True, exist_ok=True)
                selected_file_path = temp_dir / uploaded_file.name
                current_selected_path_str = str(selected_file_path)
                try:
                    with open(selected_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    if current_selected_path_str != st.session_state.get('current_file_path'):
                        st.success(f"Uploaded: {uploaded_file.name}")
                        st.session_state.current_file_path = current_selected_path_str
                        st.session_state.current_file_name = uploaded_file.name
                        st.session_state.analysis_output = None; st.session_state.comparison_results = None
                        st.session_state.batch_results = []; st.session_state.last_analyzed_file = None
                        logger.debug(f"New file uploaded: {current_selected_path_str}")
                        st.rerun()
                    selected_file_path_str = current_selected_path_str # Reflect selection
                except Exception as e: st.error(f"Error saving upload: {e}"); logger.error(f"Error saving {uploaded_file.name}", exc_info=True)
            # else: # If no file is uploaded currently, ensure state path is cleared if it was from an upload
                 # if selected_file_path_str and temp_dir in pathlib.Path(selected_file_path_str).parents:
                 #      selected_file_path_str = None # Reset path if previous was upload and now none selected

        elif upload_method == "Local Directory":
            log_dir_str = st.text_input("Log directory path:", value=".", key="log_dir_input")
            log_dir = pathlib.Path(log_dir_str)
            if log_dir.is_dir():
                try:
                    log_files_found = list(log_dir.glob('*.bbl')) + list(log_dir.glob('*.bfl')) + list(log_dir.glob('*.csv')) + list(log_dir.glob('*.log')) + list(log_dir.glob('*.txt'))
                    log_files_found.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    if log_files_found:
                        file_name_options = [f.name for f in log_files_found]
                        current_file_name_state = st.session_state.get("current_file_name")
                        current_index = file_name_options.index(current_file_name_state) if current_file_name_state in file_name_options else None
                        selected_file_name = st.selectbox("Select a log file:", file_name_options, key="log_selector_local", index=current_index, placeholder="Choose a log...")
                        if selected_file_name:
                            new_selected_path = log_dir / selected_file_name
                            new_selected_path_str = str(new_selected_path)
                            selected_file_path_str = new_selected_path_str
                            if st.session_state.get('current_file_path') != new_selected_path_str:
                                st.session_state.current_file_path = new_selected_path_str
                                st.session_state.current_file_name = selected_file_name
                                st.session_state.analysis_output = None; st.session_state.comparison_results = None
                                st.session_state.batch_results = []; st.session_state.last_analyzed_file = None
                                logger.debug(f"New file selected from dir: {new_selected_path_str}")
                                st.rerun()
                        log_files = [str(f) for f in log_files_found]
                    else: st.info("No log files found.")
                except Exception as e: st.error(f"Error accessing dir '{log_dir}': {e}"); logger.error(f"Error accessing dir {log_dir}", exc_info=True)
            else: st.warning("Invalid directory.")
        # Update state path before button check
        st.session_state.current_file_path = selected_file_path_str

        st.divider()
        st.header("⚙️ Analysis Actions")
        analyze_button_disabled = st.session_state.get("current_file_path") is None
        debug_mode = st.checkbox("Enable debug mode", value=False, key="debug_mode_checkbox")

        if st.button("Analyze Selected Log", disabled=analyze_button_disabled, key="analyze_single_button", use_container_width=True):
             current_path = st.session_state.get("current_file_path")
             if current_path:
                 logger.info(f"Analyze button clicked for: {current_path}")
                 current_output = st.session_state.get("analysis_output")
                 is_already_analyzed = (current_path == st.session_state.get("last_analyzed_file") and current_output and isinstance(current_output, dict) and "error" not in current_output)
                 if is_already_analyzed: st.info("Log already analyzed.")
                 else:
                     with st.spinner(f"Analyzing {st.session_state.get('current_file_name', 'log')}..."):
                         analysis_output = analyzer.full_log_analysis(current_path)
                         st.session_state.analysis_output = analysis_output
                         st.session_state.comparison_results = None
                         st.session_state.batch_results = []
                     if analysis_output and isinstance(analysis_output, dict) and analysis_output.get("error") is None:
                         st.session_state.last_analyzed_file = current_path
                         st.success("Analysis Complete!")
                     else:
                         st.session_state.last_analyzed_file = None
                         # Error is displayed in main area logic now
                         logger.error(f"Analysis failed for {current_path}: {analysis_output.get('error', 'Unknown')}")
                     st.rerun()
             else: st.warning("Select/upload log first.")

        # Batch Analysis Section
        if upload_method == "Local Directory" and log_files:
             st.subheader("Batch Analysis")
             max_batch = min(50, len(log_files)); default_batch = min(5, max_batch)
             num_batch = st.number_input(f"Logs for batch (max {max_batch}):", min_value=1, max_value=max_batch, value=default_batch, key="num_batch_input")
             if st.button(f"Analyze {num_batch} Recent Logs", key="analyze_batch_button"):
                  st.session_state.batch_results = []; st.session_state.analysis_output = None; st.session_state.comparison_results = None
                  recent_logs_paths = log_files[:num_batch]
                  progress_bar = st.progress(0, text="Starting batch..."); status_text = st.empty()
                  batch_success_count = 0; logger.info(f"Starting batch analysis: {num_batch} logs.")
                  for i, file_path in enumerate(recent_logs_paths):
                      file_name = os.path.basename(file_path); status_text.text(f"Analyzing {i+1}/{num_batch}: {file_name}...")
                      try:
                          result = analyzer.full_log_analysis(file_path); st.session_state.batch_results.append(result)
                          if isinstance(result, dict) and "error" not in result: batch_success_count += 1
                          elif isinstance(result, dict): logger.warning(f"Batch failure: {file_name}: {result.get('error')}")
                      except Exception as batch_e:
                          logger.error(f"Exception in batch analysis: {file_name}", exc_info=True)
                          st.session_state.batch_results.append({"error": f"Exception: {batch_e}", "metadata": {"filename": file_name}})
                      finally: progress_bar.progress((i+1)/len(recent_logs_paths), text=f"Analyzed {i+1}/{num_batch}")
                  status_text.text(f"Batch complete. Success: {batch_success_count}/{len(recent_logs_paths)}"); st.success("Batch Finished!")
                  logger.info(f"Batch finished. Success: {batch_success_count}/{len(recent_logs_paths)}"); st.rerun()

        # History & Comparison Section
        st.divider()
        st.header("📊 History & Comparison")
        # ... (History/Comparison logic from previous version - seems okay) ...
        # ... Needs keys: log_compare_select, compare_logs_button ...
        try:
            history = []
            if analyzer.tuning_history_path.exists():
                history_load_result = analyzer.get_tuning_history()
                if isinstance(history_load_result, list): history = history_load_result
                elif isinstance(history_load_result, dict) and 'error' in history_load_result: st.warning(f"History load error: {history_load_result['error']}")
                else: st.warning("History file invalid format.")
            if history:
                 log_options_map = {}
                 for entry in reversed(history):
                    log_id = entry.get('log_id'); timestamp_str = entry.get('timestamp', ''); filename = entry.get('filename', 'Unknown Log')
                    display_str = f"[{timestamp_str[:16]}] {filename}"
                    if log_id: log_options_map[display_str] = log_id
                 st.info(f"Found {len(history)} historical analyses.")
                 selected_log_display = st.multiselect("Select 2 logs to compare:", list(log_options_map.keys()), max_selections=2, key="log_compare_select")
                 st.session_state.selected_log_ids = [log_options_map[disp] for disp in selected_log_display if disp in log_options_map]
                 compare_button_disabled = len(st.session_state.selected_log_ids) != 2
                 if st.button("Compare Selected Logs", disabled=compare_button_disabled, key="compare_logs_button", use_container_width=True):
                    if len(st.session_state.selected_log_ids) == 2:
                         log_id1, log_id2 = st.session_state.selected_log_ids; logger.info(f"Compare clicked: {log_id1} vs {log_id2}")
                         with st.spinner("Comparing logs..."):
                            st.session_state.comparison_results = analyzer.compare_logs(log_id1, log_id2)
                            st.session_state.analysis_output = None; st.session_state.batch_results = []
                         if st.session_state.comparison_results and isinstance(st.session_state.comparison_results, dict) and "error" not in st.session_state.comparison_results:
                            st.success("Comparison Complete!")
                         else:
                            error_msg = st.session_state.comparison_results.get('error', 'Unknown error') if isinstance(st.session_state.comparison_results, dict) else "Comparison failed."
                            st.error(f"Comparison Failed: {error_msg}") # Show error in main area
                            logger.error(f"Comparison failed: {log_id1} vs {log_id2}: {error_msg}")
                         st.rerun()
            else: # History list is empty or loading failed
                 if not analyzer.tuning_history_path.exists(): st.info("Tuning history file not found.")
                 elif isinstance(history, list) and not history: st.info("No historical analyses found.")
        except Exception as e: st.error(f"Error setting up history/comparison: {e}"); logger.error("Error loading history/comparison setup", exc_info=True)

        st.divider()
        # Set logger level based on debug checkbox state
        if st.session_state.get("debug_mode_checkbox", False): logger.setLevel(logging.DEBUG)
        else: logger.setLevel(logging.INFO)


    # --- Main Content Area Logic ---
    st.title("🚀 Advanced Betaflight Log Analyzer")

    # Determine current view based on session state
    current_view = "welcome" # Default view
    if st.session_state.get("comparison_results"): current_view = "comparison"
    elif st.session_state.get("analysis_output"):
         output = st.session_state.analysis_output
         if isinstance(output, dict):
             if output.get("error") is not None: current_view = "single_analysis_error"
             elif "analysis_results" in output: current_view = "single_analysis"
             else: current_view = "welcome" # Fallback for unexpected structure
         else: current_view = "welcome"; st.session_state.analysis_output = None # Clear invalid state
    elif st.session_state.get("batch_results"): current_view = "batch_summary"

    # --- Display Logic ---
    if current_view == "welcome":
        display_welcome()

    elif current_view == "single_analysis":
        results = st.session_state.analysis_output
        # Display analysis tabs using the defined display functions
        st.header(f"🔎 Analysis Results for: `{results.get('metadata', {}).get('filename', 'N/A')}`")
        st.caption(f"Log ID: `{results.get('log_id', 'N/A')}`")
        analysis_data = results.get("analysis_results", {})
        df_current = results.get("df")
        debug_enabled = st.session_state.get("debug_mode_checkbox", False)

        if debug_enabled: # Show overall structure first in debug mode
            with st.expander("DEBUG: Full Analysis Output Structure"):
                 st.json(make_serializable({k: v if k != 'df' else f"DataFrame Shape: {v.shape}" if isinstance(v, pd.DataFrame) else "DataFrame Missing" for k, v in results.items()}))

        # Define Tab Structure dynamically based on available data
        tab_definitions = { # Maps Tab Name to (display_function, [args_for_function])
            "📈 Summary & Recs": (display_summary_tab, [results]),
            "📝 Metadata": (display_metadata_tab, [results]),
            "📉 PID Perf.": (display_pid_tab, [results, analyzer]),
            "⚙️ Motors": (display_motors_tab, [results, analyzer]),
            "🔊 Spectral": (display_spectral_tab, [results, analyzer]),
            "⚖️ Gyro/Accel": (display_gyro_accel_tab, [results, analyzer]),
            "🎮 RC/Pilot": (display_rc_pilot_tab, [results, analyzer]),
            "⚡ Power/Alt": (display_power_alt_tab, [results, analyzer]),
            "🛰️ RC Latency": (display_rc_latency_tab, [results, analyzer]),
            "🛩️ 3D Flight": (display_3d_flight_tab, [results, analyzer]),
            "📄 Raw Data": (display_raw_data_tab, [results]),
            "💾 History": (display_history_tab, [analyzer])
        }
        available_tabs = ["📈 Summary & Recs"] # Summary always first
        if results.get("metadata"): available_tabs.append("📝 Metadata")
        if isinstance(analysis_data.get("pid"), dict) and not analysis_data["pid"].get("error_overall"): available_tabs.append("📉 PID Perf.")
        if isinstance(analysis_data.get("motors"), dict) and not analysis_data["motors"].get("error_motors"): available_tabs.append("⚙️ Motors")
        if isinstance(analysis_data.get("spectral"), dict) and not analysis_data["spectral"].get("error"): available_tabs.append("🔊 Spectral")
        gyro_accel_res = analysis_data.get("gyro_accel") # Check result for gyro/accel
        if isinstance(gyro_accel_res, dict) and not gyro_accel_res.get("error_gyro"): available_tabs.append("⚖️ Gyro/Accel")
        rc_res = analysis_data.get("rc_commands") # Check result for rc
        if isinstance(rc_res, dict) and not rc_res.get("error"): available_tabs.append("🎮 RC/Pilot")
        alt_power_res = analysis_data.get("alt_power") # Check result for alt/power
        if isinstance(alt_power_res, dict) and not alt_power_res.get("error") and any(k not in ['error'] for k in alt_power_res.keys()): available_tabs.append("⚡ Power/Alt")
        latency_res = analysis_data.get("rc_gyro_latency") # Check result for latency
        if isinstance(latency_res, dict) and not latency_res.get("error") and any(k.endswith('_lag_ms') for k in latency_res.keys()): available_tabs.append("🛰️ RC Latency")
        if isinstance(df_current, pd.DataFrame):
            # Check conditions for 3D Flight tab
            gps_cols_exist = all(analyzer._find_col(df_current, [f'gpsCartesianCoords[{i}]']) is not None for i in range(3))
            alt_col_exist = analyzer._find_col(df_current, ['baroAlt', 'altitudeBaro', 'alt']) is not None
            if gps_cols_exist or alt_col_exist: available_tabs.append("🛩️ 3D Flight")
            available_tabs.append("📄 Raw Data") # Raw data tab if DF exists
        available_tabs.append("💾 History") # History always available

        # Create and populate tabs
        try:
            tabs = st.tabs(available_tabs)
            for i, tab_name in enumerate(available_tabs):
                with tabs[i]:
                    display_func, args_list = tab_definitions.get(tab_name, (None, None))
                    if display_func and args_list is not None:
                         try:
                              logger.debug(f"Rendering tab: {tab_name}")
                              display_func(*args_list) # Unpack arguments
                         except Exception as tab_display_err:
                              st.error(f"❌ Error displaying tab '{tab_name}': {tab_display_err}")
                              logger.error(f"Error rendering tab '{tab_name}'", exc_info=True)
                    else:
                        st.warning(f"Display function for tab '{tab_name}' not found.")
        except Exception as tab_creation_err:
            st.error(f"An error occurred while creating the analysis tabs: {tab_creation_err}")
            logger.error("Error creating analysis tabs", exc_info=True)

    elif current_view == "single_analysis_error":
        # Display the error message from the analysis attempt
        results = st.session_state.analysis_output
        filename = results.get("metadata", {}).get("filename", "the selected log")
        st.error(f"Analysis Failed for `{filename}`:")
        error_msg = results.get('error', 'Unknown error during analysis.')
        st.error(error_msg)
        dq = results.get('data_quality')
        if dq and isinstance(dq, dict): st.subheader("Data Quality Diagnostics (on Error)"); st.json(make_serializable(dq))
        df_on_error = results.get('df')
        if st.session_state.get("debug_mode_checkbox") and isinstance(df_on_error, pd.DataFrame): st.subheader("DataFrame Head (on Error)"); st.dataframe(df_on_error.head())

    elif current_view == "comparison":
        display_comparison_view(st.session_state.comparison_results) # Display comparison results

    elif current_view == "batch_summary":
        # --- Display Batch Summary Table ---
        st.header(f"📋 Batch Analysis Results ({len(st.session_state.batch_results)} Logs)")
        # ... (Batch summary display logic from previous snippet - seems okay) ...
        batch_summary_data = []
        valid_batch_indices = []
        for i, result in enumerate(st.session_state.batch_results):
            filename = result.get("metadata", {}).get("filename", f"Log {i+1}")
            timestamp = result.get("metadata", {}).get("analysis_info", {}).get("analysis_timestamp", "")
            status_icon, assessment_text, quality_score_val = "❓ Invalid", "Invalid result type", None
            if isinstance(result, dict):
                 if result.get("error") is not None:
                      status_icon = "❌ Error"; assessment_text = result.get("error", "Unknown")[:100] + "..."
                      quality_score_val = result.get("data_quality", {}).get("quality_score")
                 else:
                      status_icon = "✅ Success"; valid_batch_indices.append(i)
                      assessment = result.get("recommendations", {}).get("flight_assessment", {})
                      assessment_text = assessment.get("summary", "N/A")
                      quality_score_val = result.get("data_quality", {}).get("quality_score")
            summary_row = {"Log": filename, "Status": status_icon, "Quality Score": quality_score_val, "Assessment": assessment_text, "Timestamp": timestamp, "_Index": i}
            batch_summary_data.append(summary_row)
        if batch_summary_data:
            df_batch_disp = pd.DataFrame(batch_summary_data)
            if "Quality Score" in df_batch_disp.columns: df_batch_disp["Quality Score"] = df_batch_disp["Quality Score"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "N/A")
            if "Timestamp" in df_batch_disp.columns: df_batch_disp["Timestamp"] = pd.to_datetime(df_batch_disp["Timestamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_batch_disp.drop(columns=["_Index"]), use_container_width=True)
            if valid_batch_indices:
                 detail_options = { f"{batch_summary_data[idx]['Log']} ({batch_summary_data[idx]['Timestamp'] or 'No Time'})": idx for idx in valid_batch_indices }
                 selected_detail_key = st.selectbox("View details for a successfully analyzed log:", options=list(detail_options.keys()), index=None, placeholder="Select a log...", key="batch_detail_select")
                 if selected_detail_key:
                     selected_idx = detail_options[selected_detail_key]
                     st.session_state.analysis_output = st.session_state.batch_results[selected_idx]
                     st.session_state.current_file_name = st.session_state.analysis_output.get("metadata", {}).get("filename")
                     st.session_state.current_file_path = st.session_state.analysis_output.get("log_id")
                     st.session_state.batch_results = []; st.session_state.comparison_results = None; st.session_state.last_analyzed_file = None
                     logger.info(f"Switching from batch summary to detail view for index {selected_idx}")
                     st.rerun()
            else: st.info("No logs in batch analyzed successfully.")
        else: st.info("No batch results.")


# --- Run the App ---
if __name__ == "__main__":
     logger.info("--- Starting Streamlit App ---")
     try:
          main()
     except Exception as main_app_err:
          logger.critical(f"Critical error running Streamlit main function: {main_app_err}", exc_info=True)
          try: # Final attempt to show error in UI
               st.error("A critical error occurred in the application.")
               st.exception(main_app_err)
          except Exception as st_err:
               logger.error(f"Could not display critical error in Streamlit UI: {st_err}")
