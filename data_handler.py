# data_handler.py
import os
import csv
import serial
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from PyQt5.QtCore import QObject, pyqtSignal, QSocketNotifier, QTimer, QThread, Qt, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import QApplication
from dataclasses import dataclass
import math

from threading_components import DataThread, DataProcessingWorker

@dataclass
class CameraData:
    """Data structure for line scan camera readings"""
    raw_data: List[int]  # List of 240 pixel values
    track_edges: Tuple[int, int]  # Left and right track edge positions
    track_width: float  # Calculated track width in mm
    confidence: float  # Detection confidence 0-1

@dataclass
class MotorData:
    """Data structure for motor-related readings"""
    duty: float  # Motor duty cycle (0-100%)
    current: float  # Current draw in amps
    temperature: float  # Motor temperature in Celsius

@dataclass
class EDFData:
    """Data structure for Electric Ducted Fan readings"""
    power: float  # Power level (0-100%)
    force: float  # Measured force in Newtons
    
@dataclass
class SystemHealth:
    """Data structure for system health monitoring"""
    cpu_load: List[float]  # CPU load for each board (0-100%)
    board_temps: List[float]  # Temperature of each board
    can_errors: List[int]  # CAN bus error counts
    loop_time: float  # Control loop time in microseconds
    pid_state: int  # PID controller states bitfield
    control_mode: int  # Current control mode
    ir_values: List[float]  # IR sensor readings
    tof_distance: float  # Time of flight sensor distance

@dataclass
class SensorData:
    """Data structure for complete sensor readings"""
    # Packet metadata
    timestamp: float
    packet_counter: int
    system_state: int
    error_flags: int
    
    # Vehicle state
    speed: float
    steering_angle: float
    position_x: float
    position_y: float
    heading: float
    
    # Camera data
    camera_right: CameraData
    camera_left: CameraData  # Added for completeness
    centerline_error: float  # Track centerline error in mm
    track_confidence: float  # Overall track detection confidence (0-100)
    
    # IMU data (fusion results)
    acceleration: Tuple[float, float, float]  # X, Y, Z acceleration in m/s²
    gyroscope: Tuple[float, float, float]  # X, Y, Z rotation in deg/s
    
    # Motor systems
    motor_left: MotorData
    motor_right: MotorData
    
    # Downforce system
    edf_front: EDFData
    edf_rear: EDFData
    
    # Power system
    battery_voltage: float  # Battery voltage in volts
    battery_current: float  # Battery current draw in amps
    battery_power: float  # Calculated power consumption in watts
    
    # System health
    system_health: SystemHealth

class DataHandler(QObject):
    """Handles real-time data collection and processing"""
    data_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool)
    playback_finished = pyqtSignal()

    def __init__(self, config_manager, data_queue, parent=None):
            super().__init__(parent)
            self.config = config_manager
            self.data_queue = data_queue
            self.running = False
            self.playback_mode = False

            # Initialize position tracking variables
            self.last_timestamp = None
            self.position_x = 0.0
            self.position_y = 0.0
            self.heading = 0.0

            # Initialize data processing components
            self._setup_processing_thread()

            # Initialize other components
            self.serial_port = None
            self.notifier = None
            self.buffer_size = self.config.get("display", "plot_buffer_size", 1000)
            self._init_data_buffers()

    def _setup_processing_thread(self):
        """Set up the data processing thread and worker"""
        # Create worker first
        self.processing_worker = DataProcessingWorker()
        
        # Create thread and set up worker
        self.processing_thread = DataThread(self)
        self.processing_thread.setup_worker(self.processing_worker)
        
        # Connect signals
        self.processing_worker.data_processed.connect(self.data_ready)
        self.processing_worker.error_occurred.connect(self.error_occurred)
        
        # Start the thread
        self.processing_thread.start()

    def start(self, port: str = None, baudrate: int = None):
        """Start data collection with proper thread handling"""
        if self.playback_mode:
            self.stop_playback()

        # Store configuration for later use in the main thread
        self._port = port if port is not None else self.config.get("serial", "default_port")
        self._baudrate = baudrate if baudrate is not None else self.config.get("serial", "baudrate")

        # Use QMetaObject to ensure serial setup happens in main thread
        QMetaObject.invokeMethod(self, '_setup_serial',
                               Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _setup_serial(self):
        """Setup serial port in the main thread"""
        try:
            self.serial_port = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self.config.get("serial", "timeout")
            )

            if hasattr(self, 'notifier') and self.notifier:
                self.notifier.setEnabled(False)
                self.notifier.deleteLater()

            self.notifier = QSocketNotifier(
                self.serial_port.fileno(),
                QSocketNotifier.Type.Read,
                self
            )
            self.notifier.activated.connect(self._read_data)
            self.notifier.setEnabled(True)

            self.running = True
            self.connection_status.emit(True)

        except serial.SerialException as e:
            self.error_occurred.emit(f"Could not open port {self._port}: {str(e)}")
            self.connection_status.emit(False)

    def stop(self):
        """Stop data collection and clean up resources"""
        self.running = False
        
        # Clean up notifier
        if hasattr(self, 'notifier') and self.notifier:
            self.notifier.setEnabled(False)
            self.notifier.deleteLater()
            self.notifier = None

        # Close serial port
        if hasattr(self, 'serial_port') and self.serial_port:
            if self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port = None

        # Stop processing thread
        if hasattr(self, 'processing_thread'):
            self.processing_thread.cleanup()

        # Reset playback state
        self.playback_mode = False
        self.connection_status.emit(False)

    def _read_data(self):
        """Read and process data from serial port"""
        if not self.running or self.playback_mode:
            return

        try:
            line = self.serial_port.readline().decode('ascii').strip()
            if line:
                sensor_data = self._parse_data(line)
                if sensor_data:
                    # Update buffers in main thread
                    self._update_buffers(sensor_data)
                    # Process data in worker thread
                    self.processing_worker.process_data(sensor_data)
                    
        except Exception as e:
            self.error_occurred.emit(f"Data reading error: {str(e)}")

    def _init_data_buffers(self):
        """Initialize circular buffers for all telemetry data"""
        self.data_buffers = {
            'timestamp': deque(maxlen=self.buffer_size),
            'speed': deque(maxlen=self.buffer_size),
            'steering_angle': deque(maxlen=self.buffer_size),
            'position_x': deque(maxlen=self.buffer_size),
            'position_y': deque(maxlen=self.buffer_size),
            'heading': deque(maxlen=self.buffer_size),
            'centerline_error': deque(maxlen=self.buffer_size),
            'track_confidence': deque(maxlen=self.buffer_size),
            'battery_voltage': deque(maxlen=self.buffer_size),
            'battery_current': deque(maxlen=self.buffer_size),
            'battery_power': deque(maxlen=self.buffer_size)
        }
        
        # Initialize 3D data buffers
        for axis in ['x', 'y', 'z']:
            self.data_buffers[f'accel_{axis}'] = deque(maxlen=self.buffer_size)
            self.data_buffers[f'gyro_{axis}'] = deque(maxlen=self.buffer_size)

    def start_playback(self, csv_file: str, speed: float = 1.0):
        """Start data playback with proper thread handling"""
        try:
            self.stop()
            
            # Read playback data
            self.playback_data = []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for line in reader:
                    sensor_data = self._parse_data(','.join(line))
                    if sensor_data:
                        self.playback_data.append(sensor_data)

            if not self.playback_data:
                self.error_occurred.emit("No valid data found in CSV file")
                return

            # Initialize playback state
            self.playback_mode = True
            self.playback_speed = speed
            self.playback_index = 0
            self.running = True

            # Start playback timer
            self.playback_timer = QTimer(self)
            self.playback_timer.timeout.connect(self._playback_tick)
            interval = int(self.config.get("display", "update_interval_ms") / speed)
            self.playback_timer.start(max(1, interval))

            self.connection_status.emit(True)

        except Exception as e:
            self.error_occurred.emit(f"Playback error: {str(e)}")
            self.connection_status.emit(False)

    def _playback_tick(self):
        """Handle playback updates with thread safety"""
        if not self.running or self.playback_index >= len(self.playback_data):
            self.playback_timer.stop()
            self.playback_finished.emit()
            return

        try:
            # Get current data point
            sensor_data = self.playback_data[self.playback_index]
            
            # Update buffers in main thread
            self._update_buffers(sensor_data)
            
            # Process data in worker thread using correct method name
            self.processing_worker.process_data(sensor_data)
            
            self.playback_index += 1

        except Exception as e:
            self.error_occurred.emit(f"Playback processing error: {str(e)}")

    def stop_playback(self):
        """Stop playback and cleanup"""
        if hasattr(self, 'playback_timer') and self.playback_timer:
            self.playback_timer.stop()
        self.playback_mode = False
        self.playback_data = []
        self.playback_index = 0

    def _setup_notifier(self):
        """Create the QSocketNotifier in the correct thread"""
        try:
            # Create the notifier - this must happen in the thread
            self.notifier = QSocketNotifier(
                self.serial_port.fileno(),
                QSocketNotifier.Read
            )
            self.notifier.activated.connect(self._read_data)
            self.running = True
        except Exception as e:
            self.error_occurred.emit(f"Failed to set up notifier: {str(e)}")
            self.connection_status.emit(False)

    def _unpack_camera_data(self, data_bytes: List[str]) -> List[int]:
        """Unpack 60 bytes of camera data into 240 pixel values"""
        pixels = []
        for byte_str in data_bytes:
            byte_val = int(byte_str)
            # Each byte contains 4 2-bit pixels
            for i in range(3, -1, -1):
                pixel_val = (byte_val >> (i * 2)) & 0x03
                pixels.append(pixel_val)
        return pixels

    def _update_position(self, speed: float, steering_angle: float, timestamp: float):
        """Update vehicle position using dead reckoning"""
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return
        
        # Calculate time delta
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Convert steering angle to radians
        steering_rad = math.radians(steering_angle)
        heading_rad = math.radians(self.heading)
        
        # Calculate position change
        # For small time steps, we can approximate the arc as a straight line
        distance = speed * dt
        
        # Update heading based on steering angle and speed
        # This is a simplified model - you might want to adjust the steering effect
        heading_change = (speed * math.tan(steering_rad) / 1) * dt
        self.heading += math.degrees(heading_change)
        self.heading %= 360  # Keep heading between 0 and 360 degrees
        
        # Update position
        self.position_x += distance * math.cos(heading_rad)
        self.position_y += distance * math.sin(heading_rad)

    def _parse_data(self, line: str) -> Optional[SensorData]:
        """Parse incoming CSV data line into SensorData object with better error handling"""
        try:
            # Split CSV line and convert to numeric values
            fields = [f.strip() for f in line.split(',')]

            # Basic validation
            if len(fields) < 4:  # Minimum required fields
                raise ValueError(f"Invalid packet length: {len(fields)}")

            # Create default objects for missing data
            camera_left = CameraData(
                raw_data=[0] * 240,
                track_edges=(0, 0),
                track_width=0.0,
                confidence=0.0
            )

            camera_right = CameraData(
                raw_data=[0] * 240,
                track_edges=(0, 0),
                track_width=0.0,
                confidence=0.0
            )

            motor_left = MotorData(
                duty=0.0,
                current=0.0,
                temperature=0.0
            )

            motor_right = MotorData(
                duty=0.0,
                current=0.0,
                temperature=0.0
            )

            edf_front = EDFData(
                power=0.0,
                force=0.0
            )

            edf_rear = EDFData(
                power=0.0,
                force=0.0
            )

            system_health = SystemHealth(
                cpu_load=[0.0, 0.0, 0.0],
                board_temps=[0.0, 0.0, 0.0],
                can_errors=[0, 0],
                loop_time=0.0,
                pid_state=0,
                control_mode=0,
                ir_values=[0.0, 0.0, 0.0, 0.0],
                tof_distance=0.0
            )

            # Parse basic fields with safe conversions
            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            # Get minimum required fields
            timestamp = safe_float(fields[0]) / 1000.0 if len(fields) > 0 else time.time()
            speed = safe_float(fields[1]) if len(fields) > 1 else 0.0
            steering = safe_float(fields[2]) if len(fields) > 2 else 0.0

            # Update position tracking with available data
            self._update_position(speed, steering, timestamp)

            # Create sensor data object with available data
            return SensorData(
                timestamp=timestamp,
                packet_counter=0,
                system_state=0,
                error_flags=0,
                speed=speed,
                steering_angle=steering,
                position_x=self.position_x,
                position_y=self.position_y,
                heading=self.heading,
                camera_left=camera_left,
                camera_right=camera_right,
                centerline_error=safe_float(fields[3]) if len(fields) > 3 else 0.0,
                track_confidence=100.0,  # Default to full confidence
                acceleration=(0.0, 0.0, 0.0),
                gyroscope=(0.0, 0.0, 0.0),
                motor_left=motor_left,
                motor_right=motor_right,
                edf_front=edf_front,
                edf_rear=edf_rear,
                battery_voltage=12.0,  # Default values
                battery_current=0.0,
                battery_power=0.0,
                system_health=system_health
            )

        except Exception as e:
            self.error_occurred.emit(f"Data parsing error: {str(e)}")
            return None

    def _update_buffers(self, data: SensorData):
        """Update circular buffers with new data"""
        # Update simple numeric buffers
        self.data_buffers['timestamp'].append(data.timestamp)
        self.data_buffers['speed'].append(data.speed)
        self.data_buffers['steering_angle'].append(data.steering_angle)
        self.data_buffers['position_x'].append(data.position_x)
        self.data_buffers['position_y'].append(data.position_y)
        self.data_buffers['heading'].append(data.heading)
        self.data_buffers['centerline_error'].append(data.centerline_error)
        self.data_buffers['track_confidence'].append(data.track_confidence)
        self.data_buffers['battery_voltage'].append(data.battery_voltage)
        self.data_buffers['battery_current'].append(data.battery_current)
        self.data_buffers['battery_power'].append(data.battery_power)
        
        # Update IMU data buffers
        for i, axis in enumerate(['x', 'y', 'z']):
            self.data_buffers[f'accel_{axis}'].append(data.acceleration[i])
            self.data_buffers[f'gyro_{axis}'].append(data.gyroscope[i])

    def get_buffer_data(self, field: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get timestamps and values for a specific field"""
        try:
            times = np.array(self.data_buffers['timestamp'])
            values = np.array(self.data_buffers[field])
            return times, values
        except KeyError:
            self.error_occurred.emit(f"Invalid field requested: {field}")
            return np.array([]), np.array([])

    def get_latest_data(self) -> Optional[SensorData]:
        """Get the most recent sensor data as a proper SensorData object"""
        try:
            if not self.data_buffers['timestamp']:
                return None

            # Create CameraData objects (with dummy data if not available)
            camera_left = CameraData(
                raw_data=[0] * 240,  # Dummy data
                track_edges=(0, 0),
                track_width=0.0,
                confidence=0.0
            )

            camera_right = CameraData(
                raw_data=[0] * 240,  # Dummy data
                track_edges=(0, 0),
                track_width=0.0,
                confidence=0.0
            )

            # Create MotorData objects
            motor_left = MotorData(
                duty=0.0,
                current=0.0,
                temperature=0.0
            )

            motor_right = MotorData(
                duty=0.0,
                current=0.0,
                temperature=0.0
            )

            # Create EDFData objects
            edf_front = EDFData(
                power=0.0,
                force=0.0
            )

            edf_rear = EDFData(
                power=0.0,
                force=0.0
            )

            # Create SystemHealth object
            system_health = SystemHealth(
                cpu_load=[0.0, 0.0, 0.0],
                board_temps=[0.0, 0.0, 0.0],
                can_errors=[0, 0],
                loop_time=0.0,
                pid_state=0,
                control_mode=0,
                ir_values=[0.0, 0.0, 0.0, 0.0],
                tof_distance=0.0
            )

            # Create and return the complete SensorData object
            return SensorData(
                timestamp=self.data_buffers['timestamp'][-1],
                packet_counter=0,  # We don't store this in buffers
                system_state=0,    # We don't store this in buffers
                error_flags=0,     # We don't store this in buffers
                speed=self.data_buffers['speed'][-1],
                steering_angle=self.data_buffers['steering_angle'][-1],
                position_x=self.data_buffers['position_x'][-1],
                position_y=self.data_buffers['position_y'][-1],
                heading=self.data_buffers['heading'][-1],
                camera_left=camera_left,
                camera_right=camera_right,
                centerline_error=self.data_buffers['centerline_error'][-1],
                track_confidence=self.data_buffers['track_confidence'][-1],
                acceleration=tuple(self.data_buffers[f'accel_{axis}'][-1] for axis in ['x', 'y', 'z']),
                gyroscope=tuple(self.data_buffers[f'gyro_{axis}'][-1] for axis in ['x', 'y', 'z']),
                motor_left=motor_left,
                motor_right=motor_right,
                edf_front=edf_front,
                edf_rear=edf_rear,
                battery_voltage=self.data_buffers['battery_voltage'][-1],
                battery_current=self.data_buffers['battery_current'][-1],
                battery_power=self.data_buffers['battery_power'][-1],
                system_health=system_health
            )

        except (IndexError, KeyError) as e:
            self.error_occurred.emit(f"Error getting latest data: {str(e)}")
            return None

    def get_camera_data(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Get the latest camera data for both cameras"""
        if not hasattr(self, 'latest_sensor_data') or not self.latest_sensor_data:
            return None, None
        return (self.latest_sensor_data.camera_left.raw_data,
                self.latest_sensor_data.camera_right.raw_data)

    def get_track_info(self) -> Tuple[float, float]:
        """Get the latest track centerline error and confidence"""
        try:
            return (self.data_buffers['centerline_error'][-1],
                    self.data_buffers['track_confidence'][-1])
        except (IndexError, KeyError):
            return 0.0, 0.0

    def get_position(self) -> Tuple[float, float, float]:
        """Get the latest vehicle position and heading"""
        try:
            return (self.data_buffers['position_x'][-1],
                    self.data_buffers['position_y'][-1],
                    self.data_buffers['heading'][-1])
        except (IndexError, KeyError):
            return 0.0, 0.0, 0.0

    def get_system_health(self) -> Dict[str, float]:
        """Get the latest system health metrics"""
        if not hasattr(self, 'latest_sensor_data') or not self.latest_sensor_data:
            return {}
            
        health = self.latest_sensor_data.system_health
        return {
            'cpu_load': health.cpu_load,
            'board_temps': health.board_temps,
            'can_errors': health.can_errors,
            'loop_time': health.loop_time,
            'pid_state': health.pid_state,
            'control_mode': health.control_mode
        }

    def reset_position(self):
        """Reset position tracking to origin"""
        self.position_x = 0.0
        self.position_y = 0.0
        self.heading = 0.0
        self.last_timestamp = None