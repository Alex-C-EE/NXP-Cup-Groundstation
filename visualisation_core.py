"""
visualization_core.py - Core visualization and threading system for race car telemetry
"""

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame
import pyqtgraph as pg
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue
import logging

@dataclass
class PlotUpdateCommand:
    """Represents a single plot update operation"""
    plot_id: str
    data: Any
    operation: str  # 'update', 'clear', 'create', 'delete'
    properties: Optional[dict] = None

class PlotManager(QObject):
    """Centralized plot management system"""
    update_ready = pyqtSignal(str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plots: Dict[str, pg.PlotItem] = {}
        self.update_queue = Queue()
        self.active = True
        
        # Create update timer in main thread
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._process_updates)
        self.update_timer.start(33)  # ~30 FPS
        
        self.logger = logging.getLogger('PlotManager')

    @pyqtSlot(str, object, str)
    def queue_update(self, plot_id: str, data: Any, operation: str = 'update'):
        """Thread-safe method to queue plot updates"""
        try:
            if self.active:
                command = PlotUpdateCommand(
                    plot_id=plot_id,
                    data=data,
                    operation=operation
                )
                self.update_queue.put(command)
        except Exception as e:
            self.error_occurred.emit(f"Error queueing update: {str(e)}")

    def _process_updates(self):
        """Process all pending plot updates in the main thread"""
        try:
            batch_updates = {}
            
            # Collect all updates for this frame
            while not self.update_queue.empty() and self.active:
                command = self.update_queue.get_nowait()
                batch_updates[command.plot_id] = command
            
            # Apply updates in a single frame
            if batch_updates:
                for plot_id, command in batch_updates.items():
                    if plot_id in self.plots:
                        self._safe_update_plot(self.plots[plot_id], command)
                        
        except Exception as e:
            self.error_occurred.emit(f"Error processing updates: {str(e)}")

    def _safe_update_plot(self, plot: pg.PlotItem, command: PlotUpdateCommand):
        """Safely update a plot with error handling"""
        try:
            if command.operation == 'update':
                if isinstance(command.data, tuple) and len(command.data) == 2:
                    x_data, y_data = command.data
                    x = np.array(x_data, dtype=np.float32)
                    y = np.array(y_data, dtype=np.float32)
                    plot.setData(x=x, y=y)
                else:
                    self.logger.warning(f"Invalid data format for plot {command.plot_id}")
            elif command.operation == 'clear':
                plot.clear()
            elif command.operation == 'properties':
                if command.properties:
                    self._update_plot_properties(plot, command.properties)
        except Exception as e:
            self.error_occurred.emit(f"Error updating plot: {str(e)}")

    def _update_plot_properties(self, plot: pg.PlotItem, properties: dict):
        """Update plot properties safely"""
        try:
            for key, value in properties.items():
                if hasattr(plot, key):
                    setattr(plot, key, value)
        except Exception as e:
            self.logger.error(f"Error updating plot properties: {str(e)}")

    def register_plot(self, plot_id: str, plot: pg.PlotItem):
        """Register a plot for management"""
        self.plots[plot_id] = plot

    def unregister_plot(self, plot_id: str):
        """Safely unregister a plot"""
        if plot_id in self.plots:
            del self.plots[plot_id]

    def cleanup(self):
        """Cleanup resources properly"""
        self.active = False
        self.update_timer.stop()
        self.plots.clear()

class DataProcessingWorker(QObject):
    """Worker object that handles data processing in a separate thread"""
    data_processed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, plot_manager: PlotManager, parent=None):
        super().__init__(parent)
        self.plot_manager = plot_manager
        self.active = True
        self.logger = logging.getLogger('DataProcessingWorker')
        
        # Buffer for tracking vehicle path
        self.path_buffer = []
        self.max_path_points = 1000

    @pyqtSlot(object)
    def process_telemetry(self, sensor_data):
        """Process incoming telemetry data"""
        try:
            if not self.active or not sensor_data:
                return
                
            # Process different aspects of the data
            self._process_vehicle_state(sensor_data)
            self._process_sensor_data(sensor_data)
            self._process_track_data(sensor_data)
            
            # Emit processed data for other components
            self.data_processed.emit(sensor_data)
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing telemetry: {str(e)}")

    def _process_vehicle_state(self, sensor_data):
        """Process vehicle state data"""
        try:
            # Update path history
            self.path_buffer.append((sensor_data.position_x, sensor_data.position_y))
            if len(self.path_buffer) > self.max_path_points:
                self.path_buffer.pop(0)
            
            # Queue path update
            path_array = np.array(self.path_buffer, dtype=np.float32)
            if len(path_array) > 0:
                self.plot_manager.queue_update(
                    'vehicle_path',
                    (path_array[:, 0], path_array[:, 1])
                )
                
            # Queue vehicle marker update
            self.plot_manager.queue_update(
                'vehicle_marker',
                (np.array([sensor_data.position_x]), 
                 np.array([sensor_data.position_y])),
                'update'
            )
            
            # Queue vehicle orientation update
            self.plot_manager.queue_update(
                'vehicle_marker',
                {'angle': -sensor_data.heading},
                'properties'
            )
            
        except Exception as e:
            self.logger.error(f"Error processing vehicle state: {str(e)}")

    def _process_sensor_data(self, sensor_data):
        """Process sensor readings"""
        try:
            # Process speed data
            self.plot_manager.queue_update(
                'speed_plot',
                (np.array([sensor_data.timestamp]), 
                 np.array([sensor_data.speed]))
            )
            
            # Process steering data
            self.plot_manager.queue_update(
                'steering_plot',
                (np.array([sensor_data.timestamp]), 
                 np.array([sensor_data.steering_angle]))
            )
            
        except Exception as e:
            self.logger.error(f"Error processing sensor data: {str(e)}")

    def _process_track_data(self, sensor_data):
        """Process track detection data"""
        try:
            if hasattr(sensor_data, 'camera_left') and sensor_data.camera_left:
                left_data = np.array(sensor_data.camera_left.raw_data)
                self.plot_manager.queue_update(
                    'camera_left',
                    (np.arange(len(left_data)), left_data)
                )
                
            if hasattr(sensor_data, 'camera_right') and sensor_data.camera_right:
                right_data = np.array(sensor_data.camera_right.raw_data)
                self.plot_manager.queue_update(
                    'camera_right',
                    (np.arange(len(right_data)), right_data)
                )
                
        except Exception as e:
            self.logger.error(f"Error processing track data: {str(e)}")

    def cleanup(self):
        """Clean up worker resources"""
        self.active = False
        self.path_buffer.clear()

class ProcessingThread(QThread):
    """Dedicated thread for data processing"""
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.logger = logging.getLogger('ProcessingThread')
        
    def setup_worker(self, worker: DataProcessingWorker):
        """Set up the worker object for this thread"""
        self.worker = worker
        self.worker.moveToThread(self)
        
    def run(self):
        """Main thread loop"""
        try:
            # Start event loop
            self.exec_()
        except Exception as e:
            self.error_occurred.emit(f"Thread error: {str(e)}")
        finally:
            if self.worker:
                self.worker.cleanup()
                
    def cleanup(self):
        """Clean up thread resources"""
        self.quit()
        self.wait()

class BasePlotWidget(QWidget):
    """Base widget for plot display"""
    def __init__(self, plot_manager: PlotManager, parent=None):
        super().__init__(parent)
        self.plot_manager = plot_manager
        self.plots = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI and plots"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
    def create_plot(self, plot_id: str, **kwargs) -> pg.PlotItem:
        """Create and register a new plot"""
        plot_widget = pg.PlotWidget(**kwargs)
        plot = plot_widget.getPlotItem()
        self.plot_manager.register_plot(plot_id, plot)
        self.plots[plot_id] = plot_widget
        return plot

    def cleanup(self):
        """Clean up widget resources"""
        for plot_id in list(self.plots.keys()):
            self.plot_manager.unregister_plot(plot_id)
        self.plots.clear()

class LivePlotWidget(BasePlotWidget):
    """Widget for displaying real-time telemetry plots"""
    def setup_ui(self):
        super().setup_ui()
        
        # Create plot frame
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        
        # Create plots
        speed_plot = self.create_plot('speed_plot', title="Vehicle Speed")
        speed_plot.setLabel('left', 'Speed', units='m/s')
        speed_plot.setLabel('bottom', 'Time', units='s')
        plot_layout.addWidget(self.plots['speed_plot'])
        
        steering_plot = self.create_plot('steering_plot', title="Steering Angle")
        steering_plot.setLabel('left', 'Angle', units='deg')
        steering_plot.setLabel('bottom', 'Time', units='s')
        plot_layout.addWidget(self.plots['steering_plot'])
        
        self.layout.addWidget(plot_frame)

class TrackVisualizer(BasePlotWidget):
    """Widget for visualizing track and vehicle position"""
    def setup_ui(self):
        super().setup_ui()
        
        # Create main visualization plot
        self.track_plot = self.create_plot('track_plot')
        self.track_plot.setAspectLocked(True)
        self.track_plot.setLabel('left', 'Y Position', units='m')
        self.track_plot.setLabel('bottom', 'X Position', units='m')
        
        # Add to layout
        self.layout.addWidget(self.plots['track_plot'])
        
        # Create camera data plots
        camera_frame = QFrame()
        camera_layout = QVBoxLayout(camera_frame)
        
        left_camera = self.create_plot('camera_left', title="Left Camera")
        right_camera = self.create_plot('camera_right', title="Right Camera")
        
        camera_layout.addWidget(self.plots['camera_left'])
        camera_layout.addWidget(self.plots['camera_right'])
        
        self.layout.addWidget(camera_frame)