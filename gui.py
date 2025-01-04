# gui.py
import os
import csv
import numpy as np
from datetime import datetime
from dataclasses import asdict
from PyQt5.QtWidgets import (
    QMainWindow, QStackedWidget, QLabel, QToolBar,
    QStatusBar, QVBoxLayout, QWidget, QGridLayout,
    QFrame, QHBoxLayout, QPushButton, QApplication,
    QDialog, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox,  # Added QDoubleSpinBox
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QTextEdit,
    QProgressBar, QSplitter, QScrollArea, QShortcut,
    QTabWidget, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QThread
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QPainterPath,
    QKeySequence
)
import pyqtgraph as pg
import time
from visualisation_core import PlotManager, BasePlotWidget, LivePlotWidget
from track_visualizer import TrackVisualizer as TrackVisualizerCore
from visualisation_core import TrackVisualizer as TrackVisualizerWidget
import logging

class ClickableLabel(QLabel):
    """Label that emits a signal when clicked, useful for interactive data displays"""
    clicked = pyqtSignal(str)
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.objectName())

# CustomPlotWidget class specifically designed for PyQt5
class CustomPlotWidget(pg.PlotWidget):
    """Enhanced plot widget with optimized settings for race telemetry"""
    def __init__(self, title=None, parent=None):
        # Initialize with specific view settings for PyQt5
        super().__init__(parent=parent)
        
        # Configure background based on theme
        if parent and parent.dark_mode:
            self.setBackground('k')  # Dark theme
        else:
            self.setBackground('w')  # Light theme
        
        # Add grid with slight transparency
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Set up the title if provided
        if title:
            self.setTitle(title, size='12pt')
        
        # Configure the view box for better interaction
        view_box = self.getPlotItem().getViewBox()
        view_box.setMouseMode(pg.ViewBox.RectMode)
        
        # Set up reasonable axis defaults
        self.getPlotItem().setDownsampling(mode='peak')
        self.getPlotItem().setClipToView(True)
        
        # Configure for performance
        self.maxPoints = 1000  # Limit the number of points to display
        
        # Initialize plot curves dictionary
        self.curves = {}

    def plot(self, x, y, pen='b', name=None, clear=False, **kwargs):
        """Enhanced plotting method that reuses plot curves"""
        try:
            # Convert inputs to numpy arrays if they aren't already
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # Ensure x and y have the same shape
            if x.shape != y.shape:
                print(f"Shape mismatch: x={x.shape}, y={y.shape}")
                return None
            
            # Clear all curves if requested
            if clear:
                self.clear()
                self.curves = {}
            
            # Create or update curve
            curve_key = name if name else 'default'
            if curve_key not in self.curves:
                # Create new curve
                self.curves[curve_key] = self.getPlotItem().plot(
                    x=x,
                    y=y,
                    pen=pen,
                    name=name,
                    **kwargs
                )
            else:
                # Update existing curve
                self.curves[curve_key].setData(x=x, y=y)
            
            return self.curves[curve_key]
            
        except Exception as e:
            print(f"Error plotting data: {str(e)}")
            return None

    def clear(self):
        """Clear all plots"""
        self.getPlotItem().clear()
        self.curves = {}

class TabWidget(QWidget):
    """Base class for tab widgets with common functionality"""
    def __init__(self, config_manager, plot_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.plot_manager = plot_manager
        self.dark_mode = self.config.get("display", "dark_mode")
        
        # Create a main layout for the tab
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Create a content widget to hold the actual tab content
        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        
        # Add content widget to main layout
        self.main_layout.addWidget(self.content_widget)
        
        # Set up the UI (will be implemented by subclasses)
        self.setup_ui()
        self.apply_theme()
        
    def setup_ui(self):
        """Setup UI elements - to be implemented by subclasses"""
        pass
        
    def apply_theme(self):
        """Apply current theme to all widgets"""
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #2b2b2b; color: #ffffff; }
                QGroupBox { border: 1px solid #404040; margin-top: 6px; }
                QGroupBox::title { color: #ffffff; }
            """)
        else:
            self.setStyleSheet("")
            
        # Update plot backgrounds if any
        for plot in self.findChildren(pg.PlotWidget):
            plot.setBackground(self.palette().color(self.backgroundRole()))
            
    def get_theme_palette(self):
        """Get color palette based on current theme"""
        palette = self.palette()
        if self.dark_mode:
            # Dark theme colors
            palette.setColor(self.backgroundRole(), QColor(25, 25, 25))
            palette.setColor(self.foregroundRole(), QColor(255, 255, 255))
        else:
            # Light theme colors
            palette.setColor(self.backgroundRole(), QColor(240, 240, 240))
            palette.setColor(self.foregroundRole(), QColor(0, 0, 0))
        return palette

class RaceViewTab(TabWidget):
    """Primary race visualization tab showing track view and real-time telemetry"""
    
    def __init__(self, config_manager, track_visualizer, parent=None):
        # Initialize instance variables before super().__init__
        self.track_visualizer = track_visualizer
        self.plot_items_initialized = False
        self.path_points = []
        self.vehicle_path = None
        self.vehicle_marker = None
        self.track_plot = None
        self.camera_left = None
        self.camera_right = None
        self.speed_plot = None
        self.steering_plot = None
        
        # Now call super().__init__ which will call setup_ui
        super().__init__(config_manager, parent)
        
        # Connect visualization updates after everything is initialized
        self.track_visualizer.visualization_updated.connect(self.update_visualization)
        
    def setup_ui(self):
        # Create horizontal layout for main content
        content_split = QHBoxLayout()
        self.content_layout.addLayout(content_split)

        # Create left panel for track visualization
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)

        # Create track visualization widget
        self.track_plot = pg.PlotWidget()
        self.track_plot.setAspectLocked(True)
        self.track_plot.showGrid(x=True, y=True, alpha=0.3)
        self.track_plot.setLabel('left', 'Y Position', units='m')
        self.track_plot.setLabel('bottom', 'X Position', units='m')

        # Store plot items as instance variables
        self.plot_items = {}
        self._init_plot_items()

        left_layout.addWidget(self.track_plot)

        # Create camera visualization frame
        camera_frame = QFrame()
        camera_layout = QHBoxLayout(camera_frame)

        # Initialize camera plots with stored curves
        self.camera_left = pg.PlotWidget(title="Left Camera")
        self.camera_right = pg.PlotWidget(title="Right Camera")
        self.camera_left_curve = self.camera_left.plot([], [])
        self.camera_right_curve = self.camera_right.plot([], [])

        for camera in [self.camera_left, self.camera_right]:
            camera.setFixedHeight(150)
            camera.hideAxis('left')
            camera.setLabel('bottom', 'Pixel')
            camera_layout.addWidget(camera)

        left_layout.addWidget(camera_frame)

        # Create right panel for telemetry display
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        # Vehicle dynamics group
        dynamics_group = QGroupBox("Vehicle Dynamics")
        dynamics_layout = QGridLayout(dynamics_group)

        # Initialize plots with stored curves
        self.speed_plot = CustomPlotWidget("Speed", self)
        self.steering_plot = CustomPlotWidget("Steering Angle", self)
        self.speed_curve = self.speed_plot.plot([], [], pen='b')
        self.steering_curve = self.steering_plot.plot([], [], pen='r')

        dynamics_layout.addWidget(self.speed_plot, 0, 0)
        dynamics_layout.addWidget(self.steering_plot, 0, 1)

        # Add layouts to groups
        dynamics_group.setLayout(dynamics_layout)

        # Add groups to layouts
        right_layout.addWidget(dynamics_group)
        right_panel.setLayout(right_layout)

        # Add panels to main content layout
        content_split.addWidget(left_panel, stretch=2)
        content_split.addWidget(right_panel, stretch=1)

    def _init_plot_items(self):
        """Initialize plot items only once"""
        if not hasattr(self, 'plot_items_initialized') or not self.plot_items_initialized:
            # Vehicle path curve
            self.path_curve = pg.PlotCurveItem(
                pen=pg.mkPen('y', width=2),
                name='Vehicle Path'
            )
            self.track_plot.addItem(self.path_curve)

            # Vehicle marker (scatter plot)
            self.vehicle_marker = pg.ScatterPlotItem(
                pen=pg.mkPen('r', width=2),
                brush=pg.mkBrush('r'),
                symbol='t',
                size=20,
                name='Vehicle Position'
            )
            self.track_plot.addItem(self.vehicle_marker)

            self.plot_items_initialized = True

    def update_display(self, sensor_data):
        """Update all display elements with new sensor data"""
        if not sensor_data:
            return

        try:
            # Update vehicle position and path
            self.path_points.append((sensor_data.position_x, sensor_data.position_y))
            if len(self.path_points) > 1000:  # Limit path length
                self.path_points.pop(0)

            # Convert points to numpy arrays for plotting
            if self.path_points:
                points = np.array(self.path_points)
                self.logger.debug(f"Updating path plot with {len(points)} points")
                self.path_curve.setData(
                    x=points[:, 0].astype(np.float32),
                    y=points[:, 1].astype(np.float32)
                )

            # Update vehicle marker
            self.vehicle_marker.setData(
                x=[sensor_data.position_x],
                y=[sensor_data.position_y]
            )

            # Update dynamics plots
            current_time = np.array([sensor_data.timestamp], dtype=np.float32)

            # Speed plot
            self.speed_curve.setData(
                x=current_time,
                y=np.array([sensor_data.speed], dtype=np.float32)
            )

            # Steering plot
            self.steering_curve.setData(
                x=current_time,
                y=np.array([sensor_data.steering_angle], dtype=np.float32)
            )

            if self.playback_index % 100 == 0:
                self.logger.debug(f"Display updated: Speed={sensor_data.speed:.2f}, Position=({sensor_data.position_x:.2f}, {sensor_data.position_y:.2f})")

        except Exception as e:
            self.logger.error(f"Error updating display: {str(e)}", exc_info=True)
            
    def _update_dynamics_plots(self, sensor_data):
        """Update vehicle dynamics plots"""
        try:
            # Get current time for x-axis
            current_time = np.array([sensor_data.timestamp], dtype=np.float32)
            
            # Speed plot
            self.speed_plot.plot(
                x=current_time,
                y=np.array([sensor_data.speed], dtype=np.float32),
                clear=True
            )
            
            # Steering plot
            self.steering_plot.plot(
                x=current_time,
                y=np.array([sensor_data.steering_angle], dtype=np.float32),
                clear=True
            )
            
        except Exception as e:
            print(f"Error updating dynamics plots: {str(e)}")
            
    def reset_display(self):
        """Reset all display elements"""
        if not self.plot_items_initialized:
            return
            
        self.path_points.clear()
        if self.vehicle_path:
            self.vehicle_path.setData(x=np.array([]), y=np.array([]))
        if self.vehicle_marker:
            self.vehicle_marker.setData(x=np.array([]), y=np.array([]))
            
    def reset_display(self):
        """Reset all display elements"""
        self.path_points.clear()
        if self.vehicle_path:
            self.vehicle_path.setData(x=np.array([]), y=np.array([]))
        if self.vehicle_marker:
            self.vehicle_marker.setData(x=np.array([]), y=np.array([]))
        
    def update_visualization(self, plot_items):
        """Update the visualization with new plot items"""
        if 'vehicle' in plot_items:
            self.track_plot.addItem(plot_items['vehicle'])
        if 'path' in plot_items:
            self.track_plot.addItem(plot_items['path'])
        if 'predicted_path' in plot_items:
            self.track_plot.addItem(plot_items['predicted_path'])
        if 'left_boundary' in plot_items:
            self.track_plot.addItem(plot_items['left_boundary'])
        if 'right_boundary' in plot_items:
            self.track_plot.addItem(plot_items['right_boundary'])

        
    def setup_plot_colors(self):
        """Configure plot colors based on theme"""
        if self.dark_mode:
            self.plot_colors = {
                'speed': 'g',
                'steering': 'y',
                'motor_left': 'c',
                'motor_right': 'm',
                'battery': 'r'
            }
        else:
            self.plot_colors = {
                'speed': 'darkGreen',
                'steering': 'darkBlue',
                'motor_left': 'darkCyan',
                'motor_right': 'darkMagenta',
                'battery': 'darkRed'
            }
        
    def _update_system_plots(self, sensor_data):
        """Update system status plots"""
        # Motor plot
        self.motors_plot.plot(
            [sensor_data.timestamp],
            [sensor_data.motor_left.duty],
            pen=self.plot_colors['motor_left'],
            name="Left Motor",
            clear=True
        )
        self.motors_plot.plot(
            [sensor_data.timestamp],
            [sensor_data.motor_right.duty],
            pen=self.plot_colors['motor_right'],
            name="Right Motor"
        )
        
        # Battery plot
        self.battery_plot.plot(
            [sensor_data.timestamp],
            [sensor_data.battery_power],
            pen=self.plot_colors['battery'],
            clear=True
        )

class SystemHealthTab(TabWidget):
    """Tab for detailed system health monitoring"""
    
    def __init__(self, config_manager, plot_manager, parent=None):
        # Initialize instance variables
        self.plot_items = {}
        self.cpu_plots = []
        self.motor_temp_plot = None
        self.board_temp_plot = None
        self.can_error_plot = None
        self.plot_manager = plot_manager
        super().__init__(config_manager, parent)
        
    def setup_ui(self):
        # CPU monitoring section
        cpu_group = QGroupBox("CPU Load")
        cpu_layout = QHBoxLayout(cpu_group)
        
        self.cpu_plots = []
        for i in range(3):
            plot = CustomPlotWidget(f"CPU {i+1}", self)
            self.cpu_plots.append(plot)
            # Initialize plot items for each CPU
            self.plot_items[f'cpu_{i}'] = plot.plot([], [], 
                                                   pen='b',
                                                   name=f'CPU {i+1}')
            cpu_layout.addWidget(plot)
            
        self.content_layout.addWidget(cpu_group)
        
        # Temperature monitoring section
        temp_group = QGroupBox("Temperature Monitoring")
        temp_layout = QGridLayout(temp_group)
        
        self.motor_temp_plot = CustomPlotWidget("Motor Temperatures", self)
        self.board_temp_plot = CustomPlotWidget("Board Temperatures", self)
        
        # Initialize motor temperature plot items
        self.plot_items['motor_left'] = self.motor_temp_plot.plot([], [],
                                                                 pen='r',
                                                                 name="Left Motor")
        self.plot_items['motor_right'] = self.motor_temp_plot.plot([], [],
                                                                  pen='g',
                                                                  name="Right Motor")
        
        # Initialize board temperature plot items
        for i in range(3):
            self.plot_items[f'board_temp_{i}'] = self.board_temp_plot.plot(
                [], [],
                pen=pg.intColor(i, 3),
                name=f"Board {i+1}"
            )
        
        temp_layout.addWidget(self.motor_temp_plot, 0, 0)
        temp_layout.addWidget(self.board_temp_plot, 0, 1)
        
        self.content_layout.addWidget(temp_group)
        
        # CAN bus monitoring section
        can_group = QGroupBox("CAN Bus Status")
        can_layout = QHBoxLayout(can_group)
        
        self.can_error_plot = CustomPlotWidget("CAN Error Rate", self)
        # Initialize CAN error plot items
        for i in range(2):
            self.plot_items[f'can_error_{i}'] = self.can_error_plot.plot(
                [], [],
                pen=pg.intColor(i, 2),
                name=f"CAN {i+1}"
            )
        can_layout.addWidget(self.can_error_plot)
        
        self.content_layout.addWidget(can_group)
        
    def update_display(self, sensor_data):
        """Update system health displays"""
        if not sensor_data or not hasattr(sensor_data, 'system_health'):
            return
            
        try:
            health = sensor_data.system_health
            current_time = np.array([sensor_data.timestamp], dtype=np.float32)
            
            # Update CPU loads
            for i, load in enumerate(health.cpu_load):
                self.plot_items[f'cpu_{i}'].setData(
                    x=current_time,
                    y=np.array([load], dtype=np.float32)
                )
                
            # Update motor temperatures
            self.plot_items['motor_left'].setData(
                x=current_time,
                y=np.array([sensor_data.motor_left.temperature], dtype=np.float32)
            )
            self.plot_items['motor_right'].setData(
                x=current_time,
                y=np.array([sensor_data.motor_right.temperature], dtype=np.float32)
            )
            
            # Update board temperatures
            for i, temp in enumerate(health.board_temps):
                self.plot_items[f'board_temp_{i}'].setData(
                    x=current_time,
                    y=np.array([temp], dtype=np.float32)
                )
                
            # Update CAN error plots - handle each error count separately
            for i, error_count in enumerate(health.can_errors):
                self.plot_items[f'can_error_{i}'].setData(
                    x=current_time,
                    y=np.array([error_count], dtype=np.float32)
                )
                
        except Exception as e:
            print(f"Error updating health display: {str(e)}")

    def reset_display(self):
        """Reset all plot data"""
        try:
            for plot_item in self.plot_items.values():
                plot_item.setData([], [])
        except Exception as e:
            print(f"Error resseting display: {str(e)}")


class PerformanceTab(TabWidget):
    """Tab for analyzing race performance metrics and trends"""
    
    def setup_ui(self):
        # Top section for lap timing and speed profile
        timing_group = QGroupBox("Lap Performance")
        timing_layout = QHBoxLayout(timing_group)
        
        # Lap time display and history
        self.lap_time_plot = CustomPlotWidget("Lap Times", self)
        self.speed_profile_plot = CustomPlotWidget("Speed Profile", self)
        
        timing_layout.addWidget(self.lap_time_plot)
        timing_layout.addWidget(self.speed_profile_plot)
        self.content_layout.addWidget(timing_group)
        
        # Middle section for acceleration and steering analysis
        dynamics_group = QGroupBox("Vehicle Dynamics")
        dynamics_layout = QGridLayout(dynamics_group)
        
        self.accel_plot = CustomPlotWidget("Acceleration Profile", self)
        self.steering_response_plot = CustomPlotWidget("Steering Response", self)
        
        dynamics_layout.addWidget(self.accel_plot, 0, 0)
        dynamics_layout.addWidget(self.steering_response_plot, 0, 1)
        self.content_layout.addWidget(dynamics_group)
        
        # Bottom section for track detection and system performance
        analysis_group = QGroupBox("System Analysis")
        analysis_layout = QGridLayout(analysis_group)
        
        self.track_detection_plot = CustomPlotWidget("Track Detection Confidence", self)
        self.power_efficiency_plot = CustomPlotWidget("Power Efficiency", self)
        
        analysis_layout.addWidget(self.track_detection_plot, 0, 0)
        analysis_layout.addWidget(self.power_efficiency_plot, 0, 1)
        self.content_layout.addWidget(analysis_group)
        
        # Initialize data storage for analysis
        self.lap_times = []
        self.speed_data = []
        self.last_lap_start = None

        
    def update_display(self, sensor_data):
        """Update performance analysis displays"""
        if not sensor_data:
            return
            
        # Update lap timing if we detect a lap completion
        # This is a simplified example - you'll need to implement actual lap detection
        if self._check_lap_completion(sensor_data):
            if self.last_lap_start is not None:
                lap_time = sensor_data.timestamp - self.last_lap_start
                self.lap_times.append(lap_time)
                
                # Update lap time plot
                self.lap_time_plot.plot(
                    range(len(self.lap_times)),
                    self.lap_times,
                    pen='b',
                    symbol='o',
                    clear=True
                )
            
            self.last_lap_start = sensor_data.timestamp
        
        # Update speed profile
        self.speed_profile_plot.plot(
            [sensor_data.timestamp],
            [sensor_data.speed],
            pen='g',
            clear=True
        )
        
        # Calculate acceleration from speed changes
        if len(self.speed_data) > 1:
            dt = sensor_data.timestamp - self.speed_data[-1][0]
            dv = sensor_data.speed - self.speed_data[-1][1]
            acceleration = dv / dt if dt > 0 else 0
            
            self.accel_plot.plot(
                [sensor_data.timestamp],
                [acceleration],
                pen='r',
                clear=True
            )
        
        self.speed_data.append((sensor_data.timestamp, sensor_data.speed))
        
        # Update steering response plot
        self.steering_response_plot.plot(
            [sensor_data.timestamp],
            [sensor_data.steering_angle],
            pen='y',
            clear=True
        )
        
        # Update track detection confidence
        self.track_detection_plot.plot(
            [sensor_data.timestamp],
            [sensor_data.track_confidence],
            pen='c',
            clear=True
        )
        
        # Calculate and plot power efficiency
        power_efficiency = (sensor_data.speed * 100) / max(0.1, sensor_data.battery_power)
        self.power_efficiency_plot.plot(
            [sensor_data.timestamp],
            [power_efficiency],
            pen='m',
            clear=True
        )
        
    def _check_lap_completion(self, sensor_data):
        """Detect if a lap has been completed based on position and heading"""
        # This is a placeholder - implement actual lap detection logic
        # Could use position near start/finish line and correct heading
        return False

class DebugConsoleTab(TabWidget):
    """Tab for technical debugging and detailed system information"""
    
    def setup_ui(self):
        # Create split view for log display and control panel
        content_split = QHBoxLayout()
        self.content_layout.addLayout(content_split)
        
        # Left side - Log display and filtering
        log_panel = QVBoxLayout()
        
        # Filter controls
        filter_group = QGroupBox("Log Filters")
        filter_layout = QHBoxLayout(filter_group)
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter logs...")
        self.filter_input.textChanged.connect(self._apply_filter)
        
        self.level_combo = QComboBox()
        self.level_combo.addItems(["All", "Info", "Warning", "Error"])
        self.level_combo.currentTextChanged.connect(self._apply_filter)
        
        filter_layout.addWidget(self.filter_input)
        filter_layout.addWidget(self.level_combo)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        
        log_panel_widget = QWidget()
        log_panel_widget.setLayout(log_panel)
        log_panel.addWidget(filter_group)
        log_panel.addWidget(self.log_text)
        
        # Right side - System state and controls
        state_panel = QVBoxLayout()
        
        # PID state visualization
        pid_group = QGroupBox("PID Controllers")
        pid_layout = QGridLayout(pid_group)
        
        self.pid_displays = {}
        pid_names = ["Speed", "Steering", "Stability"]
        for i, name in enumerate(pid_names):
            display = QLabel()
            display.setStyleSheet("border: 1px solid gray; padding: 5px;")
            self.pid_displays[name] = display
            pid_layout.addWidget(QLabel(f"{name}:"), i, 0)
            pid_layout.addWidget(display, i, 1)
        
        # Control mode display
        mode_group = QGroupBox("Control Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_label = QLabel()
        mode_layout.addWidget(self.mode_label)
        
        # Error counter display
        error_group = QGroupBox("Error Counters")
        error_layout = QGridLayout(error_group)
        
        self.error_counters = {}
        error_types = ["CAN", "Sensor", "Control", "System"]
        for i, error_type in enumerate(error_types):
            counter = QLabel("0")
            counter.setStyleSheet("color: red;")
            self.error_counters[error_type] = counter
            error_layout.addWidget(QLabel(f"{error_type}:"), i, 0)
            error_layout.addWidget(counter, i, 1)
        
        state_panel_widget = QWidget()
        state_panel_widget.setLayout(state_panel)
        state_panel.addWidget(pid_group)
        state_panel.addWidget(mode_group)
        state_panel.addWidget(error_group)
        
        # Add panels to content split
        content_split.addWidget(log_panel_widget, stretch=2)
        content_split.addWidget(state_panel_widget, stretch=1)
        
        # Initialize log buffer and counters
        self.log_buffer = []
        self.error_counts = {error_type: 0 for error_type in error_types}
        
    def update_display(self, sensor_data):
        """Update debug information display"""
        if not sensor_data:
            return
            
        # Update PID state displays
        pid_state = sensor_data.system_health.pid_state
        for name, display in self.pid_displays.items():
            # Extract relevant bits for each PID controller
            # This is a placeholder - implement actual bit parsing
            state = "Active" if pid_state & 1 else "Inactive"
            display.setText(state)
            pid_state >>= 1
        
        # Update control mode
        mode_names = {
            0: "Manual",
            1: "Autonomous",
            2: "Safety",
            3: "Calibration"
        }
        mode = mode_names.get(sensor_data.system_health.control_mode, "Unknown")
        self.mode_label.setText(f"Mode: {mode}")
        
        # Log any new errors or warnings
        self._check_and_log_errors(sensor_data)
        
    def _check_and_log_errors(self, sensor_data):
        """Check for and log any error conditions"""
        timestamp = datetime.fromtimestamp(sensor_data.timestamp).strftime('%H:%M:%S.%f')[:-3]
        
        # Check various error conditions
        if sensor_data.error_flags:
            self._add_log(f"{timestamp} [ERROR] System error flags: {sensor_data.error_flags}", "Error")
            self.error_counts["System"] += 1
            
        if any(err > 0 for err in sensor_data.system_health.can_errors):
            self._add_log(f"{timestamp} [ERROR] CAN bus errors detected", "Error")
            self.error_counts["CAN"] += 1
            
        # Update error counter displays
        for error_type, count in self.error_counts.items():
            self.error_counters[error_type].setText(str(count))
            
    def _add_log(self, message: str, level: str):
        """Add a message to the log buffer"""
        self.log_buffer.append((level, message))
        if len(self.log_buffer) > 1000:  # Limit buffer size
            self.log_buffer.pop(0)
        self._apply_filter()
        
    def _apply_filter(self):
        """Apply current filters to log display"""
        filter_text = self.filter_input.text().lower()
        level_filter = self.level_combo.currentText()
        
        filtered_logs = []
        for level, message in self.log_buffer:
            if level_filter != "All" and level != level_filter:
                continue
            if filter_text and filter_text not in message.lower():
                continue
            filtered_logs.append(message)
        
        self.log_text.setText("\n".join(filtered_logs))
        
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class MainWindow(QMainWindow):
    def __init__(self, config_manager, data_handler=None, track_visualizer=None):
        """Initialize the main window with all necessary components."""
        super().__init__()
        
        # Set up logging first
        self.logger = logging.getLogger('race_ground_station.gui')
        self.logger.debug("Initializing MainWindow")
        
        # Store configuration and components
        self.config = config_manager
        self.data_handler = data_handler
        self.track_visualizer = track_visualizer
        
        # Initialize plot manager
        self.plot_manager = PlotManager(self)
        
        # Initialize UI
        self.setup_window()
        self.create_tabs()
        self.setup_toolbar()
        self.setup_statusbar()
        
        # Initialize timers
        self.setup_timers()
        
        # Connect signals LAST after everything is set up
        self.connect_signals()
        
        self.logger.debug("MainWindow initialization complete")

    def create_tabs(self):
        """Create application tabs with new plotting widgets"""
        self.tabs = {}

        # Create tab container
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        # Create specialized plotting widgets
        self.tabs['race'] = LivePlotWidget(self.plot_manager, self)
        self.tabs['track'] = TrackVisualizerWidget(self.plot_manager, self)
        self.tabs['health'] = SystemHealthTab(self.config, self.plot_manager, self)
        self.tabs['performance'] = PerformanceTab(self.config, self.plot_manager, self)
        self.tabs['debug'] = DebugConsoleTab(self.config, self)
        
        # Add tabs
        self.tab_widget.addTab(self.tabs['race'], "Race View")
        self.tab_widget.addTab(self.tabs['track'], "Track View")
        self.tab_widget.addTab(self.tabs['health'], "System Health")
        self.tab_widget.addTab(self.tabs['performance'], "Performance")
        self.tab_widget.addTab(self.tabs['debug'], "Debug Console")

    def connect_signals(self):
        """Connect all signals between components"""
        # Data handler signals
        self.data_handler.data_ready.connect(self.handle_new_data)
        self.data_handler.error_occurred.connect(self.handle_error)
        self.data_handler.connection_status.connect(self.update_connection_status)
        self.data_handler.playback_finished.connect(self.handle_playback_finished)

        # Plot manager signals
        self.plot_manager.error_occurred.connect(self.handle_error)

        # Track visualizer signals
        if self.track_visualizer:
            self.track_visualizer.error_occurred.connect(self.handle_error)

        self.logger.debug("Signal connections established")

    def closeEvent(self, event):
        """Handle application shutdown with proper thread cleanup"""
        self.logger.debug("Starting MainWindow cleanup")
        try:
            # Stop all timers first
            if hasattr(self, 'update_timer'):
                self.logger.debug("Stopping update timer")
                self.update_timer.stop()
            if hasattr(self, 'status_timer'):
                self.logger.debug("Stopping status timer")
                self.status_timer.stop()
            
            # Stop data handler and wait for threads
            if hasattr(self, 'data_handler'):
                self.logger.debug("Stopping data handler")
                self.data_handler.stop()
                # Give threads time to stop
                QThread.msleep(100)
            
            # Clean up plot manager
            if hasattr(self, 'plot_manager'):
                self.logger.debug("Cleaning up plot manager")
                self.plot_manager.cleanup()
            
            # Clean up tab widgets
            if hasattr(self, 'tabs'):
                self.logger.debug("Cleaning up tabs")
                for tab in self.tabs.values():
                    if hasattr(tab, 'cleanup'):
                        tab.cleanup()
            
            # Save configuration
            self.logger.debug("Saving configuration")
            self.config.save_config()
            
            self.logger.info("MainWindow cleanup completed successfully")
            event.accept()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            event.accept()  # Accept anyway to ensure application can close


    def handle_new_data(self, sensor_data):
        """Process new sensor data updates"""
        try:
            # Update all active tabs
            if hasattr(self, 'tabs'):
                # Update current tab
                current_tab = self.tab_widget.currentWidget()
                if hasattr(current_tab, 'update_display'):
                    current_tab.update_display(sensor_data)

            # Update status bar with latest data
            self.update_status_bar(sensor_data)

        except Exception as e:
            self.handle_error(f"Error handling new data: {str(e)}")

    def setup_timers(self):
        """Configure update timers"""
        # Status update timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_status)
        self.status_timer.start(1000)  # Check status every second

    def update_displays(self):
        """Update all display elements with current data"""
        try:
            current_tab = self.central_widget.currentWidget()
            latest_data = self.data_handler.get_latest_data()
            
            if latest_data:
                # Update current tab
                if hasattr(current_tab, 'update_display'):
                    current_tab.update_display(latest_data)
                
                # Update track visualizer
                self.track_visualizer.update_vehicle_state(latest_data)
                
                # Always update status bar
                self.update_status_bar(latest_data)
        except Exception as e:
            self.handle_error(f"Error updating displays: {str(e)}")

    def calculate_metrics(self):
        """Calculate and update performance metrics"""
        try:
            latest_data = self.data_handler.get_latest_data()
            if not latest_data:
                return
                
            # Get raw data from buffers
            speed_values = self.data_handler.get_buffer_data('speed')[1]
            accel_values = self.data_handler.get_buffer_data('accel_y')[1]
            
            # Calculate metrics
            metrics = {
                'avg_speed': float(np.mean(speed_values)) if len(speed_values) > 0 else 0.0,
                'max_accel': float(np.max(np.abs(accel_values))) if len(accel_values) > 0 else 0.0,
                'power_usage': float(latest_data.battery_voltage * latest_data.battery_current)
            }
            
            # Update performance tab if it's visible
            if self.central_widget.currentWidget() == self.tabs['performance']:
                self.tabs['performance'].update_metrics(metrics)
                
        except Exception as e:
            self.handle_error(f"Error calculating metrics: {str(e)}")

    def show_settings(self):
        """Show the settings dialog and handle port configuration"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            # Stop current data collection if any
            self.data_handler.stop()

            # Get new port settings
            port = self.config.get("serial", "default_port")
            baudrate = self.config.get("serial", "baudrate")

            # Restart data collection with new settings
            self.data_handler.start(port, baudrate)

            # Apply other settings
            self.update_timer.setInterval(self.config.get("display", "update_interval_ms"))
            for tab in self.tabs.values():
                tab.apply_theme()

    def setup_window(self):
        """Initialize the main window properties"""
        self.setWindowTitle("Race Car Ground Station")
        self.setGeometry(100, 100, 1600, 900)  # Larger default size for race display
        
        # Create central widget
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
    def setup_toolbar(self):
            """Create and configure the main toolbar"""
            self.toolbar = QToolBar()
            self.toolbar.setMovable(False)
            self.addToolBar(Qt.TopToolBarArea, self.toolbar)
            
            # Create toolbar buttons
            button_style = """
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:pressed {
                    background-color: #303030;
                }
            """
            
            # Define toolbar buttons - only keep utility buttons
            toolbar_buttons = [
                ("Record Data", self.toggle_recording),
                ("Playback Data", self.show_playback_dialog),
                ("Settings", self.show_settings)
            ]
            
            # Create and add buttons to toolbar
            for text, handler in toolbar_buttons:
                button = QPushButton(text)
                button.setStyleSheet(button_style)
                button.clicked.connect(handler)
                self.toolbar.addWidget(button)
            
            # Add recording indicator
            self.recording_indicator = QLabel("")
            self.recording_indicator.setStyleSheet("color: red; font-weight: bold;")
            self.toolbar.addWidget(self.recording_indicator)
        
    def setup_statusbar(self):
        """Initialize the status bar with system indicators"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Create status indicators
        self.connection_indicator = QLabel("⚫ Disconnected")
        self.data_status = QLabel("No Data")
        self.battery_status = QLabel("Battery: --V")
        self.mode_status = QLabel("Mode: --")
        
        # Add permanent widgets to status bar
        for widget in [self.connection_indicator, self.data_status, 
                      self.battery_status, self.mode_status]:
            widget.setStyleSheet("padding: 0 8px;")
            self.statusbar.addPermanentWidget(widget)
        
    def setup_shortcuts(self):
        """Configure keyboard shortcuts for quick access to features"""
        shortcuts = {
            'R': lambda: self.switch_tab('race'),
            'H': lambda: self.switch_tab('health'),
            'P': lambda: self.switch_tab('performance'),
            'D': lambda: self.switch_tab('debug'),
            'Space': self.toggle_recording,
            'Esc': self.data_handler.stop
        }
        
        for key, handler in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(handler)

    def switch_tab(self, tab_name: str):
        """Switch to the specified tab and update its content"""
        if tab_name in self.tab_indices:
            self.central_widget.setCurrentIndex(self.tab_indices[tab_name])
            # Force an immediate update of the new tab
            self.update_displays()

    def update_status_bar(self, sensor_data):
        """Update status bar indicators with current system state"""
        if not sensor_data:
            return
            
        # Update battery status with color coding
        voltage = sensor_data.battery_voltage
        if voltage < 14.0:
            color = "red"
        elif voltage < 14.5:
            color = "orange"
        else:
            color = "green"
        self.battery_status.setText(
            f"Battery: <span style='color: {color}'>{voltage:.1f}V</span>"
        )
        
        # Update mode status
        mode_names = {
            0: "Manual",
            1: "Autonomous",
            2: "Safety",
            3: "Calibration"
        }
        mode = mode_names.get(sensor_data.system_health.control_mode, "Unknown")
        self.mode_status.setText(f"Mode: {mode}")
        
        # Update data status with timestamp
        time_str = datetime.fromtimestamp(sensor_data.timestamp).strftime('%H:%M:%S.%f')[:-3]
        self.data_status.setText(f"Last Update: {time_str}")

    def check_status(self):
        """Periodic check of system status and health"""
        latest_data = self.data_handler.get_latest_data()
        
        if latest_data:
            # Check data freshness
            current_time = time.time()
            time_since_update = current_time - latest_data.timestamp
            
            if time_since_update > 1.0:  # Data is stale
                self.data_status.setStyleSheet("color: red")
            else:
                self.data_status.setStyleSheet("color: green")
            
            # Check for any error flags
            if latest_data.error_flags:
                self.handle_error(f"System error flags: {latest_data.error_flags}")


    def handle_error(self, error_message: str):
        """Handle and display error messages"""
        # Show error in status bar
        self.statusbar.showMessage(f"Error: {error_message}", 5000)
        
        # Add to debug console if it exists
        if 'debug' in self.tabs:
            self.tabs['debug']._add_log(error_message, "Error")

    def update_connection_status(self, connected: bool):
        """Update connection status indicator"""
        if connected:
            self.connection_indicator.setText("🟢 Connected")
            self.connection_indicator.setStyleSheet("color: green")
        else:
            self.connection_indicator.setText("⚫ Disconnected")
            self.connection_indicator.setStyleSheet("color: red")

    def toggle_recording(self):
        """Toggle data recording state"""
        if not hasattr(self, 'recording'):
            self.recording = False
        
        self.recording = not self.recording
        
        if self.recording:
            # Start new recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_file = os.path.join(
                os.path.expanduser(self.config.get("data_logging", "directory")),
                f"race_log_{timestamp}.csv"
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.record_file), exist_ok=True)
            
            # Write header row
            with open(self.record_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header based on sensor data structure
                writer.writerow(['timestamp', 'speed', 'steering_angle', ...])  # Add all relevant fields
                
            self.recording_indicator.setText("⚫ Recording")
            
        else:
            # Stop recording
            self.recording_indicator.setText("")
            self.record_file = None

    def _record_data(self, sensor_data):
        """Record sensor data to CSV file"""
        if not hasattr(self, 'record_file') or not self.record_file:
            return
            
        try:
            with open(self.record_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # Write all relevant fields from sensor data
                writer.writerow([
                    sensor_data.timestamp,
                    sensor_data.speed,
                    sensor_data.steering_angle,
                    # Add all other relevant fields
                ])
                
        except Exception as e:
            self.handle_error(f"Recording error: {str(e)}")
            self.toggle_recording()  # Stop recording on error

    def _check_alerts(self, sensor_data):
        """Check for and handle alert conditions"""
        alerts = []
        
        # Check battery voltage
        if sensor_data.battery_voltage < 14.0:
            alerts.append("Low Battery Voltage")
            
        # Check motor temperature
        if sensor_data.motor_left.temperature > 80 or sensor_data.motor_right.temperature > 80:
            alerts.append("High Motor Temperature")
            
        # Check track detection
        if sensor_data.track_confidence < 50:
            alerts.append("Low Track Detection Confidence")
            
        # Display alerts if any
        if alerts:
            alert_text = " | ".join(alerts)
            self.statusbar.showMessage(alert_text, 2000)

    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            # Refresh the UI with new settings
            self.update_timer.setInterval(self.config.get("display", "update_interval_ms"))
            for tab in self.tabs.values():
                tab.apply_theme()

    def show_playback_dialog(self):
        """Show dialog for CSV playback control"""
        dialog = PlaybackDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            if settings['file_path']:
                self.data_handler.start_playback(
                    settings['file_path'],
                    settings['speed']
                )

    def handle_playback_finished(self):
        """Handle completion of data playback"""
        self.statusbar.showMessage("Playback finished", 2000)
        

class SettingsDialog(QDialog):
    """Dialog for configuring application settings"""
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("Ground Station Settings")
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Add settings tabs
        self.setup_communication_tab()
        self.setup_display_tab()
        self.setup_logging_tab()
        self.setup_alerts_tab()
        
        # Add button box
        button_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        
        save_btn.clicked.connect(self.save_settings)
        cancel_btn.clicked.connect(self.reject)
        
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)

    def setup_communication_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        
        self.port_input = QLineEdit(self.config.get("serial", "default_port"))
        layout.addRow("Serial Port:", self.port_input)
        
        self.baudrate_input = QSpinBox()
        self.baudrate_input.setRange(9600, 921600)
        self.baudrate_input.setValue(self.config.get("serial", "baudrate"))
        layout.addRow("Baud Rate:", self.baudrate_input)
        
        self.tabs.addTab(tab, "Communication")

    def setup_display_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        
        self.dark_mode_check = QCheckBox()
        self.dark_mode_check.setChecked(self.config.get("display", "dark_mode"))
        layout.addRow("Dark Mode:", self.dark_mode_check)
        
        self.update_interval = QSpinBox()
        self.update_interval.setRange(16, 1000)
        self.update_interval.setValue(self.config.get("display", "update_interval_ms"))
        layout.addRow("Update Interval (ms):", self.update_interval)
        
        self.tabs.addTab(tab, "Display")

    def setup_logging_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        
        self.logging_enabled = QCheckBox()
        self.logging_enabled.setChecked(self.config.get("data_logging", "enabled"))
        layout.addRow("Enable Logging:", self.logging_enabled)
        
        self.log_dir = QLineEdit(self.config.get("data_logging", "directory"))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_log_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.log_dir)
        dir_layout.addWidget(browse_btn)
        layout.addRow("Log Directory:", dir_layout)
        
        self.tabs.addTab(tab, "Logging")

    def setup_alerts_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        
        self.battery_threshold = QDoubleSpinBox()
        self.battery_threshold.setRange(12.0, 16.8)
        self.battery_threshold.setValue(self.config.get("alerts", "battery_threshold"))
        layout.addRow("Battery Alert (V):", self.battery_threshold)
        
        self.temp_threshold = QSpinBox()
        self.temp_threshold.setRange(50, 100)
        self.temp_threshold.setValue(self.config.get("alerts", "temperature_threshold"))
        layout.addRow("Temperature Alert (°C):", self.temp_threshold)
        
        self.tabs.addTab(tab, "Alerts")

    def _browse_log_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Log Directory",
            self.log_dir.text()
        )
        if directory:
            self.log_dir.setText(directory)

    def validate_port(self, port: str) -> bool:
        """Validate the selected serial port"""
        if os.name == 'nt':  # Windows
            import serial.tools.list_ports
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            return port in available_ports
        else:  # Unix-like
            return os.path.exists(port) or port.startswith('/dev/tty')

    def save_settings(self):
        # Validate port before saving
        port = self.port_input.text()
        if not self.validate_port(port):
            QMessageBox.warning(
                self,
                "Invalid Port",
                f"The port {port} does not exist or is not accessible.\n"
                "Please check your connection and port settings."
            )
            return
        
        # Save communication settings
        self.config.set("serial", "default_port", self.port_input.text())
        self.config.set("serial", "baudrate", self.baudrate_input.value())
        
        # Save display settings
        self.config.set("display", "dark_mode", self.dark_mode_check.isChecked())
        self.config.set("display", "update_interval_ms", self.update_interval.value())
        
        # Save logging settings
        self.config.set("data_logging", "enabled", self.logging_enabled.isChecked())
        self.config.set("data_logging", "directory", self.log_dir.text())
        
        # Save alert settings
        self.config.set("alerts", "battery_threshold", self.battery_threshold.value())
        self.config.set("alerts", "temperature_threshold", self.temp_threshold.value())
        
        self.config.save_config()
        self.accept()
        
    def setup_ui(self):
        """Create and arrange the settings interface"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for categorized settings
        tabs = QTabWidget()
        
        # Communication Settings Tab
        comm_tab = QWidget()
        comm_layout = QFormLayout(comm_tab)
        
        # Serial configuration
        self.port_input = QLineEdit(self.config.get("serial", "default_port"))
        comm_layout.addRow("Serial Port:", self.port_input)
        
        self.baudrate_input = QSpinBox()
        self.baudrate_input.setRange(9600, 921600)
        self.baudrate_input.setValue(self.config.get("serial", "baudrate"))
        self.baudrate_input.setSingleStep(9600)
        comm_layout.addRow("Baud Rate:", self.baudrate_input)
        
        # Add timeout setting
        self.timeout_input = QDoubleSpinBox()
        self.timeout_input.setRange(0.1, 5.0)
        self.timeout_input.setValue(self.config.get("serial", "timeout"))
        self.timeout_input.setSingleStep(0.1)
        comm_layout.addRow("Timeout (seconds):", self.timeout_input)
        
        tabs.addTab(comm_tab, "Communication")
        
        # Display Settings Tab
        display_tab = QWidget()
        display_layout = QFormLayout(display_tab)
        
        # Theme selection
        self.dark_mode_check = QCheckBox()
        self.dark_mode_check.setChecked(self.config.get("display", "dark_mode"))
        display_layout.addRow("Dark Mode:", self.dark_mode_check)
        
        # Update intervals
        self.update_interval = QSpinBox()
        self.update_interval.setRange(16, 1000)  # 60 FPS max
        self.update_interval.setValue(self.config.get("display", "update_interval_ms"))
        display_layout.addRow("Display Update Interval (ms):", self.update_interval)
        
        # Buffer size for plots
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(100, 10000)
        self.buffer_size.setValue(self.config.get("display", "plot_buffer_size"))
        self.buffer_size.setSingleStep(100)
        display_layout.addRow("Plot Buffer Size:", self.buffer_size)
        
        tabs.addTab(display_tab, "Display")
        
        # Logging Settings Tab
        logging_tab = QWidget()
        logging_layout = QFormLayout(logging_tab)
        
        # Enable logging
        self.logging_enabled = QCheckBox()
        self.logging_enabled.setChecked(self.config.get("data_logging", "enabled"))
        logging_layout.addRow("Enable Data Logging:", self.logging_enabled)
        
        # Log directory
        log_dir_layout = QHBoxLayout()
        self.log_dir = QLineEdit(self.config.get("data_logging", "directory"))
        self.log_dir_button = QPushButton("Browse...")
        self.log_dir_button.clicked.connect(self._browse_log_dir)
        log_dir_layout.addWidget(self.log_dir)
        log_dir_layout.addWidget(self.log_dir_button)
        logging_layout.addRow("Log Directory:", log_dir_layout)
        
        # Maximum log file size
        self.max_log_size = QSpinBox()
        self.max_log_size.setRange(10, 1000)
        self.max_log_size.setValue(self.config.get("data_logging", "max_log_size_mb"))
        self.max_log_size.setSuffix(" MB")
        logging_layout.addRow("Maximum Log File Size:", self.max_log_size)
        
        tabs.addTab(logging_tab, "Logging")
        
        # Alert Settings Tab
        alerts_tab = QWidget()
        alerts_layout = QFormLayout(alerts_tab)
        
        # Battery voltage alert threshold
        self.battery_threshold = QDoubleSpinBox()
        self.battery_threshold.setRange(12.0, 16.8)
        self.battery_threshold.setValue(self.config.get("alerts", "battery_threshold", 14.0))
        self.battery_threshold.setSingleStep(0.1)
        alerts_layout.addRow("Low Battery Alert Threshold (V):", self.battery_threshold)
        
        # Motor temperature alert threshold
        self.temp_threshold = QSpinBox()
        self.temp_threshold.setRange(50, 100)
        self.temp_threshold.setValue(self.config.get("alerts", "temperature_threshold", 80))
        self.temp_threshold.setSuffix(" °C")
        alerts_layout.addRow("High Temperature Alert Threshold:", self.temp_threshold)
        
        # Track detection confidence threshold
        self.track_confidence = QSpinBox()
        self.track_confidence.setRange(0, 100)
        self.track_confidence.setValue(self.config.get("alerts", "track_confidence_threshold", 50))
        self.track_confidence.setSuffix("%")
        alerts_layout.addRow("Track Detection Alert Threshold:", self.track_confidence)
        
        tabs.addTab(alerts_tab, "Alerts")
        
        layout.addWidget(tabs)
        
        # Add buttons
        button_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        
        save_btn.clicked.connect(self.save_settings)
        cancel_btn.clicked.connect(self.reject)
        
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)
        
    def _browse_log_dir(self):
        """Open directory browser for log location"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Log Directory",
            self.log_dir.text(),
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.log_dir.setText(directory)
            
    def save_settings(self):
        """Save all settings to configuration"""
        # Communication settings
        self.config.set("serial", "default_port", self.port_input.text())
        self.config.set("serial", "baudrate", self.baudrate_input.value())
        self.config.set("serial", "timeout", self.timeout_input.value())
        
        # Display settings
        self.config.set("display", "dark_mode", self.dark_mode_check.isChecked())
        self.config.set("display", "update_interval_ms", self.update_interval.value())
        self.config.set("display", "plot_buffer_size", self.buffer_size.value())
        
        # Logging settings
        self.config.set("data_logging", "enabled", self.logging_enabled.isChecked())
        self.config.set("data_logging", "directory", self.log_dir.text())
        self.config.set("data_logging", "max_log_size_mb", self.max_log_size.value())
        
        # Alert settings
        self.config.set("alerts", "battery_threshold", self.battery_threshold.value())
        self.config.set("alerts", "temperature_threshold", self.temp_threshold.value())
        self.config.set("alerts", "track_confidence_threshold", self.track_confidence.value())
        
        self.config.save_config()
        self.accept()

class PlaybackDialog(QDialog):
    """Dialog for controlling data playback from recorded files"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Playback Control")
        self.setup_ui()
        
    def setup_ui(self):
        """Create and arrange the playback control interface"""
        layout = QVBoxLayout(self)
        
        # File selection section
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout(file_group)
        
        # File selection controls
        file_input_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select recorded data file...")
        self.file_path.setMinimumWidth(400)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        
        file_input_layout.addWidget(self.file_path)
        file_input_layout.addWidget(self.browse_btn)
        file_layout.addLayout(file_input_layout)
        
        # Add file info display
        self.file_info = QLabel()
        self.file_info.setWordWrap(True)
        file_layout.addWidget(self.file_info)
        
        layout.addWidget(file_group)
        
        # Playback settings section
        settings_group = QGroupBox("Playback Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Playback speed control
        self.speed_selector = QComboBox()
        speeds = ['0.25x', '0.5x', '1.0x', '2.0x', '4.0x', '8.0x']
        self.speed_selector.addItems(speeds)
        self.speed_selector.setCurrentText('1.0x')
        settings_layout.addRow("Playback Speed:", self.speed_selector)
        
        # Loop playback option
        self.loop_playback = QCheckBox()
        settings_layout.addRow("Loop Playback:", self.loop_playback)
        
        layout.addWidget(settings_group)
        
        # Progress display
        self.progress_group = QGroupBox("Playback Progress")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setEnabled(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.time_label = QLabel("Duration: --:--:--")
        progress_layout.addWidget(self.time_label)
        
        layout.addWidget(self.progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Playback")
        self.start_btn.clicked.connect(self.accept)
        self.start_btn.setEnabled(False)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Set dialog size
        self.setMinimumWidth(600)
        
    def browse_file(self):
        """Open file browser for selecting recorded data"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Recorded Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_name:
            self.file_path.setText(file_name)
            self._analyze_file(file_name)
            
    def _analyze_file(self, file_path):
        """Analyze selected file and update interface"""
        try:
            # Get file size
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            
            # Read first and last lines to determine duration
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # Seek to the end and then backward to find the last line
                f.seek(0, os.SEEK_END)
                pos = f.tell() - 2
                while pos > 0 and f.read(1) != b"\n":
                    pos -= 1
                    f.seek(pos, os.SEEK_SET)
                last_line = f.readline().strip()
                
            # Parse timestamps
            try:
                start_time = float(first_line.split(',')[0])
                end_time = float(last_line.split(',')[0])
                duration = end_time - start_time
                
                hours = int(duration / 3600)
                minutes = int((duration % 3600) / 60)
                seconds = int(duration % 60)
                
                self.time_label.setText(
                    f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}"
                )
                
            except (ValueError, IndexError):
                self.time_label.setText("Duration: Unknown")
                
            # Update file info
            self.file_info.setText(
                f"File Size: {size_mb:.1f} MB\n"
                f"Path: {file_path}"
            )
            
            # Enable playback controls
            self.start_btn.setEnabled(True)
            self.progress_bar.setEnabled(True)
            
        except Exception as e:
            self.file_info.setText(f"Error analyzing file: {str(e)}")
            self.start_btn.setEnabled(False)
            self.progress_bar.setEnabled(False)
            
    def get_settings(self):
        """Get the current playback settings"""
        return {
            'file_path': self.file_path.text(),
            'speed': float(self.speed_selector.currentText().replace('x', '')),
            'loop': self.loop_playback.isChecked()
        }