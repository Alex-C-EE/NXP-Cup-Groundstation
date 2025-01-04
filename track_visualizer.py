# track_visualizer.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from PyQt5.QtCore import QObject, pyqtSignal, Qt  # Added Qt import
import pyqtgraph as pg
from datetime import datetime
import math

@dataclass
class TrackPoint:
    """Represents a point on the race track with detection confidence"""
    x: float  # X coordinate relative to start position
    y: float  # Y coordinate relative to start position
    confidence: float  # Detection confidence (0-1)
    timestamp: float  # When this point was detected

@dataclass
class VehicleState:
    """Represents the current state of the vehicle for visualization"""
    x: float  # X coordinate
    y: float  # Y coordinate
    heading: float  # Vehicle heading in degrees
    speed: float  # Current speed in m/s
    steering_angle: float  # Current steering angle in degrees
    timestamp: float  # State timestamp

class TrackVisualizer(QObject):
    """Handles visualization of the race track and vehicle position"""
    
    # Signal emitted when visualization updates are ready
    visualization_updated = pyqtSignal(object)  # Emits dictionary of plot items
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config_manager):
        """Initialize the track visualization system"""
        super().__init__()
        self.config = config_manager
        
        # Track visualization settings
        self.track_memory_time = 30.0  # How many seconds of track to remember
        self.confidence_threshold = 0.5  # Minimum confidence for track points
        self.max_track_points = 1000  # Maximum number of track points to store
        
        # Initialize storage for track and vehicle data
        self.track_points: List[TrackPoint] = []
        self.vehicle_states: List[VehicleState] = []
        self.current_state: Optional[VehicleState] = None
        
        # Track boundary visualization
        self.left_boundary: List[TrackPoint] = []
        self.right_boundary: List[TrackPoint] = []
        
        # Performance optimization settings
        self.update_decimation = 3  # Only update visualization every N points
        self.update_counter = 0
        
        # Initialize plot items
        self._init_plot_items()

    def _init_plot_items(self):
        """Initialize PyQtGraph items for visualization"""
        # Initialize all plot items as None first
        self.vehicle_marker = None
        self.left_boundary_plot = None
        self.right_boundary_plot = None
        self.path_plot = None
        self.predicted_path_plot = None
        
        # Create plot items only when needed
        self._create_plot_items()

    def _create_plot_items(self):
        """Create plot items if they don't exist"""
        if self.vehicle_marker is None:
            self.vehicle_marker = pg.ArrowItem(
                angle=0, 
                tipAngle=30,
                headLen=20, 
                tailLen=0,
                pen={'color': 'r', 'width': 2},
                brush='r'
            )

        if self.left_boundary_plot is None:
            self.left_boundary_plot = pg.PlotDataItem(
                pen={'color': 'b', 'width': 2},
                symbol='o',
                symbolSize=4,
                symbolBrush='b'
            )

        if self.right_boundary_plot is None:
            self.right_boundary_plot = pg.PlotDataItem(
                pen={'color': 'g', 'width': 2},
                symbol='o',
                symbolSize=4,
                symbolBrush='g'
            )

        if self.path_plot is None:
            self.path_plot = pg.PlotDataItem(
                pen={'color': 'y', 'width': 2}
            )

        if self.predicted_path_plot is None:
            self.predicted_path_plot = pg.PlotDataItem(
                pen={'color': (255, 255, 0, 100), 'width': 2, 'style': Qt.DashLine}
            )

    def _update_visualization(self):
        """Update all visualization elements"""
        if not self.current_state:
            return

        try:
            # Ensure plot items exist
            self._create_plot_items()

            # Update vehicle marker position
            self.vehicle_marker.setPos(
                self.current_state.position_x,
                self.current_state.position_y
            )
            self.vehicle_marker.setRotation(-self.current_state.heading)

            # Update path history
            if self.vehicle_states:
                path_x = np.array([state.position_x for state in self.vehicle_states], dtype=np.float32)
                path_y = np.array([state.position_y for state in self.vehicle_states], dtype=np.float32)
                self.path_plot.setData(x=path_x, y=path_y)

            # Update predicted path
            predicted_path = self._calculate_predicted_path(
                self.current_state,
                prediction_time=2.0,
                steps=20
            )

            if predicted_path:
                points = np.array(predicted_path, dtype=np.float32)
                self.predicted_path_plot.setData(x=points[:, 0], y=points[:, 1])

            # Update track boundaries
            if self.left_boundary:
                left_x = np.array([p.x for p in self.left_boundary], dtype=np.float32)
                left_y = np.array([p.y for p in self.left_boundary], dtype=np.float32)
                self.left_boundary_plot.setData(x=left_x, y=left_y)

            if self.right_boundary:
                right_x = np.array([p.x for p in self.right_boundary], dtype=np.float32)
                right_y = np.array([p.y for p in self.right_boundary], dtype=np.float32)
                self.right_boundary_plot.setData(x=right_x, y=right_y)

            # Emit all plot items for the GUI to update
            self.visualization_updated.emit({
                'vehicle': self.vehicle_marker,
                'path': self.path_plot,
                'predicted_path': self.predicted_path_plot,
                'left_boundary': self.left_boundary_plot,
                'right_boundary': self.right_boundary_plot
            })

        except Exception as e:
            self.error_occurred.emit(f"Error updating visualization: {str(e)}")
        
    def update_vehicle_state(self, state: VehicleState):
        """Update the current vehicle state and trigger visualization update"""
        self.current_state = state
        self.vehicle_states.append(state)
        
        # Limit the number of stored states
        current_time = state.timestamp
        self.vehicle_states = [
            s for s in self.vehicle_states
            if current_time - s.timestamp <= self.track_memory_time
        ]
        
        # Update visualization periodically
        self.update_counter += 1
        if self.update_counter >= self.update_decimation:
            self.update_counter = 0
            self._update_visualization()
            
    def update_track_boundaries(self, camera_data: tuple, 
                              vehicle_state: VehicleState):
        """Update track boundaries based on camera data"""
        left_camera, right_camera = camera_data
        if not left_camera or not right_camera:
            return
            
        try:
            # Process left camera data
            left_edge = self._process_camera_edge(
                left_camera,
                vehicle_state,
                is_left=True
            )
            if left_edge:
                self.left_boundary.append(left_edge)
                
            # Process right camera data
            right_edge = self._process_camera_edge(
                right_camera,
                vehicle_state,
                is_left=False
            )
            if right_edge:
                self.right_boundary.append(right_edge)
                
            # Maintain boundary history
            current_time = vehicle_state.timestamp
            self.left_boundary = [
                point for point in self.left_boundary
                if current_time - point.timestamp <= self.track_memory_time
            ]
            self.right_boundary = [
                point for point in self.right_boundary
                if current_time - point.timestamp <= self.track_memory_time
            ]
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing track boundaries: {str(e)}")
            
    def _process_camera_edge(self, camera_data: List[int], 
                           vehicle_state: VehicleState,
                           is_left: bool) -> Optional[TrackPoint]:
        """Process camera data to extract track edge points"""
        try:
            # Find edge position in camera data
            edge_pixel = self._find_edge(camera_data, is_left)
            if edge_pixel is None:
                return None
                
            # Convert pixel position to physical distance
            # This conversion depends on camera parameters and mounting
            distance = self._pixel_to_distance(edge_pixel)
            angle = self._pixel_to_angle(edge_pixel)
            
            # Calculate global position of track point
            vehicle_heading_rad = math.radians(vehicle_state.heading)
            point_angle = vehicle_heading_rad + angle
            
            # Offset from vehicle position based on camera mounting
            camera_offset = 0.2  # 20cm forward of vehicle center
            base_x = vehicle_state.x + camera_offset * math.cos(vehicle_heading_rad)
            base_y = vehicle_state.y + camera_offset * math.sin(vehicle_heading_rad)
            
            # Calculate final point position
            x = base_x + distance * math.cos(point_angle)
            y = base_y + distance * math.sin(point_angle)
            
            # Calculate confidence based on edge clarity
            confidence = self._calculate_edge_confidence(camera_data, edge_pixel)
            
            return TrackPoint(
                x=x,
                y=y,
                confidence=confidence,
                timestamp=vehicle_state.timestamp
            )
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing camera edge: {str(e)}")
            return None
            
    def _find_edge(self, camera_data: List[int], is_left: bool) -> Optional[int]:
        """Find the track edge position in camera data"""
        # Convert 2-bit pixel values to binary track/non-track
        binary_data = [1 if x > 1 else 0 for x in camera_data]
        
        # Find transitions between track and non-track
        transitions = []
        for i in range(1, len(binary_data)):
            if binary_data[i] != binary_data[i-1]:
                transitions.append(i)
                
        if not transitions:
            return None
            
        # For left edge, find first transition from non-track to track
        # For right edge, find last transition from track to non-track
        if is_left:
            for pos in transitions:
                if binary_data[pos] == 1:  # Found track
                    return pos
        else:
            for pos in reversed(transitions):
                if binary_data[pos] == 0:  # Found non-track
                    return pos
                    
        return None
        
    def _pixel_to_distance(self, pixel: int) -> float:
        """Convert pixel position to physical distance
        
        This is a simplified linear conversion - in reality you would
        need to account for camera parameters, lens distortion, and
        perspective effects.
        """
        pixels_per_meter = 50  # This needs calibration
        return pixel / pixels_per_meter
        
    def _pixel_to_angle(self, pixel: int) -> float:
        """Convert pixel position to viewing angle
        
        This is a simplified linear conversion - in reality you would
        need to account for camera parameters and lens distortion.
        """
        camera_fov = math.radians(60)  # Camera field of view
        pixel_count = 240  # Number of pixels in scan line
        return (pixel - pixel_count/2) * (camera_fov / pixel_count)
        
    def _calculate_edge_confidence(self, camera_data: List[int], 
                                 edge_pixel: int) -> float:
        """Calculate confidence value for detected edge"""
        # Look at contrast around edge position
        window = 3  # Pixels to check on each side
        if edge_pixel < window or edge_pixel >= len(camera_data) - window:
            return 0.5  # Edge near camera limits - lower confidence
            
        # Calculate average values on each side of edge
        track_side = np.mean(camera_data[edge_pixel:edge_pixel+window])
        nontrack_side = np.mean(camera_data[edge_pixel-window:edge_pixel])
        
        # Confidence based on contrast
        contrast = abs(track_side - nontrack_side)
        return min(1.0, contrast / 2.0)  # Normalize to 0-1 range
        
    def _calculate_predicted_path(self, state, prediction_time: float, steps: int):
        """Calculate predicted vehicle path based on current state"""
        try:
            if not hasattr(state, 'speed') or abs(state.speed) < 0.1:
                return []

            dt = prediction_time / steps
            path = [(state.position_x, state.position_y)]

            # Convert steering angle to turning radius
            wheelbase = 0.25  # Vehicle wheelbase in meters
            if abs(state.steering_angle) < 0.1:
                # Essentially straight - use simpler calculation
                heading_rad = math.radians(state.heading)
                for i in range(steps):
                    distance = state.speed * dt
                    x = path[-1][0] + distance * math.cos(heading_rad)
                    y = path[-1][1] + distance * math.sin(heading_rad)
                    path.append((x, y))
            else:
                # Calculate turning radius
                steering_rad = math.radians(state.steering_angle)
                turn_radius = wheelbase / math.tan(abs(steering_rad))
                turn_direction = 1 if state.steering_angle > 0 else -1

                # Start with current heading
                heading_rad = math.radians(state.heading)

                for i in range(steps):
                    # Calculate angular velocity
                    angular_velocity = (state.speed / turn_radius) * turn_direction

                    # Update heading
                    heading_rad += angular_velocity * dt

                    # Calculate new position
                    distance = state.speed * dt
                    x = path[-1][0] + distance * math.cos(heading_rad)
                    y = path[-1][1] + distance * math.sin(heading_rad)
                    path.append((x, y))

            return path

        except Exception as e:
            self.error_occurred.emit(f"Error calculating predicted path: {str(e)}")
            return []

            
    def reset_visualization(self):
        """Reset all visualization elements"""
        self.track_points.clear()
        self.vehicle_states.clear()
        self.left_boundary.clear()
        self.right_boundary.clear()
        self.current_state = None
        self._update_visualization()

    def set_dark_mode(self, enabled: bool):
        """Update visualization colors for dark/light mode"""
        if enabled:
            # Dark mode colors
            self.vehicle_marker.setPen(pg.mkPen('r', width=2))
            self.vehicle_marker.setBrush(pg.mkBrush('r'))
            self.path_plot.setPen(pg.mkPen('y', width=2))
            self.left_boundary_plot.setPen(pg.mkPen('b', width=2))
            self.right_boundary_plot.setPen(pg.mkPen('g', width=2))
            self.predicted_path_plot.setPen(
                pg.mkPen((255, 255, 0, 100), width=2, style=Qt.DashLine)
            )
        else:
            # Light mode colors
            self.vehicle_marker.setPen(pg.mkPen('darkRed', width=2))
            self.vehicle_marker.setBrush(pg.mkBrush('darkRed'))
            self.path_plot.setPen(pg.mkPen('darkYellow', width=2))
            self.left_boundary_plot.setPen(pg.mkPen('darkBlue', width=2))
            self.right_boundary_plot.setPen(pg.mkPen('darkGreen', width=2))
            self.predicted_path_plot.setPen(
                pg.mkPen((128, 128, 0, 100), width=2, style=Qt.DashLine)
            )

    def export_track_data(self, filename: str) -> bool:
        """Export track boundary data to a file for later analysis"""
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("timestamp,type,x,y,confidence\n")
                
                # Write left boundary points
                for point in self.left_boundary:
                    f.write(f"{point.timestamp},left,{point.x:.3f},"
                           f"{point.y:.3f},{point.confidence:.3f}\n")
                
                # Write right boundary points
                for point in self.right_boundary:
                    f.write(f"{point.timestamp},right,{point.x:.3f},"
                           f"{point.y:.3f},{point.confidence:.3f}\n")
                           
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error exporting track data: {str(e)}")
            return False

    def import_track_data(self, filename: str) -> bool:
        """Import previously exported track boundary data"""
        try:
            self.left_boundary.clear()
            self.right_boundary.clear()
            
            with open(filename, 'r') as f:
                # Skip header
                next(f)
                
                # Read points
                for line in f:
                    timestamp, type_, x, y, confidence = line.strip().split(',')
                    point = TrackPoint(
                        x=float(x),
                        y=float(y),
                        confidence=float(confidence),
                        timestamp=float(timestamp)
                    )
                    
                    if type_ == 'left':
                        self.left_boundary.append(point)
                    else:
                        self.right_boundary.append(point)
                        
            # Trigger visualization update
            self._update_visualization()
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error importing track data: {str(e)}")
            return False

    def get_track_statistics(self) -> Dict[str, float]:
        """Calculate and return statistics about the tracked course"""
        stats = {
            'track_length': 0.0,
            'average_width': 0.0,
            'min_width': float('inf'),
            'max_width': 0.0,
            'average_confidence': 0.0
        }
        
        try:
            # Calculate track length using left boundary
            # (could average with right boundary for better accuracy)
            if len(self.left_boundary) > 1:
                length = 0.0
                for i in range(1, len(self.left_boundary)):
                    prev = self.left_boundary[i-1]
                    curr = self.left_boundary[i]
                    length += math.sqrt(
                        (curr.x - prev.x)**2 + (curr.y - prev.y)**2
                    )
                stats['track_length'] = length
                
            # Calculate track width statistics
            widths = []
            confidences = []
            
            # Match up left and right boundary points by timestamp
            left_dict = {p.timestamp: p for p in self.left_boundary}
            right_dict = {p.timestamp: p for p in self.right_boundary}
            
            # Find timestamps that exist in both boundaries
            common_times = set(left_dict.keys()) & set(right_dict.keys())
            
            for timestamp in common_times:
                left = left_dict[timestamp]
                right = right_dict[timestamp]
                
                # Calculate width at this point
                width = math.sqrt(
                    (right.x - left.x)**2 + (right.y - left.y)**2
                )
                widths.append(width)
                
                # Store confidence values
                confidences.append(left.confidence)
                confidences.append(right.confidence)
                
            if widths:
                stats['average_width'] = sum(widths) / len(widths)
                stats['min_width'] = min(widths)
                stats['max_width'] = max(widths)
                
            if confidences:
                stats['average_confidence'] = sum(confidences) / len(confidences)
                
            return stats
            
        except Exception as e:
            self.error_occurred.emit(f"Error calculating track statistics: {str(e)}")
            return stats

    def get_closest_boundary_points(self) -> Tuple[Optional[TrackPoint], 
                                                 Optional[TrackPoint]]:
        """Get the closest track boundary points to current vehicle position"""
        if not self.current_state or not self.left_boundary or not self.right_boundary:
            return None, None
            
        try:
            # Find closest left boundary point
            closest_left = min(
                self.left_boundary,
                key=lambda p: (p.x - self.current_state.x)**2 + 
                            (p.y - self.current_state.y)**2
            )
            
            # Find closest right boundary point
            closest_right = min(
                self.right_boundary,
                key=lambda p: (p.x - self.current_state.x)**2 + 
                            (p.y - self.current_state.y)**2
            )
            
            return closest_left, closest_right
            
        except Exception as e:
            self.error_occurred.emit(
                f"Error finding closest boundary points: {str(e)}"
            )
            return None, None

    def calculate_centerline_error(self) -> Optional[float]:
        """Calculate the current deviation from track centerline"""
        closest_left, closest_right = self.get_closest_boundary_points()
        if not closest_left or not closest_right:
            return None
            
        try:
            # Calculate centerline point
            center_x = (closest_left.x + closest_right.x) / 2
            center_y = (closest_left.y + closest_right.y) / 2
            
            if not self.current_state:
                return None
                
            # Calculate distance from vehicle to centerline
            error = math.sqrt(
                (self.current_state.x - center_x)**2 + 
                (self.current_state.y - center_y)**2
            )
            
            # Determine sign based on which side of centerline we're on
            # This requires knowing the track direction - this is a simplified version
            # A more accurate version would consider the track's local direction
            if self.current_state.y > center_y:
                error = -error
                
            return error
            
        except Exception as e:
            self.error_occurred.emit(
                f"Error calculating centerline error: {str(e)}"
            )
            return None

    def estimate_track_curvature(self) -> Optional[float]:
        """Estimate the track curvature at current vehicle position"""
        try:
            # Get a window of recent track points
            window_size = 5
            recent_points = []
            
            # Collect recent left and right boundary points
            timestamps = sorted(set(
                [p.timestamp for p in self.left_boundary] +
                [p.timestamp for p in self.right_boundary]
            ))[-window_size:]
            
            for t in timestamps:
                left = next((p for p in self.left_boundary 
                           if abs(p.timestamp - t) < 0.01), None)
                right = next((p for p in self.right_boundary 
                            if abs(p.timestamp - t) < 0.01), None)
                
                if left and right:
                    # Use centerline point
                    recent_points.append((
                        (left.x + right.x) / 2,
                        (left.y + right.y) / 2
                    ))
                    
            if len(recent_points) < 3:
                return None
                
            # Fit a circle to these points to estimate curvature
            # This uses the modified least squares circle fit algorithm
            x = np.array([p[0] for p in recent_points])
            y = np.array([p[1] for p in recent_points])
            
            # Construct the data matrix
            A = np.column_stack([x, y, np.ones(len(x))])
            b = -(x**2 + y**2)
            
            # Solve the system of equations
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Extract circle parameters
            a = float(-solution[0] / 2)
            b = float(-solution[1] / 2)
            r = float(np.sqrt(a**2 + b**2 - solution[2]))
            
            # Return curvature (1/radius)
            return 1.0 / r if r > 0.1 else 0.0
            
        except Exception as e:
            self.error_occurred.emit(
                f"Error estimating track curvature: {str(e)}"
            )
            return None