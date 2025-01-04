# NXP-Cup-Groundstation
 Groundstation Software for Monitoring Race Car Performance

## Overall Architecture:

This is a race car ground station application using PyQt5
Uses a multi-threaded architecture for handling real-time data
Follows the Model-View-Controller pattern


## Key Components:

DataHandler: Manages data collection and processing
TrackVisualizer: Handles visualization of track and vehicle position
MainWindow: Main GUI with multiple specialized tabs
ProcessingThread/Worker: Handles data processing in separate threads


## Threading Model:

Uses Qt's event-driven threading model
Main thread handles GUI
Separate threads for data processing and collection
Uses QSocketNotifier for async serial port reading
Thread safety handled through Qt's signal/slot mechanism


## Data Flow:

Serial Port -> DataHandler -> ProcessingWorker -> GUI Components
                          -> Track Visualizer

## Key Design Patterns:

Observer Pattern (via Qt signals/slots)
Factory Pattern (for creating UI components)
Strategy Pattern (for different visualization modes)

## Error Handling & Logging:

Comprehensive error propagation through signal/slot system
Centralized error handling in MainWindow
Hierarchical logging with both file and console outputs
Good separation between UI errors and system errors

## Configuration Management:

Flexible config system via ConfigManager
Settings persisted between sessions
Runtime configuration changes supported
Handles both UI and system configurations

## Data Structures:

Well-defined dataclasses for different types of sensor data:

CameraData for line scan readings
MotorData for drive system
EDFData for Electric Ducted Fan
SystemHealth for diagnostics
SensorData as main composite structure

## Data Processing Pipeline:

Raw Serial Data -> Parse & Validate -> Queue -> ProcessingWorker -> 
Plot Manager -> Visualization Components

## Visualization Architecture:

Layered approach to plotting:

Base PlotWidget provides common functionality
Specialized widgets (LivePlot, TrackVisualizer) extend base
Plot Manager handles thread-safe updates
Efficient update batching and decimation

## Memory Management:

Circular buffers for time-series data
Explicit cleanup in component destructors
Smart resource management through Qt parent-child relationships
Careful handling of thread lifetimes

## Inter-thread Communication:

Strictly follows Qt's threading rules
Main thread owns GUI elements
Worker thread ownership properly transferred
Careful use of QMetaObject for thread-safe operations

## Real-time Considerations:

Non-blocking I/O with QSocketNotifier
Update rate management
Performance optimization via update decimation
Efficient data structures for real-time processing

## State Management:

Clear separation of system states
Thread states properly tracked
Playback vs live modes handled cleanly
Graceful state transitions

## Developer Experience:

Clean separation of concerns
Well-documented classes and methods
Consistent error handling patterns
Clear initialization sequences


