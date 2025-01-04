# main.py

import sys
import signal
import logging
import traceback
from queue import Queue
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QThread

# Import our application components
from config_manager import ConfigManager
from data_handler import DataHandler
from track_visualizer import TrackVisualizer as TrackVisualizerCore
from gui import MainWindow

class ApplicationManager:
    """
    Manages the lifecycle of the race car ground station application.
    Handles initialization, cleanup, and coordination between components.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = None
        self.data_handler = None
        self.track_visualizer = None
        self.main_window = None
        self.app = None
        
        # Thread management
        self.threads = []
        
        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """
        Configure application logging with both file and console handlers.
        Creates a new log file for each session with timestamp.
        """
        # Create logs directory if it doesn't exist
        log_dir = Path.home() / ".2space" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ground_station_{timestamp}.log"
        
        # Configure logging format
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.DEBUG)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(logging.INFO)
        
        # Create and configure logger
        logger = logging.getLogger('race_ground_station')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _signal_handler(self, signum, frame):
        """
        Handle system signals for graceful shutdown.
        This is particularly important for cleaning up serial connections
        and ensuring all data is saved.
        """
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.cleanup()
        sys.exit(0)

    def initialize(self) -> bool:
        """
        Initialize all application components in the correct order.
        Returns True if initialization is successful, False otherwise.
        """
        try:
            self.logger.info("Initializing Race Car Ground Station...")

            # Initialize configuration
            self.logger.debug("Loading configuration...")
            self.config = ConfigManager()

            # Create Qt application
            self.logger.debug("Creating Qt application...")
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
            self.app = QApplication(sys.argv)
            self.app.setStyle('Fusion')  

            # Set up data communication queue
            self.logger.debug("Setting up data queue...")
            data_queue = Queue()

            # Initialize track visualizer
            self.logger.debug("Initializing track visualizer...")
            self.track_visualizer = TrackVisualizerCore(self.config)

            # Initialize data handler
            self.logger.debug("Initializing data handler...")
            self.data_handler = DataHandler(self.config, data_queue)

            # Create main window
            self.logger.debug("Creating main window...")
            self.main_window = MainWindow(
                self.config,
                self.data_handler,
                self.track_visualizer
            )
            
            # Connect signals between components
            self._connect_signals()
            
            # Show main window
            self.main_window.show()
            
            self.logger.info("Initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _connect_signals(self):
        """
        Connect signals between various components for communication.
        This helps maintain loose coupling between components.
        """
        try:
            # Connect data handler signals
            self.data_handler.data_ready.connect(self.track_visualizer.update_vehicle_state)
            self.data_handler.error_occurred.connect(self._handle_error)
            
            # Connect track visualizer signals
            self.track_visualizer.error_occurred.connect(self._handle_error)
            
            self.logger.debug("Signals connected successfully")
            
        except Exception as e:
            self.logger.error(f"Error connecting signals: {str(e)}")
            raise

    def _handle_error(self, error_message: str):
        """
        Central error handling method that logs errors and updates the UI.
        """
        self.logger.error(error_message)
        if self.main_window:
            self.main_window.handle_error(error_message)

    def run(self) -> int:
        """
        Start the application and enter the main event loop.
        Returns the application exit code.
        """
        if not self.app:
            self.logger.error("Cannot run: Application not initialized")
            return 1
            
        try:
            # Start data collection
            self.logger.info("Starting data collection...")
            self.data_handler.start()
            
            # Enter the Qt event loop
            self.logger.info("Entering main event loop")
            return self.app.exec_()
            
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 1

    def cleanup(self):
        """Clean up resources and perform orderly shutdown."""
        self.logger.info("Starting application cleanup...")
        
        try:
            # Stop data collection first
            if self.data_handler:
                self.logger.debug("Stopping data handler...")
                self.data_handler.stop()
                # Give threads time to stop
                QThread.msleep(100)
            
            # Save configuration
            if self.config:
                self.logger.debug("Saving configuration...")
                self.config.save_config()
            
            # Clean up remaining threads
            for thread in self.threads:
                if thread.isRunning():
                    self.logger.debug(f"Stopping thread: {thread.objectName()}")
                    thread.quit()
                    # Wait with timeout
                    if not thread.wait(1000):  # 1 second timeout
                        self.logger.warning(f"Thread {thread.objectName()} did not stop gracefully")
                        thread.terminate()
            
            # Clean up main window explicitly
            if self.main_window:
                self.logger.debug("Cleaning up main window...")
                # This will trigger the closeEvent handler
                self.main_window.close()
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """
    Main entry point for the race car ground station application.
    Handles high-level application lifecycle and error cases.
    """
    # Create application manager
    app_manager = ApplicationManager()
    
    try:
        # Initialize the application
        if not app_manager.initialize():
            app_manager.logger.error("Failed to initialize application")
            return 1
            
        # Run the application
        exit_code = app_manager.run()
        
        # Perform cleanup
        app_manager.cleanup()
        
        return exit_code
        
    except Exception as e:
        app_manager.logger.error(f"Unhandled exception in main: {str(e)}")
        app_manager.logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())