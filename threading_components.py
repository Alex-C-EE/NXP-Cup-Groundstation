# threading_components.py

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
import logging
from typing import Optional
from queue import Queue

class DataProcessingWorker(QObject):
    """Worker object that processes data in a separate thread"""
    data_processed = pyqtSignal(object)  # Emits processed sensor data
    error_occurred = pyqtSignal(str)     # Emits error messages
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active = True
        self.logger = logging.getLogger('DataProcessingWorker')
        self._setup_processing_queue()

    def _setup_processing_queue(self):
        """Initialize the processing queue"""
        self.processing_queue = Queue()

    @pyqtSlot(object)
    def process_data(self, sensor_data):
        """Process incoming sensor data"""
        try:
            if not self.active:
                return
                
            # Add data to processing queue
            self.processing_queue.put(sensor_data)
            
            # Process all available data
            self._process_queue()
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing data: {str(e)}")

    def _process_queue(self):
        """Process all data in the queue"""
        try:
            while not self.processing_queue.empty() and self.active:
                # Get next data item
                sensor_data = self.processing_queue.get_nowait()
                
                # Process the data
                # We're keeping the original data structure intact
                self.data_processed.emit(sensor_data)
                
        except Exception as e:
            self.error_occurred.emit(f"Error in processing queue: {str(e)}")

    def cleanup(self):
        """Clean up worker resources"""
        self.active = False
        # Clear the processing queue
        while not self.processing_queue.empty():
            self.processing_queue.get_nowait()

class DataThread(QThread):
    """Dedicated thread for data processing"""
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[DataProcessingWorker] = None
        self.logger = logging.getLogger('DataThread')

    def setup_worker(self, worker: DataProcessingWorker):
        """Set up the worker object for this thread"""
        self.worker = worker
        # This is crucial - move the worker to this thread
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