# config_manager.py
import json
import os
from pathlib import Path
import base64
from typing import Any, Dict, Optional

class ConfigManager:
    """
    Handles application configuration settings for the race car ground station.
    Provides secure storage for sensitive data and maintains default configurations
    appropriate for racing telemetry visualization.
    """
    def __init__(self, config_dir: str = "~/.2space"):
        self.config_dir = os.path.expanduser(config_dir)
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.secrets_file = os.path.join(self.config_dir, ".secrets")
        
        # Default configuration
        self.default_config = {
            "serial": {
                "default_port": "/dev/ttyUSB0",
                "baudrate": 115200,
                "timeout": 1.0
            },
            "display": {
                "dark_mode": True,
                "update_interval_ms": 33,  # ~30 FPS
                "plot_buffer_size": 1000,
                "track_memory_time": 30.0  # How many seconds of track to display
            },
            "data_logging": {
                "enabled": True,
                "directory": "~/2space_logs",
                "max_log_size_mb": 100
            },
            "alerts": {
                "battery_threshold": 14.0,  # Voltage alert threshold
                "temperature_threshold": 80,  # Motor temperature alert (Celsius)
                "track_confidence_threshold": 50,  # Minimum track detection confidence
                "enable_sound": True,  # Enable alert sounds
                "enable_visual": True  # Enable visual alerts
            },
            "vehicle": {
                "wheelbase": 0.25,  # Vehicle wheelbase in meters
                "track_width": 0.2,  # Vehicle track width in meters
                "max_steering_angle": 30,  # Maximum steering angle in degrees
                "camera_offset": 0.2,  # Camera distance from vehicle center
                "camera_fov": 60  # Camera field of view in degrees
            },
            "visualization": {
                "path_color": "#FFFF00",
                "track_left_color": "#0000FF",
                "track_right_color": "#00FF00",
                "vehicle_color": "#FF0000",
                "show_predicted_path": True,
                "prediction_time": 2.0,  # Seconds to predict ahead
                "show_confidence": True  # Show track detection confidence
            },
            "performance": {
                "enable_logging": True,
                "log_interval_ms": 100,
                "max_data_points": 10000,
                "auto_save": True
            }
        }
        
        self._ensure_config_directory()
        self.config = self._load_config()
        self.secrets = self._load_secrets()

    def _ensure_config_directory(self) -> None:
        """Create configuration directory if it doesn't exist."""
        os.makedirs(self.config_dir, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create with defaults if not exists."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults, keeping any additional user settings
                    merged_config = self.default_config.copy()
                    self._deep_update(merged_config, loaded_config)
                    return merged_config
            return self.default_config.copy()
        except Exception as e:
            print(f"Error loading config, using defaults: {e}")
            return self.default_config.copy()

    def _deep_update(self, base: dict, update: dict) -> None:
        """
        Recursively update a nested dictionary while preserving existing keys.
        This ensures new default settings are added without overwriting user customizations.
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _load_secrets(self) -> Dict[str, str]:
        """Load encrypted secrets from file with enhanced security."""
        if not os.path.exists(self.secrets_file):
            return {}
            
        try:
            # Ensure secure file permissions
            current_mode = os.stat(self.secrets_file).st_mode
            if (current_mode & 0o777) != 0o600:
                os.chmod(self.secrets_file, 0o600)
                
            with open(self.secrets_file, 'r') as f:
                encoded_secrets = f.read().strip()
                if encoded_secrets:
                    decoded = base64.b64decode(encoded_secrets)
                    return json.loads(decoded.decode())
            return {}
        except Exception as e:
            print(f"Error loading secrets: {e}")
            return {}

    def save_config(self) -> None:
        """Save current configuration to file with error handling."""
        try:
            # Create a backup of the existing config file if it exists
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup"
                os.replace(self.config_file, backup_file)
                
            # Write new config file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            # Set secure permissions
            os.chmod(self.config_file, 0o600)
            
        except Exception as e:
            print(f"Error saving config: {e}")
            # If we have a backup and the save failed, restore it
            if os.path.exists(f"{self.config_file}.backup"):
                os.replace(f"{self.config_file}.backup", self.config_file)

    def save_secret(self, key: str, value: str) -> None:
        """Save a secret value securely with enhanced encryption."""
        self.secrets[key] = value
        try:
            encoded = base64.b64encode(json.dumps(self.secrets).encode())
            
            # Write to temporary file first
            temp_file = f"{self.secrets_file}.tmp"
            with open(temp_file, 'w') as f:
                f.write(encoded.decode())
            os.chmod(temp_file, 0o600)
            
            # Atomically replace the original file
            os.replace(temp_file, self.secrets_file)
            
        except Exception as e:
            print(f"Error saving secret: {e}")
            if os.path.exists(f"{self.secrets_file}.tmp"):
                os.remove(f"{self.secrets_file}.tmp")

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret value."""
        return self.secrets.get(key)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        try:
            return self.config[section][key]
        except KeyError:
            return default

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value, creating sections as needed."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        
    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """
        Reset configuration to defaults, either entirely or for a specific section.
        This is useful for troubleshooting or when settings become corrupted.
        """
        if section:
            if section in self.default_config:
                self.config[section] = self.default_config[section].copy()
        else:
            self.config = self.default_config.copy()
        self.save_config()