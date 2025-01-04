# config_manager.py
import json
import os
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

class ConfigManager:
    """
    Enhanced configuration manager for race car ground station.
    Provides secure storage, logging, and robust configuration management.
    """
    def __init__(self, config_dir: str = "~/.2space"):
        # Initialize logging
        self.logger = logging.getLogger('race_ground_station.config')
        
        # Set up paths using pathlib for better cross-platform support
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.config_file = self.config_dir / "config.json"
        self.secrets_file = self.config_dir / ".secrets"
        
        # Default configuration - preserving your existing structure
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
        
        self.logger.info("Configuration manager initialized successfully")

    def _ensure_config_directory(self) -> None:
        """Create configuration directory if it doesn't exist."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured config directory exists: {self.config_dir}")
        except Exception as e:
            self.logger.error(f"Error creating config directory: {e}")
            raise

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create with defaults if not exists."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults, keeping any additional user settings
                    merged_config = self.default_config.copy()
                    self._deep_update(merged_config, loaded_config)
                    self.logger.debug("Loaded and merged existing configuration")
                    return merged_config
                    
            self.logger.info("No existing config found, using defaults")
            return self.default_config.copy()
            
        except Exception as e:
            self.logger.error(f"Error loading config, using defaults: {e}")
            return self.default_config.copy()

    def _deep_update(self, base: dict, update: dict) -> None:
        """Recursively update nested dictionary while preserving existing keys."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _load_secrets(self) -> Dict[str, str]:
        """Load encrypted secrets with enhanced security and logging."""
        if not self.secrets_file.exists():
            self.logger.debug("No secrets file found")
            return {}
            
        try:
            # Ensure secure file permissions
            current_mode = os.stat(self.secrets_file).st_mode
            if (current_mode & 0o777) != 0o600:
                os.chmod(self.secrets_file, 0o600)
                self.logger.warning("Fixed secrets file permissions")
                
            with open(self.secrets_file, 'r') as f:
                encoded_secrets = f.read().strip()
                if encoded_secrets:
                    decoded = base64.b64decode(encoded_secrets)
                    self.logger.debug("Successfully loaded secrets")
                    return json.loads(decoded.decode())
                    
            return {}
            
        except Exception as e:
            self.logger.error(f"Error loading secrets: {e}")
            return {}

    def save_config(self) -> None:
        """Save configuration with backup and error handling."""
        try:
            # Create backup directory if needed
            backup_dir = self.config_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Create timestamped backup if file exists
            if self.config_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = backup_dir / f"config_{timestamp}.json"
                self.config_file.rename(backup_file)
                self.logger.debug(f"Created backup: {backup_file}")
                
            # Write new config file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            # Set secure permissions
            os.chmod(self.config_file, 0o600)
            
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            # Attempt to restore from backup
            latest_backup = sorted(backup_dir.glob("config_*.json"))[-1]
            if latest_backup.exists():
                latest_backup.rename(self.config_file)
                self.logger.warning("Restored from latest backup")

    def save_secret(self, key: str, value: str) -> None:
        """Save secret value with enhanced security and atomic operations."""
        self.secrets[key] = value
        try:
            encoded = base64.b64encode(json.dumps(self.secrets).encode())
            
            # Use temporary file for atomic write
            temp_file = self.secrets_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                f.write(encoded.decode())
            os.chmod(temp_file, 0o600)
            
            # Atomic replace
            temp_file.replace(self.secrets_file)
            self.logger.debug(f"Successfully saved secret: {key}")
            
        except Exception as e:
            self.logger.error(f"Error saving secret: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret value."""
        return self.secrets.get(key)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value with logging."""
        try:
            return self.config[section][key]
        except KeyError:
            self.logger.debug(f"Config value not found, using default: {section}.{key}")
            return default

    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value with logging."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.logger.debug(f"Set config value: {section}.{key}")
        
    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """Reset configuration to defaults with logging."""
        if section:
            if section in self.default_config:
                self.config[section] = self.default_config[section].copy()
                self.logger.info(f"Reset section to defaults: {section}")
        else:
            self.config = self.default_config.copy()
            self.logger.info("Reset entire configuration to defaults")
        self.save_config()