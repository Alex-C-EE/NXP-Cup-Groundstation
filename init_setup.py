#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
from pathlib import Path
import pkg_resources
import json

def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print("Python version OK")

def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    
    required = {
        "PyQt5": "5.15.0",
        "pyqtgraph": "0.12.0",
        "numpy": "1.19.0",
        "pyserial": "3.5"
    }
    
    missing = []
    outdated = []
    
    for package, min_version in required.items():
        try:
            version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                outdated.append((package, version, min_version))
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing or outdated:
        print("\nDependency issues found:")
        if missing:
            print("\nMissing packages:")
            for package in missing:
                print(f"  - {package}")
        if outdated:
            print("\nOutdated packages:")
            for package, current, required in outdated:
                print(f"  - {package}: {current} (requires >= {required})")
        
        if input("\nWould you like to install/upgrade these packages? (y/n): ").lower() == 'y':
            install_dependencies()
        else:
            print("Please install required packages manually")
            sys.exit(1)
    else:
        print("All dependencies OK")

def install_dependencies():
    """Install or upgrade required packages."""
    print("\nInstalling/upgrading dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def setup_directories():
    """Create necessary directories for the application."""
    print("\nSetting up application directories...")
    
    home = Path.home()
    app_dir = home / ".2space"
    directories = {
        "config": app_dir,
        "logs": app_dir / "logs",
        "data": app_dir / "data",
        "cache": app_dir / "cache"
    }
    
    for name, path in directories.items():
        print(f"Creating {name} directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        
        # Set appropriate permissions
        if name in ["config", "logs"]:
            path.chmod(0o700)  # Restricted access for sensitive directories
        else:
            path.chmod(0o755)  # More permissive for other directories

def check_system_requirements():
    """Check system requirements and optimize settings."""
    print("\nChecking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 2 * 1024 * 1024 * 1024:  # 2GB
            print("Warning: Less than 2GB of RAM available")
            print("Consider reducing plot buffer sizes and update intervals")
    except ImportError:
        print("Warning: Could not check system memory (psutil not installed)")
    
    # Check if running on Raspberry Pi
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "Raspberry Pi" in cpuinfo:
            print("Detected Raspberry Pi - optimizing settings...")
            optimize_for_raspberry_pi()
    except Exception:
        print("Not running on Raspberry Pi")

def optimize_for_raspberry_pi():
    """Apply optimized settings for Raspberry Pi."""
    config_file = Path.home() / ".2space" / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        
        # Apply optimized settings
        config["display"]["update_interval_ms"] = 50  # Reduce update frequency
        config["display"]["plot_buffer_size"] = 500  # Reduce buffer size
        config["visualization"]["prediction_time"] = 1.0  # Shorter prediction
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print("Applied optimized settings for Raspberry Pi")


def setup_udev_rules():
    """Set up udev rules for serial device access."""
    print("\nSetting up udev rules for serial devices...")
    
    rule_content = '''# Race Car Ground Station serial device rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0666", GROUP="dialout", SYMLINK+="racecar"
'''
    
    rule_file = "/etc/udev/rules.d/99-racecar.rules"
    
    try:
        if os.geteuid() == 0:  # Running as root
            with open(rule_file, 'w') as f:
                f.write(rule_content)
            subprocess.run(['udevadm', 'control', '--reload-rules'])
            subprocess.run(['udevadm', 'trigger'])
            print("udev rules installed successfully")
        else:
            print("Warning: Need root privileges to install udev rules")
            print(f"To install manually, create {rule_file} with the following content:")
            print(rule_content)
            print("Then run:")
            print("  sudo udevadm control --reload-rules")
            print("  sudo udevadm trigger")
    except Exception as e:
        print(f"Error setting up udev rules: {e}")
        print("Please set up serial device permissions manually")

def configure_user_groups():
    """Add user to necessary groups for hardware access."""
    print("\nConfiguring user groups...")
    
    required_groups = ['dialout', 'gpio', 'video']
    username = os.getenv('SUDO_USER', os.getenv('USER'))
    
    if not username:
        print("Could not determine username")
        return
        
    for group in required_groups:
        try:
            subprocess.run(['groups', username], capture_output=True, text=True)
            result = subprocess.run(['groups', username], capture_output=True, text=True)
            
            if group not in result.stdout:
                if os.geteuid() == 0:  # Running as root
                    subprocess.run(['usermod', '-a', '-G', group, username])
                    print(f"Added user to {group} group")
                else:
                    print(f"Please run: sudo usermod -a -G {group} {username}")
        except Exception as e:
            print(f"Error configuring {group} group: {e}")

def create_desktop_entry():
    """Create desktop entry for easy launching."""
    print("\nCreating desktop entry...")
    
    entry_content = f'''[Desktop Entry]
Name=Race Car Ground Station
Comment=Telemetry and control interface for autonomous race car
Exec={sys.executable} {os.path.abspath('main.py')}
Terminal=false
Type=Application
Categories=Development;Engineering;
Icon={os.path.abspath('resources/icon.png')}
'''
    
    desktop_file = os.path.expanduser('~/.local/share/applications/racecar-gs.desktop')
    
    try:
        os.makedirs(os.path.dirname(desktop_file), exist_ok=True)
        with open(desktop_file, 'w') as f:
            f.write(entry_content)
        os.chmod(desktop_file, 0o755)
        print("Desktop entry created successfully")
    except Exception as e:
        print(f"Error creating desktop entry: {e}")
        print("You can still run the application from the command line")

def setup_autostart():
    """Configure automatic start on boot if requested."""
    print("\nWould you like to start the ground station automatically on boot? (y/n)")
    if input().lower() != 'y':
        return
        
    autostart_dir = os.path.expanduser('~/.config/autostart')
    autostart_file = os.path.join(autostart_dir, 'racecar-gs.desktop')
    
    try:
        os.makedirs(autostart_dir, exist_ok=True)
        shutil.copy2(
            os.path.expanduser('~/.local/share/applications/racecar-gs.desktop'),
            autostart_file
        )
        print("Autostart configured successfully")
    except Exception as e:
        print(f"Error configuring autostart: {e}")

def create_virtual_environment():
    """Create a Python virtual environment for the application."""
    print("\nCreating virtual environment...")
    
    venv_dir = 'venv'
    if os.path.exists(venv_dir):
        print("Virtual environment already exists")
        return
        
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
        
        # Determine the pip path in the virtual environment
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_dir, 'Scripts', 'pip')
        else:  # Unix-like
            pip_path = os.path.join(venv_dir, 'bin', 'pip')
        
        # Install requirements in virtual environment
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("Virtual environment created and dependencies installed")
        
        # Create activation script reminder
        print("\nTo activate the virtual environment:")
        if os.name == 'nt':
            print(f"  {venv_dir}\\Scripts\\activate")
        else:
            print(f"  source {venv_dir}/bin/activate")
            
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def verify_installation():
    """Verify the installation by running basic checks."""
    print("\nVerifying installation...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", lambda: os.path.isdir(os.path.expanduser("~/.2space"))),
        ("Configuration", lambda: os.path.isfile(os.path.expanduser("~/.2space/config.json"))),
        ("Virtual Environment", lambda: os.path.isdir("venv"))
    ]
    
    all_passed = True
    for name, check in checks:
        try:
            result = check()
            if result is not False:  # None or True indicates success
                print(f"✓ {name}: OK")
            else:
                print(f"✗ {name}: Failed")
                all_passed = False
        except Exception as e:
            print(f"✗ {name}: Error - {str(e)}")
            all_passed = False
    
    return all_passed

def main():
    """Main setup procedure."""
    print("Race Car Ground Station Setup")
    print("============================")
    
    try:
        check_python_version()
        check_dependencies()
        setup_directories()
        check_system_requirements()
        
        if os.name != 'nt':  # Skip on Windows
            setup_udev_rules()
            configure_user_groups()
        
        create_virtual_environment()
        create_desktop_entry()
        setup_autostart()
        
        if verify_installation():
            print("\nSetup completed successfully!")
            print("\nYou can now:")
            print("1. Activate the virtual environment")
            print("2. Run 'python main.py' to start the ground station")
            print("\nFor more information, see the documentation.")
        else:
            print("\nSetup completed with some issues.")
            print("Please review the messages above and address any problems.")
            print("Consult the troubleshooting guide in the documentation.")
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()