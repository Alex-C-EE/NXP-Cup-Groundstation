#!/usr/bin/env python3

import math
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import serial
import csv
import os
from datetime import datetime

class TestDataGenerator:
    """Generates realistic test data for race car telemetry testing"""
    
    def __init__(self):
        # Vehicle state
        self.speed = 0.0  # m/s
        self.steering_angle = 0.0  # degrees
        self.position_x = 0.0  # meters
        self.position_y = 0.0  # meters
        self.heading = 0.0  # degrees
        
        # Vehicle dynamics constraints
        self.MAX_SPEED = 10.0  # m/s
        self.MAX_ACCELERATION = 5.0  # m/s²
        self.MAX_STEERING_ANGLE = 30.0  # degrees
        self.MAX_STEERING_RATE = 90.0  # degrees/s
        self.WHEELBASE = 0.25  # meters
        
        # Track detection parameters
        self.TRACK_WIDTH = 1.0  # meters
        self.CAMERA_FOV = 60  # degrees
        self.CAMERA_PIXELS = 240  # pixels per camera
        
        # System state
        self.battery_voltage = 16.8  # Start fully charged
        self.motor_temp_left = 25.0  # Start at ambient temperature
        self.motor_temp_right = 25.0
        self.packet_counter = 0
        self.last_update = time.time()
        
        # Initialize noise generators
        self.speed_noise = self._create_noise_generator(0.1)  # 0.1 m/s noise
        self.imu_noise = self._create_noise_generator(0.2)  # 0.2 m/s² noise
        
        # CSV file handle
        self.csv_file = None
        self.csv_writer = None

    def _create_noise_generator(self, magnitude):
        """Creates a generator function that produces Gaussian noise with given magnitude"""
        def noise_gen():
            return random.gauss(0, magnitude)
        return noise_gen

    def _update_vehicle_dynamics(self, dt):
        """Update vehicle state based on simple dynamics model"""
        # Add random variations to speed and steering
        self.speed += random.uniform(-0.5, 0.5) * dt
        self.speed = max(0, min(self.speed, self.MAX_SPEED))
        
        self.steering_angle += random.uniform(-10, 10) * dt
        self.steering_angle = max(-self.MAX_STEERING_ANGLE, 
                                min(self.steering_angle, self.MAX_STEERING_ANGLE))
        
        # Update position based on bicycle model
        v = self.speed
        beta = math.radians(self.steering_angle)
        self.heading += (v * math.tan(beta) / self.WHEELBASE) * dt
        
        self.position_x += v * math.cos(self.heading) * dt
        self.position_y += v * math.sin(self.heading) * dt

    def _update_system_state(self, dt):
        """Update system state variables like temperatures and battery"""
        # Battery discharge based on speed
        power_draw = 0.5 + abs(self.speed) * 0.3  # Basic power model
        self.battery_voltage = max(12.0, self.battery_voltage - power_draw * dt * 0.01)
        
        # Motor temperatures rise with speed and cool with ambient
        ambient_temp = 25.0
        heat_rate = self.speed * self.speed * 0.1
        cooling_rate = 0.1
        
        self.motor_temp_left += (heat_rate - cooling_rate * (self.motor_temp_left - ambient_temp)) * dt
        self.motor_temp_right += (heat_rate - cooling_rate * (self.motor_temp_right - ambient_temp)) * dt

    def _generate_camera_data(self) -> Tuple[List[int], List[int]]:
        """Generate simulated camera data showing track edges"""
        left_camera = [0] * self.CAMERA_PIXELS
        right_camera = [0] * self.CAMERA_PIXELS
        
        # Simulate track edges as bright lines
        left_edge = int(self.CAMERA_PIXELS * 0.3 + random.uniform(-10, 10))
        right_edge = int(self.CAMERA_PIXELS * 0.7 + random.uniform(-10, 10))
        
        # Add bright pixels at edges with some noise
        for i in range(self.CAMERA_PIXELS):
            if abs(i - left_edge) < 3:
                left_camera[i] = 3
                right_camera[i] = 3
            elif abs(i - right_edge) < 3:
                left_camera[i] = 3
                right_camera[i] = 3
            else:
                left_camera[i] = random.randint(0, 1)
                right_camera[i] = random.randint(0, 1)
        
        return left_camera, right_camera
        
    def start_csv_logging(self, filename=None):
        """Start logging data to CSV file"""
        if filename is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"race_telemetry_{timestamp}.csv"
            
        try:
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            header = [
                "timestamp", "packet_counter", "system_state", "error_flags",
                "speed", "steering_angle"
            ]
            # Add camera data headers
            for i in range(60):  # 60 bytes for each camera
                header.extend([f"left_camera_{i}", f"right_camera_{i}"])
            
            header.extend([
                "left_edge_pos", "right_edge_pos", "track_width", "edge_confidence",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "left_motor_duty", "left_motor_current", "left_motor_temp",
                "right_motor_duty", "right_motor_current", "right_motor_temp",
                "front_edf_power", "front_edf_force", "rear_edf_power", "rear_edf_force",
                "battery_voltage", "battery_current",
                "cpu_load_1", "cpu_load_2", "cpu_load_3",
                "board_temp_1", "board_temp_2", "board_temp_3",
                "can_error_1", "can_error_2",
                "loop_time", "pid_state", "control_mode",
                "ir_value_1", "ir_value_2", "ir_value_3", "ir_value_4",
                "tof_distance",
                "centerline_error", "track_confidence"
            ])
            
            self.csv_writer.writerow(header)
            return True
            
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            return False
            
    def stop_csv_logging(self):
        """Stop logging and close CSV file"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def generate_packet(self) -> str:
        """Generate a complete telemetry packet and optionally save to CSV"""
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Update vehicle and system state
        self._update_vehicle_dynamics(dt)
        self._update_system_state(dt)
        
        # Generate camera data
        left_camera, right_camera = self._generate_camera_data()
        
        # Create packet fields
        timestamp_ms = int(current_time * 1000)
        self.packet_counter += 1
        system_state = 1  # Normal operation
        error_flags = 0  # No errors
        
        # Generate IMU data with noise
        accel_x = self.imu_noise()
        accel_y = self.speed_noise()  # Longitudinal acceleration
        accel_z = -9.81 + self.imu_noise()  # Gravity + noise
        
        gyro_x = self.imu_noise()
        gyro_y = self.imu_noise()
        gyro_z = math.degrees(self.speed * math.tan(math.radians(self.steering_angle)) / self.WHEELBASE)
        
        # Format camera data
        camera_bytes = []
        for i in range(0, self.CAMERA_PIXELS, 4):
            # Pack 4 pixels into one byte for each camera
            left_byte = sum(left_camera[i + j] << (6 - j*2) for j in range(4))
            right_byte = sum(right_camera[i + j] << (6 - j*2) for j in range(4))
            camera_bytes.extend([left_byte, right_byte])
        
        # Create packet string
        packet = f"{timestamp_ms},{self.packet_counter},{system_state},{error_flags},"
        packet += f"{self.speed*100:.0f},{self.steering_angle*100:.0f},"  # Speed and steering scaled
        packet += ','.join(str(x) for x in camera_bytes) + ','
        packet += f"0,240,{self.TRACK_WIDTH*1000:.0f},95,"  # Left camera edges
        packet += f"0,240,{self.TRACK_WIDTH*1000:.0f},95,"  # Right camera edges
        packet += f"{accel_x*100:.0f},{accel_y*100:.0f},{accel_z*100:.0f},"  # IMU data scaled
        packet += f"{gyro_x*100:.0f},{gyro_y*100:.0f},{gyro_z*100:.0f},"
        packet += f"{abs(self.speed*10):.0f},0,{self.motor_temp_left:.0f},"  # Left motor
        packet += f"{abs(self.speed*10):.0f},0,{self.motor_temp_right:.0f},"  # Right motor
        packet += "50,100,50,100,"  # EDF data
        packet += f"{self.battery_voltage*100:.0f},{abs(self.speed*15):.0f},"  # Power system
        packet += "20,15,25,"  # CPU loads
        packet += "45,42,40,"  # Board temperatures
        packet += "0,0,"  # CAN errors
        packet += "500,1,1,"  # Loop time, PID state, control mode
        packet += "100,200,300,400,"  # IR values
        packet += "150,"  # ToF distance
        packet += "0,100"  # Centerline error and track confidence
        
        # If CSV logging is active, write the packet
        if self.csv_writer:
            try:
                # Split the packet string and convert to appropriate types
                fields = packet.strip().split(',')
                
                # Convert string values to appropriate numeric types
                converted_fields = []
                for i, field in enumerate(fields):
                    try:
                        if '.' in field:
                            converted_fields.append(float(field))
                        else:
                            converted_fields.append(int(field))
                    except ValueError:
                        converted_fields.append(field)
                
                self.csv_writer.writerow(converted_fields)
                
            except Exception as e:
                print(f"Error writing to CSV: {e}")
        
        return packet + '\n'

def main():
    """Run test data generator with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test telemetry data')
    parser.add_argument('--port', help='Serial port to use (optional)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baudrate')
    parser.add_argument('--rate', type=float, default=30.0, help='Update rate in Hz')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration in seconds (0 for infinite)')
    parser.add_argument('--csv', help='CSV output file (defaults to timestamped file)', default=None)
    parser.add_argument('--no-csv', action='store_true', help='Disable CSV logging')
    args = parser.parse_args()
    
    # Initialize generator variable
    generator = None
    
    # Set up default CSV filename if logging isn't disabled
    if not args.no_csv:
        if args.csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.csv = f"race_telemetry_{timestamp}.csv"
    
    try:
        # Open serial port if specified
        ser = None
        if args.port:
            ser = serial.Serial(args.port, args.baudrate)
            print(f"Opened {args.port} at {args.baudrate} baud")
        
        # Create generator
        generator = TestDataGenerator()
        
        # Start CSV logging if specified
        if args.csv:
            if generator.start_csv_logging(args.csv):
                print(f"Logging to {args.csv}")
            else:
                return
        
        # Main loop
        start_time = time.time()
        packet_count = 0
        
        while True:
            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break
                
            # Generate packet
            packet = generator.generate_packet()
            packet_count += 1
            
            # Send to serial port if configured
            if ser:
                ser.write(packet.encode('ascii'))
            
            # Print status every second
            if packet_count % int(args.rate) == 0:
                elapsed = time.time() - start_time
                print(f"\rGenerated {packet_count} packets ({elapsed:.1f} seconds)", end='')
            
            # Wait for next update
            time.sleep(1.0 / args.rate)
            
    except KeyboardInterrupt:
        print("\nStopping data generation")
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    finally:
        if ser:
            ser.close()
        if generator:
            generator.stop_csv_logging()
        print(f"\nGenerated {packet_count} packets")

if __name__ == '__main__':
    main()