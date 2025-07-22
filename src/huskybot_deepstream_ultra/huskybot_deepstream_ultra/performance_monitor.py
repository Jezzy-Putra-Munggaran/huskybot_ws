#!/usr/bin/env python3
"""
Performance Monitor for Ultra-High Performance System
Real-time monitoring of FPS, GPU usage, memory, and system performance
"""

import rclpy
from rclpy.node import Node
import time
import threading
import subprocess
import re
import psutil
import os

class PerformanceMonitor(Node):
    """Real-time performance monitoring"""
    
    def __init__(self):
        super().__init__('performance_monitor')
        
        self.monitoring_active = True
        
        # Performance metrics
        self.fps_targets = {
            'camera_front': 0,
            'camera_front_left': 0,
            'camera_left': 0,
            'camera_rear': 0,
            'camera_rear_right': 0,
            'camera_right': 0
        }
        
        self.setup_monitoring()
        self.start_monitoring_threads()
        
        self.get_logger().info("🚀 Performance Monitor Started!")
    
    def setup_monitoring(self):
        """Setup monitoring components"""
        # Timer for periodic reports
        self.report_timer = self.create_timer(5.0, self.generate_report)
        
        # GPU monitoring timer
        self.gpu_timer = self.create_timer(2.0, self.monitor_gpu)
    
    def start_monitoring_threads(self):
        """Start monitoring threads"""
        # System monitoring thread
        self.system_thread = threading.Thread(
            target=self.monitor_system,
            daemon=True,
            name="SystemMonitor"
        )
        self.system_thread.start()
        
        # ROS topic monitoring thread
        self.topic_thread = threading.Thread(
            target=self.monitor_topics,
            daemon=True,
            name="TopicMonitor"
        )
        self.topic_thread.start()
    
    def monitor_gpu(self):
        """Monitor GPU performance"""
        try:
            # Run nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            if result.stdout:
                values = result.stdout.strip().split(', ')
                if len(values) >= 4:
                    gpu_util = float(values[0])
                    mem_used = int(values[1])
                    mem_total = int(values[2])
                    temp = float(values[3])
                    
                    mem_percent = (mem_used / mem_total) * 100
                    
                    self.get_logger().info(
                        f"🎮 GPU: {gpu_util}% | Memory: {mem_used}/{mem_total}MB ({mem_percent:.1f}%) | Temp: {temp}°C"
                    )
                    
                    # Check if GPU is being fully utilized
                    if gpu_util < 80:
                        self.get_logger().warn(f"⚠️ GPU underutilized: {gpu_util}% (Target: >80%)")
                    
                    if mem_percent < 50:
                        self.get_logger().warn(f"⚠️ GPU memory underutilized: {mem_percent:.1f}% (Target: >50%)")
        
        except Exception as e:
            self.get_logger().error(f"❌ GPU monitoring error: {e}")
    
    def monitor_system(self):
        """Monitor system performance"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Temperature (if available)
                temp_info = "N/A"
                try:
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps:
                        temp_info = f"{temps['coretemp'][0].current}°C"
                except:
                    pass
                
                self.get_logger().info(
                    f"💻 CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% ({memory.used//1024//1024}MB/{memory.total//1024//1024}MB) | Temp: {temp_info}"
                )
                
                # Check if system resources are being utilized
                if cpu_percent < 50:
                    self.get_logger().warn(f"⚠️ CPU underutilized: {cpu_percent:.1f}% (Consider higher parallelism)")
                
                if memory.percent < 40:
                    self.get_logger().warn(f"⚠️ RAM underutilized: {memory.percent:.1f}% (Can increase batch sizes)")
                
                time.sleep(3.0)
                
            except Exception as e:
                self.get_logger().error(f"❌ System monitoring error: {e}")
                time.sleep(5.0)
    
    def monitor_topics(self):
        """Monitor ROS topic rates"""
        while self.monitoring_active:
            try:
                # Monitor camera topics
                camera_topics = [
                    '/camera_front/image_raw',
                    '/camera_front_left/image_raw',
                    '/camera_left/image_raw',
                    '/camera_rear/image_raw',
                    '/camera_rear_right/image_raw',
                    '/camera_right/image_raw'
                ]
                
                for topic in camera_topics:
                    try:
                        # Use ros2 topic hz to get rate
                        result = subprocess.run([
                            'timeout', '3', 'ros2', 'topic', 'hz', topic
                        ], capture_output=True, text=True)
                        
                        if result.stdout:
                            # Parse hz output
                            match = re.search(r'average rate: ([\d.]+)', result.stdout)
                            if match:
                                rate = float(match.group(1))
                                camera_name = topic.split('/')[1]
                                self.fps_targets[camera_name] = rate
                                
                                if rate < 20:
                                    self.get_logger().warn(f"⚠️ {topic}: {rate:.1f} Hz (Low rate)")
                                elif rate >= 100:
                                    self.get_logger().info(f"🎯 {topic}: {rate:.1f} Hz (TARGET ACHIEVED!)")
                                else:
                                    self.get_logger().info(f"📊 {topic}: {rate:.1f} Hz")
                        
                    except Exception as e:
                        self.get_logger().debug(f"Topic {topic} monitoring error: {e}")
                
                time.sleep(10.0)  # Check topics every 10 seconds
                
            except Exception as e:
                self.get_logger().error(f"❌ Topic monitoring error: {e}")
                time.sleep(15.0)
    
    def generate_report(self):
        """Generate performance report"""
        try:
            current_time = time.time()
            
            # Calculate total FPS
            total_fps = sum(self.fps_targets.values())
            avg_fps = total_fps / len(self.fps_targets) if self.fps_targets else 0
            
            # Performance status
            status = "🎯 TARGET ACHIEVED!" if avg_fps >= 100 else "⚡ OPTIMIZING..."
            
            # Generate report
            report = [
                "=" * 80,
                "🚀 HUSKYBOT ULTRA-HIGH PERFORMANCE REPORT",
                "=" * 80,
                f"📊 Average FPS: {avg_fps:.1f} | Target: 100+ FPS",
                f"🎯 Status: {status}",
                f"📡 Individual Camera Rates:",
            ]
            
            for camera, fps in self.fps_targets.items():
                status_icon = "🎯" if fps >= 100 else "⚡" if fps >= 50 else "❌"
                report.append(f"   {status_icon} {camera}: {fps:.1f} FPS")
            
            report.extend([
                "=" * 80,
                f"🕒 Report Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 80
            ])
            
            # Log report
            for line in report:
                self.get_logger().info(line)
                
        except Exception as e:
            self.get_logger().error(f"❌ Report generation error: {e}")
    
    def check_jetson_optimization(self):
        """Check if Jetson is optimized"""
        try:
            # Check nvpmodel
            result = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True)
            if 'MAXN' not in result.stdout:
                self.get_logger().warn("⚠️ Jetson not in MAXN mode. Run: sudo nvpmodel -m 0")
            
            # Check jetson_clocks
            result = subprocess.run(['jetson_clocks', '--show'], capture_output=True, text=True)
            if 'jetson_clocks is not running' in result.stdout:
                self.get_logger().warn("⚠️ jetson_clocks not active. Run: sudo jetson_clocks --fan")
            
        except Exception as e:
            self.get_logger().debug(f"Jetson optimization check error: {e}")
    
    def destroy_node(self):
        """Clean shutdown"""
        self.monitoring_active = False
        super().destroy_node()

def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = PerformanceMonitor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Performance Monitor...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
