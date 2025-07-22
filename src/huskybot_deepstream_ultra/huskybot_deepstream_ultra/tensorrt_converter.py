#!/usr/bin/env python3
"""
TensorRT Model Converter for Ultra-High Performance
Converts YOLO11 models to optimized TensorRT engines for 100+ FPS
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def download_model(model_name):
    """Download YOLO model if not exists"""
    try:
        if not os.path.exists(model_name):
            print(f"📥 Downloading {model_name}...")
            if ULTRALYTICS_AVAILABLE:
                model = YOLO(model_name)
                print(f"✅ Downloaded: {model_name}")
                return True
            else:
                print(f"❌ Ultralytics not available")
                return False
        else:
            print(f"✅ Model exists: {model_name}")
            return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def convert_to_tensorrt(model_path, output_dir, precision='fp16', workspace=4096):
    """Convert YOLO model to TensorRT engine"""
    try:
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available")
        
        print(f"🔥 Converting {model_path} to TensorRT {precision.upper()}...")
        
        # Load model
        model = YOLO(model_path)
        
        # Export to TensorRT
        engine_path = model.export(
            format='engine',
            device=0,  # GPU 0
            half=(precision == 'fp16'),
            int8=(precision == 'int8'),
            workspace=workspace,
            imgsz=640,
            batch=1,
            verbose=True
        )
        
        if engine_path and os.path.exists(engine_path):
            # Move to output directory
            output_path = os.path.join(output_dir, os.path.basename(engine_path))
            if engine_path != output_path:
                os.rename(engine_path, output_path)
            
            print(f"✅ TensorRT engine created: {output_path}")
            return output_path
        else:
            raise RuntimeError("Engine creation failed")
            
    except Exception as e:
        print(f"❌ TensorRT conversion failed: {e}")
        return None

def benchmark_model(engine_path, num_runs=100):
    """Benchmark TensorRT engine performance"""
    try:
        if not ULTRALYTICS_AVAILABLE:
            return None
        
        print(f"🏁 Benchmarking {engine_path}...")
        
        model = YOLO(engine_path)
        
        # Warmup
        print("🔥 Warming up...")
        for i in range(10):
            import numpy as np
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy, verbose=False)
        
        # Benchmark
        print(f"⏱️ Running {num_runs} inferences...")
        start_time = time.time()
        
        for i in range(num_runs):
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy, verbose=False)
            
            if (i + 1) % 25 == 0:
                print(f"Progress: {i+1}/{num_runs}")
        
        total_time = time.time() - start_time
        avg_inference = total_time / num_runs
        fps = 1.0 / avg_inference
        
        print(f"📊 BENCHMARK RESULTS:")
        print(f"   ⚡ Average Inference: {avg_inference*1000:.2f}ms")
        print(f"   🚀 Theoretical FPS: {fps:.1f}")
        print(f"   ⏰ Total Time: {total_time:.2f}s")
        
        return {
            'avg_inference_ms': avg_inference * 1000,
            'fps': fps,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None

def optimize_jetson():
    """Optimize Jetson for maximum performance"""
    try:
        print("🔧 Optimizing Jetson AGX Orin for MAXIMUM PERFORMANCE...")
        
        # Set maximum performance mode
        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
        print("✅ Set to maximum performance mode")
        
        # Enable maximum clocks
        subprocess.run(['sudo', 'jetson_clocks', '--fan'], check=True)
        print("✅ Enabled maximum clocks and fan")
        
        # Set CPU governor to performance
        for i in range(12):  # AGX Orin has 12 cores
            try:
                subprocess.run([
                    'sudo', 'sh', '-c', 
                    f'echo performance > /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor'
                ], check=True)
            except:
                pass
        print("✅ Set CPU governor to performance")
        
        # Optimize memory
        subprocess.run(['sudo', 'sysctl', '-w', 'vm.swappiness=1'], check=True)
        print("✅ Optimized memory settings")
        
        print("🎯 Jetson optimization complete!")
        
    except Exception as e:
        print(f"❌ Jetson optimization failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='TensorRT Model Converter for Ultra Performance')
    parser.add_argument('--model', default='yolo11x-seg.pt', help='YOLO model to convert')
    parser.add_argument('--precision', choices=['fp16', 'int8'], default='fp16', help='TensorRT precision')
    parser.add_argument('--output', default='/home/kmp-orin/jezzy/huskybot', help='Output directory')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after conversion')
    parser.add_argument('--optimize-jetson', action='store_true', help='Optimize Jetson for maximum performance')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    if not ULTRALYTICS_AVAILABLE:
        print("❌ Error: Ultralytics not available. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🚀 ULTRA-HIGH PERFORMANCE TENSORRT CONVERTER")
    print("=" * 60)
    
    # Optimize Jetson if requested
    if args.optimize_jetson:
        optimize_jetson()
        print()
    
    # Download model if needed
    if not download_model(args.model):
        sys.exit(1)
    
    print()
    
    # Convert to TensorRT
    engine_path = convert_to_tensorrt(args.model, args.output, args.precision)
    if not engine_path:
        sys.exit(1)
    
    print()
    
    # Benchmark if requested
    if args.benchmark:
        results = benchmark_model(engine_path, args.runs)
        if results:
            if results['fps'] >= 100:
                print("🎯 TARGET ACHIEVED: 100+ FPS!")
            else:
                print(f"⚡ Performance: {results['fps']:.1f} FPS (Target: 100+ FPS)")
    
    print("\n✅ Conversion complete!")
    print(f"📁 TensorRT Engine: {engine_path}")
    print("\n🚀 Ready for ULTRA-HIGH PERFORMANCE inference!")

if __name__ == '__main__':
    main()
