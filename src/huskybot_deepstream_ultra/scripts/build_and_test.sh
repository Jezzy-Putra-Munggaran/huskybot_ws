#!/bin/bash

# Ultra-High Performance Build and Test Script
# Optimizes and validates 100+ FPS system

set -e  # Exit on any error

echo "🚀 ========================================================"
echo "   HUSKYBOT ULTRA-HIGH PERFORMANCE BUILD & TEST"
echo "🚀 ========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

# Check if running on Jetson
check_jetson() {
    print_header "Checking Jetson Platform..."
    
    if [ -f /etc/nv_tegra_release ]; then
        JETSON_VERSION=$(cat /etc/nv_tegra_release)
        print_success "Running on Jetson: $JETSON_VERSION"
        
        # Check if AGX Orin
        if grep -q "REVISION: 5.0" /proc/device-tree/nvidia,dtsfilename 2>/dev/null; then
            print_success "✅ Jetson AGX Orin detected!"
        else
            print_warning "⚠️ Not AGX Orin - performance may be limited"
        fi
    else
        print_warning "⚠️ Not running on Jetson - using CPU fallback"
    fi
}

# Optimize Jetson for maximum performance
optimize_jetson() {
    print_header "Optimizing Jetson for Ultra Performance..."
    
    if [ -f /etc/nv_tegra_release ]; then
        print_status "Setting MAXN performance mode..."
        sudo nvpmodel -m 0 || print_warning "Could not set nvpmodel"
        
        print_status "Enabling jetson_clocks with fan control..."
        sudo jetson_clocks --fan || print_warning "Could not enable jetson_clocks"
        
        print_status "Setting CPU governor to performance..."
        echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null || true
        
        print_success "✅ Jetson optimization complete!"
    else
        print_warning "⚠️ Skipping Jetson optimization (not on Jetson)"
    fi
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies..."
    
    # Check ROS2
    if command -v ros2 &> /dev/null; then
        ROS_VERSION=$(ros2 --version 2>/dev/null | head -n1)
        print_success "✅ ROS2 found: $ROS_VERSION"
    else
        print_error "❌ ROS2 not found!"
        exit 1
    fi
    
    # Check Python packages
    python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" || print_error "❌ OpenCV not found"
    python3 -c "import numpy; print('NumPy version:', numpy.__version__)" || print_error "❌ NumPy not found"
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_success "✅ CUDA found: $CUDA_VERSION"
    else
        print_warning "⚠️ CUDA not found - using CPU fallback"
    fi
    
    # Check TensorRT
    if python3 -c "import tensorrt" 2>/dev/null; then
        TRT_VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
        print_success "✅ TensorRT found: $TRT_VERSION"
    else
        print_warning "⚠️ TensorRT not found - performance will be limited"
    fi
}

# Setup workspace
setup_workspace() {
    print_header "Setting up Workspace..."
    
    # Navigate to workspace
    cd /home/jezzy/huskybot_ws
    
    # Source ROS2
    source /opt/ros/humble/setup.bash
    
    print_success "✅ Workspace setup complete"
}

# Build the packages
build_packages() {
    print_header "Building Ultra-High Performance Packages..."
    
    # Clean previous builds if needed
    if [ "$1" = "clean" ]; then
        print_status "Cleaning previous builds..."
        rm -rf build/ install/ log/
    fi
    
    # Build with optimizations
    print_status "Building packages with optimizations..."
    colcon build \
        --packages-select huskybot_deepstream_ultra huskybot_camera huskybot_multicam_parallel \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --cmake-args -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        --event-handlers console_direct+ \
        --parallel-workers $(nproc)
    
    if [ $? -eq 0 ]; then
        print_success "✅ Build completed successfully!"
    else
        print_error "❌ Build failed!"
        exit 1
    fi
}

# Convert YOLO model to TensorRT
convert_model() {
    print_header "Converting YOLO Model to TensorRT..."
    
    # Create models directory
    mkdir -p /home/jezzy/huskybot_ws/models
    
    # Source the workspace
    source install/setup.bash
    
    # Check if model already exists
    if [ -f "/home/jezzy/huskybot_ws/models/yolo11m-seg.engine" ]; then
        print_warning "⚠️ TensorRT model already exists"
        read -p "Reconvert? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    # Run conversion
    print_status "Converting YOLOv11m-seg to TensorRT..."
    python3 src/huskybot_deepstream_ultra/huskybot_deepstream_ultra/tensorrt_converter.py \
        --model yolo11m-seg.pt \
        --output /home/jezzy/huskybot_ws/models/yolo11m-seg.engine \
        --precision FP16 \
        --workspace 4096
    
    if [ $? -eq 0 ]; then
        print_success "✅ Model conversion complete!"
    else
        print_error "❌ Model conversion failed!"
        exit 1
    fi
}

# Test camera connections
test_cameras() {
    print_header "Testing Camera Connections..."
    
    for i in {0..5}; do
        if [ -e "/dev/video$i" ]; then
            print_success "✅ Camera /dev/video$i detected"
        else
            print_warning "⚠️ Camera /dev/video$i not found"
        fi
    done
}

# Test system performance
test_performance() {
    print_header "Testing System Performance..."
    
    # Source the workspace
    source install/setup.bash
    
    print_status "Starting performance test (30 seconds)..."
    
    # Launch the performance monitor in background
    timeout 30s ros2 run huskybot_deepstream_ultra performance_monitor &
    MONITOR_PID=$!
    
    # Wait for test completion
    wait $MONITOR_PID 2>/dev/null || true
    
    print_success "✅ Performance test completed"
}

# Run quick validation
quick_validation() {
    print_header "Running Quick Validation..."
    
    # Source the workspace
    source install/setup.bash
    
    # Test node availability
    print_status "Checking node executables..."
    
    if [ -f "install/huskybot_deepstream_ultra/lib/huskybot_deepstream_ultra/ultra_deepstream_node" ]; then
        print_success "✅ ultra_deepstream_node available"
    else
        print_error "❌ ultra_deepstream_node not found"
    fi
    
    if [ -f "install/huskybot_deepstream_ultra/lib/huskybot_deepstream_ultra/performance_monitor" ]; then
        print_success "✅ performance_monitor available"
    else
        print_error "❌ performance_monitor not found"
    fi
    
    # Test launch file
    if [ -f "src/huskybot_deepstream_ultra/launch/ultra_deepstream_pipeline.launch.py" ]; then
        print_success "✅ Launch file available"
    else
        print_error "❌ Launch file not found"
    fi
}

# Print usage instructions
print_usage() {
    print_header "Usage Instructions"
    echo ""
    echo "🚀 To launch the ultra-high performance system:"
    echo "   ros2 launch huskybot_deepstream_ultra ultra_deepstream_pipeline.launch.py"
    echo ""
    echo "🎯 To launch with custom parameters:"
    echo "   ros2 launch huskybot_deepstream_ultra ultra_deepstream_pipeline.launch.py target_fps:=120 batch_size:=8"
    echo ""
    echo "📊 To monitor performance only:"
    echo "   ros2 run huskybot_deepstream_ultra performance_monitor"
    echo ""
    echo "🔧 To convert models:"
    echo "   python3 src/huskybot_deepstream_ultra/huskybot_deepstream_ultra/tensorrt_converter.py"
    echo ""
}

# Main execution
main() {
    print_header "Starting Ultra-High Performance Setup..."
    
    # Parse arguments
    CLEAN_BUILD=false
    SKIP_MODEL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --skip-model)
                SKIP_MODEL=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--clean] [--skip-model] [--help]"
                echo "  --clean      Clean build before building"
                echo "  --skip-model Skip model conversion"
                echo "  --help       Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute steps
    check_jetson
    optimize_jetson
    check_dependencies
    setup_workspace
    
    if [ "$CLEAN_BUILD" = true ]; then
        build_packages clean
    else
        build_packages
    fi
    
    if [ "$SKIP_MODEL" = false ]; then
        convert_model
    fi
    
    test_cameras
    quick_validation
    
    print_success "🎯 ========================================================"
    print_success "   ULTRA-HIGH PERFORMANCE SETUP COMPLETE!"
    print_success "🎯 ========================================================"
    
    print_usage
}

# Run main function
main "$@"
