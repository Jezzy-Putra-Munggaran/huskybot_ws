from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'huskybot_deepstream_ultra'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'std_msgs',
        'cv_bridge',
        'ultralytics>=8.3.0',
        'numpy',
        'opencv-python',
        'pyyaml',
        'torch',
        'torchvision',
        'tensorrt',
        'pycuda',
    ],
    zip_safe=True,
    maintainer='Jezzy Putra Munggaran',
    maintainer_email='mungguran.jezzy.putra@gmail.com',
    description='Ultra-High Performance DeepStream YOLO11 TensorRT for 100+ FPS 360° segmentation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ultra_deepstream_node = huskybot_deepstream_ultra.ultra_deepstream_node:main',
            'tensorrt_converter = huskybot_deepstream_ultra.tensorrt_converter:main',
            'performance_monitor = huskybot_deepstream_ultra.performance_monitor:main',
        ],
    },
)
