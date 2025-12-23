import tensorflow as tf
import sys
import os

print("=" * 60)
print("TensorFlow 安装验证")
print("=" * 60)

# 设置 CUDA 环境
cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
os.environ['CUDA_PATH'] = cuda_path
os.environ['PATH'] = f"{cuda_path}\\bin;{cuda_path}\\libnvvp;{os.environ['PATH']}"

print(f"Python 版本: {sys.version}")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"TensorFlow 路径: {tf.__file__}")

print(f"\n构建信息:")
print(f"  使用 CUDA 构建: {tf.test.is_built_with_cuda()}")
print(f"  使用 GPU 构建: {tf.test.is_built_with_gpu_support()}")

print(f"\n设备列表:")
gpu_devices = tf.config.list_physical_devices('GPU')
cpu_devices = tf.config.list_physical_devices('CPU')
print(f"  GPU 设备: {gpu_devices}")
print(f"  CPU 设备: {cpu_devices}")

if gpu_devices:
    print("\n🎉 GPU 支持已启用！正在测试性能...")
    # 测试 GPU 性能
    with tf.device('/GPU:0'):
        import time

        start_time = time.time()

        # 创建较大的计算任务
        a = tf.random.normal([2000, 2000])
        b = tf.random.normal([2000, 2000])
        c = tf.matmul(a, b)

        end_time = time.time()
        print(f"✅ GPU 矩阵乘法完成!")
        print(f"   计算时间: {end_time - start_time:.2f} 秒")
        print(f"   结果形状: {c.shape}")

        # 显示 GPU 信息
        for i, gpu in enumerate(gpu_devices):
            print(f"   GPU {i}: {gpu}")
else:
    print("\n❌ 未检测到 GPU 设备")

# 显示所有可用设备
from tensorflow.python.client import device_lib

print(f"\n所有计算设备:")
devices = device_lib.list_local_devices()
for device in devices:
    print(f"  - {device.name} | {device.device_type} | 内存: {device.memory_limit / 1024 ** 3:.1f} GB")