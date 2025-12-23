import os
import sys
import subprocess
import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU 深度诊断")
print("=" * 60)

# 设置 CUDA 11.8 环境
cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
os.environ['CUDA_PATH'] = cuda_path
os.environ['PATH'] = f"{cuda_path}\\bin;{cuda_path}\\libnvvp;{os.environ['PATH']}"

print("1. 🛠️ 环境变量设置:")
print(f"   CUDA_PATH: {os.environ.get('CUDA_PATH')}")
print(f"   PATH 开头部分: {os.environ['PATH'][:200]}...")

print("\n2. 🔍 系统级检查:")
try:
    # 检查 nvidia-smi
    nvidia_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if nvidia_result.returncode == 0:
        print("   ✅ nvidia-smi 正常工作")
        # 提取驱动版本
        for line in nvidia_result.stdout.split('\n'):
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
    else:
        print("   ❌ nvidia-smi 失败")
except Exception as e:
    print(f"   ❌ nvidia-smi 错误: {e}")

print("\n3. 📚 库文件检查:")
cuda_libs = [
    "cudart64_110.dll",
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cudnn64_8.dll",
    "cufft64_10.dll",
    "curand64_10.dll"
]

for lib in cuda_libs:
    lib_path = os.path.join(cuda_path, "bin", lib)
    exists = os.path.exists(lib_path)
    status = "✅" if exists else "❌"
    print(f"   {status} {lib}: {exists}")

print("\n4. 🐍 Python 和 TensorFlow 检查:")
print(f"   Python 版本: {sys.version}")
print(f"   TensorFlow 版本: {tf.__version__}")
print(f"   TensorFlow 路径: {tf.__file__}")

print("\n5. 🔧 TensorFlow 构建信息:")
print(f"   使用 CUDA 构建: {tf.test.is_built_with_cuda()}")
print(f"   使用 GPU 构建: {tf.test.is_built_with_gpu_support()}")

print("\n6. 💻 设备列表:")
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"   GPU 设备: {gpu_devices}")

    if not gpu_devices:
        print("   ⚠️  未找到 GPU 设备，检查详细日志...")

except Exception as e:
    print(f"   ❌ 获取设备列表错误: {e}")

print("\n7. 📋 所有计算设备:")
try:
    from tensorflow.python.client import device_lib

    devices = device_lib.list_local_devices()
    for device in devices:
        print(f"   - {device.name} | {device.device_type}")
except Exception as e:
    print(f"   ❌ 获取详细设备错误: {e}")

print("\n8. 🚀 尝试强制 GPU 操作:")
try:
    # 尝试创建一个简单的 GPU 操作
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
        print(f"   ✅ GPU 操作成功: {c.numpy()}")
except Exception as e:
    print(f"   ❌ GPU 操作失败: {e}")

print("\n" + "=" * 60)