import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
import joblib

warnings.filterwarnings('ignore')


class QuantileNet(nn.Module):
    """分位数回归神经网络"""

    def __init__(self, input_size, output_size, hidden_sizes=[128, 64, 32]):
        super(QuantileNet, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DistributionalRandomForestGPU(BaseEstimator, RegressorMixin):
    """
    Distributional Random Forests - PyTorch GPU版本
    使用神经网络模拟分位数回归森林
    """

    def __init__(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                 hidden_sizes=[128, 64, 32], batch_size=256,
                 n_epochs=100, learning_rate=0.001, random_state=42):
        self.quantiles = quantiles
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def quantile_loss(self, y_pred, y_true, q):
        """分位数损失函数"""
        errors = y_true - y_pred
        return torch.max((q - 1) * errors, q * errors).mean()

    def fit_single_quantile(self, X, y, quantile):
        """训练单个分位数模型"""
        torch.manual_seed(self.random_state)

        # 数据准备
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 模型初始化
        input_size = X.shape[1]
        model = QuantileNet(input_size, 1, self.hidden_sizes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # 训练循环
        model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X).squeeze()
                loss = self.quantile_loss(predictions, batch_y, quantile)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(
                    f"    分位数 {quantile} - 轮次 {epoch + 1}/{self.n_epochs}, 损失: {total_loss / len(dataloader):.4f}")

        return model

    def fit(self, X, y):
        """训练所有分位数模型"""
        print(f"训练PyTorch GPU分位数模型 ({len(self.quantiles)}个分位数)...")

        # 确保数据是numpy数组
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # 为每个分位数训练模型
        for i, q in enumerate(self.quantiles):
            print(f"  训练分位数 {q} ({i + 1}/{len(self.quantiles)})...")
            model = self.fit_single_quantile(X, y, q)
            self.models[q] = model

        return self

    def predict(self, X, return_distribution=False):
        """预测"""
        if hasattr(X, 'values'):
            X = X.values
        X = X.astype(np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)

        if return_distribution:
            # 返回完整分布
            distribution = {}
            for q, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    pred = model(X_tensor).cpu().numpy().flatten()
                distribution[q] = pred
            return distribution
        else:
            # 返回中位数
            model = self.models[0.5]
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().flatten()
            return pred

    def predict_interval(self, X, confidence=0.9):
        """预测区间"""
        alpha = (1 - confidence) / 2
        lower_q = alpha
        upper_q = 1 - alpha

        # 找到最接近的分位数
        lower_quantile = min(self.quantiles, key=lambda x: abs(x - lower_q))
        upper_quantile = min(self.quantiles, key=lambda x: abs(x - upper_q))

        distribution = self.predict(X, return_distribution=True)
        lower_bound = distribution[lower_quantile]
        upper_bound = distribution[upper_quantile]

        return lower_bound, upper_bound

    def get_uncertainty(self, X):
        """获取预测不确定性"""
        distribution = self.predict(X, return_distribution=True)
        predictions = np.array(list(distribution.values()))
        uncertainty = np.percentile(predictions, 75, axis=0) - np.percentile(predictions, 25, axis=0)
        return uncertainty


# 主程序开始
start_time = time.time()

# 检查GPU
print("检查GPU可用性...")
if torch.cuda.is_available():
    print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print("❌ GPU不可用，使用CPU")

# 加载数据
print("加载数据...")
train_dataSet = pd.read_csv('modified_数据集Time_Series662_detail.dat')
test_dataSet = pd.read_csv('modified_数据集Time_Series661_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']


def fast_feature_engineering(data):
    """极速特征工程"""
    features = data[noise_columns].copy()
    features['noise_mean'] = features[noise_columns].mean(axis=1)
    features = features.fillna(method='bfill')
    return features


print("开始特征工程...")
X_train_enhanced = fast_feature_engineering(train_dataSet)
X_test_enhanced = fast_feature_engineering(test_dataSet)

y_train = train_dataSet[columns].values
y_test = test_dataSet[columns].values

print(f"数据形状 - 训练集: {X_train_enhanced.shape}, 测试集: {X_test_enhanced.shape}")


def sample_training_data(X, y, sample_ratio=0.6):
    """采样训练数据"""
    n_samples = int(len(X) * sample_ratio)
    indices = np.random.choice(len(X), n_samples, replace=False)
    return X.iloc[indices] if hasattr(X, 'iloc') else X[indices], y[indices]


print("训练数据采样中...")
X_train_sampled, y_train_sampled = sample_training_data(X_train_enhanced, y_train, 0.6)
print(f"采样后形状 - 训练集: {X_train_sampled.shape}")

print("目标变量标准化...")
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_sampled)

print("开始PyTorch GPU分位数模型训练...")
models = []
cv_scores = []

# GPU优化参数
gpu_params = {
    'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
    'hidden_sizes': [256, 128, 64],  # 更大的网络
    'batch_size': 512,  # 更大的批次
    'n_epochs': 100,
    'learning_rate': 0.001,
    'random_state': 42
}

for i, col in enumerate(columns):
    print(f"\n训练 {col} ({i + 1}/{len(columns)})...")

    # 使用PyTorch GPU模型
    model = DistributionalRandomForestGPU(**gpu_params)

    # 训练模型
    model.fit(X_train_sampled.values, y_train_scaled[:, i])
    models.append(model)

    # 快速验证
    train_pred = model.predict(X_train_sampled.values)
    train_mae = np.mean(np.abs(y_train_scaled[:, i] - train_pred))
    cv_scores.append(train_mae)

    print(f"  {col} 训练MAE: {train_mae:.4f}")

print("\n进行完整测试集预测...")
test_predictions_scaled = []
test_uncertainties = []

for i, col in enumerate(columns):
    print(f"预测 {col}...")

    # 预测
    pred_scaled = models[i].predict(X_test_enhanced.values)
    test_predictions_scaled.append(pred_scaled)

    # 获取不确定性
    uncertainty = models[i].get_uncertainty(X_test_enhanced.values)
    test_uncertainties.append(uncertainty)

# 反标准化
y_pred_scaled = np.column_stack(test_predictions_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 计算不确定性（原始尺度）
uncertainties_original = []
for i in range(len(columns)):
    col_scale = scaler_y.scale_[i]
    uncertainty_original = test_uncertainties[i] * col_scale
    uncertainties_original.append(uncertainty_original)

uncertainties_original = np.column_stack(uncertainties_original)

# 计算误差
test_mae = np.mean(np.abs(y_test - y_pred), axis=0)
final_test_error = test_mae.mean()

# 保存结果
print(f"\n保存所有 {len(y_test)} 条结果...")
results = []
batch_size = 10000
total_batches = (len(y_test) + batch_size - 1) // batch_size

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(y_test))

    batch_results = []
    for j in range(start_idx, end_idx):
        True_Value = y_test[j]
        Predicted_Value = y_pred[j]
        error = np.abs(True_Value - Predicted_Value)
        uncertainty = uncertainties_original[j]

        formatted_true_value = ' '.join([f"{val:.6f}" for val in True_Value])
        formatted_predicted_value = ' '.join([f"{val:.6f}" for val in Predicted_Value])
        formatted_error = ' '.join([f"{val:.6f}" for val in error])
        formatted_uncertainty = ' '.join([f"{val:.6f}" for val in uncertainty])

        batch_results.append([formatted_true_value, formatted_predicted_value, formatted_error, formatted_uncertainty])

    result_df = pd.DataFrame(batch_results, columns=['True_Value', 'Predicted_Value', 'Error', 'Uncertainty'])
    if batch_idx == 0:
        result_df.to_csv("result_DRF_PyTorch_GPU.csv", index=False)
    else:
        result_df.to_csv("result_DRF_PyTorch_GPU.csv", mode='a', header=False, index=False)

    print(f"  进度: {end_idx}/{len(y_test)} ({end_idx / len(y_test) * 100:.1f}%)")

end_time = time.time()
total_time = end_time - start_time

print(f"\n最终测试误差: {final_test_error:.6f}")
print(f"总耗时: {total_time / 60:.2f} 分钟")

print("\n" + "=" * 50)
print("PyTorch GPU版本特性")
print("=" * 50)
print("✓ 使用PyTorch GPU加速")
print("✓ 神经网络分位数回归")
print("✓ 完整的分布预测")
print("✓ 量化不确定性")
print(f"✓ 使用设备: {models[0].device}")

if final_test_error < 0.5:
    print("🎉 成功！模型平均误差低于目标值 0.5！")
else:
    print(f"📊 当前误差 {final_test_error:.6f}")

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()