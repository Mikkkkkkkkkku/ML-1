import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """自定义时间序列数据集"""

    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DNNRegressor(nn.Module):
    """深度神经网络回归模型"""

    def __init__(self, input_size, output_size, hidden_layers=[512, 256, 128, 64], dropout_rate=0.2):
        super(DNNRegressor, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class AdvancedDNNRegressor(nn.Module):
    """更先进的DNN模型，包含残差连接"""

    def __init__(self, input_size, output_size, hidden_layers=[512, 256, 128, 64], dropout_rate=0.2):
        super(AdvancedDNNRegressor, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            block = ResidualBlock(hidden_layers[i], hidden_layers[i + 1], dropout_rate)
            self.residual_blocks.append(block)

        self.output_layer = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, input_size, output_size, dropout_rate):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(output_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)

        # 如果输入输出维度不匹配，使用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DNNTraining:
    """DNN训练类"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=10):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 学习率调度
            if scheduler:
                scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_dnn_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_dnn_model.pth'))


def main():
    start_time = time.time()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    train_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series661_detail.dat')
    test_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series662_detail.dat')

    # 定义列
    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                     'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

    CL = columns + noise_columns

    # 数据质量检查
    data = train_dataSet[CL]
    missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
    missingDf.columns = ['feature', 'miss_num']
    missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]
    print("缺失值比例:")
    print(missingDf)

    # 异常值检测
    outlier_ratios = {}
    for column in CL:
        z_scores = np.abs(stats.zscore(train_dataSet[column]))
        outliers = (z_scores > 2)
        outlier_ratio = outliers.mean()
        outlier_ratios[column] = outlier_ratio

    print("*" * 30)
    print("异常值的比例:")
    for column, ratio in outlier_ratios.items():
        print(f"{column}: {ratio:.2%}")

    # 数据预处理
    X_train = train_dataSet[noise_columns]
    y_train = train_dataSet[columns]
    X_test = test_dataSet[noise_columns]
    y_test = test_dataSet[columns]

    # 特征标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 转换为DataFrame以保持列名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=noise_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=noise_columns)
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=columns)

    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_size = len(noise_columns)
    output_size = len(columns)

    # 可以选择基础DNN或高级DNN
    model = DNNRegressor(input_size, output_size, hidden_layers=[512, 256, 128, 64], dropout_rate=0.2).to(device)
    # model = AdvancedDNNRegressor(input_size, output_size, hidden_layers=[512, 256, 128, 64], dropout_rate=0.2).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练模型
    trainer = DNNTraining(model, device)
    trainer.train(train_loader, val_loader, criterion, optimizer, scheduler, epochs=200, patience=15)

    # 测试模型
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.numpy())

    # 合并预测结果
    predictions_scaled = np.vstack(all_predictions)
    targets_scaled = np.vstack(all_targets)

    # 反标准化
    predictions = scaler_y.inverse_transform(predictions_scaled)
    targets = scaler_y.inverse_transform(targets_scaled)

    # 计算指标
    r2 = r2_score(targets, predictions)
    mse = mean_squared_error(targets, predictions)

    print(f"\nDNN模型性能:")
    print(f'R2 Score: {r2:.6f}')
    print(f'MSE: {mse:.6f}')

    # 保存结果
    results = []
    for true_value, predicted_value in zip(targets, predictions):
        error = np.abs(true_value - predicted_value)
        formatted_true_value = ' '.join(map(str, true_value))
        formatted_predicted_value = ' '.join(map(str, predicted_value))
        formatted_error = ' '.join(map(str, error))
        results.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("result_DNN.csv", index=False)

    # 计算平均误差
    data = pd.read_csv("result_DNN.csv")
    column3 = data.iloc[:, 2]
    numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
    means = numbers.mean()

    print("<*>" * 50)
    print("6个数据的平均误差为：\n", means)
    print(f"总体平均误差: {means.mean():.6f}")

    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.3f}秒")


if __name__ == "__main__":
    main()