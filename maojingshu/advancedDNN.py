import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
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


class AdvancedDNNRegressor(nn.Module):
    """优化的深度神经网络回归模型"""

    def __init__(self, input_size, output_size, hidden_layers=[1024, 512, 256, 128], dropout_rate=0.3):
        super(AdvancedDNNRegressor, self).__init__()

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            block = ResidualBlock(hidden_layers[i], hidden_layers[i + 1], dropout_rate)
            self.residual_blocks.append(block)

        # 注意力机制
        self.attention = SelfAttention(hidden_layers[-1])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.BatchNorm1d(hidden_layers[-1] // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_layers[-1] // 2, output_size)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        # 应用注意力
        x = self.attention(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """改进的残差块"""

    def __init__(self, input_size, output_size, dropout_rate):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
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
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.leaky_relu(out)

        return out


class SelfAttention(nn.Module):
    """自注意力机制"""

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 自注意力机制
        q = self.query(x).unsqueeze(1)  # (batch, 1, hidden)
        k = self.key(x).unsqueeze(1)  # (batch, 1, hidden)
        v = self.value(x).unsqueeze(1)  # (batch, 1, hidden)

        # 计算注意力分数
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_weights = self.softmax(attn_weights)

        # 应用注意力权重
        attn_output = torch.bmm(attn_weights, v).squeeze(1)

        # 残差连接
        return x + attn_output


class EnsembleModel:
    """集成模型，结合多个DNN预测"""

    def __init__(self, input_size, output_size, n_models=5):
        self.models = []
        self.n_models = n_models
        self.input_size = input_size
        self.output_size = output_size

    def create_models(self, device):
        for i in range(self.n_models):
            # 每个模型使用不同的架构
            hidden_layers = [
                [1024, 512, 256, 128],
                [1024, 512, 256, 128, 64],
                [512, 512, 256, 128],
                [1024, 512, 256],
                [1024, 512, 256, 128, 64, 32]
            ][i % 5]

            model = AdvancedDNNRegressor(
                self.input_size,
                self.output_size,
                hidden_layers=hidden_layers,
                dropout_rate=0.2 + 0.05 * i
            ).to(device)
            self.models.append(model)

    def predict(self, x, device):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(device)).cpu().numpy()
                predictions.append(pred)

        # 平均预测结果
        return np.mean(predictions, axis=0)


class DNNTraining:
    """改进的DNN训练类"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')

    def train(self, train_loader, val_loader, epochs=300, patience=20):
        """训练模型"""
        # 使用组合损失函数
        criterion1 = nn.MSELoss()
        criterion2 = nn.L1Loss()

        # 使用AdamW优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=1,
            eta_min=1e-6
        )

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

                # 组合损失函数
                loss = criterion1(outputs, batch_targets) + 0.3 * criterion2(outputs, batch_targets)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                    loss = criterion1(outputs, batch_targets) + 0.3 * criterion2(outputs, batch_targets)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # 早停
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_dnn_model.pth')
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f} *')
            else:
                patience_counter += 1
                if (epoch + 1) % 20 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_dnn_model.pth'))


def create_polynomial_features(df, degree=2):
    """创建多项式特征"""
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df)
    poly_columns = poly.get_feature_names_out(df.columns)

    return pd.DataFrame(poly_features, columns=poly_columns, index=df.index)


def create_interaction_features(df):
    """创建交互特征"""
    interaction_df = df.copy()
    columns = df.columns

    # 创建两两交互特征
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_name = f"{columns[i]}_{columns[j]}"
            interaction_df[col_name] = df[columns[i]] * df[columns[j]]

    return interaction_df


def main():
    start_time = time.time()

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

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

    # 异常值检测和处理
    outlier_ratios = {}
    for column in CL:
        # 使用更健壮的异常值检测
        Q1 = train_dataSet[column].quantile(0.25)
        Q3 = train_dataSet[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (train_dataSet[column] < lower_bound) | (train_dataSet[column] > upper_bound)
        outlier_ratio = outliers.mean()
        outlier_ratios[column] = outlier_ratio

        # 缩尾处理异常值
        train_dataSet[column] = np.clip(train_dataSet[column], lower_bound, upper_bound)
        test_dataSet[column] = np.clip(test_dataSet[column], lower_bound, upper_bound)

    print("*" * 30)
    print("异常值的比例:")
    for column, ratio in outlier_ratios.items():
        print(f"{column}: {ratio:.2%}")

    # 处理缺失值 - 使用中位数填充
    train_dataSet[CL] = train_dataSet[CL].fillna(train_dataSet[CL].median())
    test_dataSet[CL] = test_dataSet[CL].fillna(test_dataSet[CL].median())

    # 数据预处理
    X_train = train_dataSet[noise_columns]
    y_train = train_dataSet[columns]
    X_test = test_dataSet[noise_columns]
    y_test = test_dataSet[columns]

    # 特征工程 - 创建多项式特征和交互特征
    print("进行特征工程...")
    X_train_enhanced = create_interaction_features(X_train)
    X_test_enhanced = create_interaction_features(X_test)

    # 特征标准化 - 使用RobustScaler对异常值更稳健
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_enhanced)
    X_test_scaled = scaler_X.transform(X_test_enhanced)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 转换为DataFrame以保持列名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_enhanced.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_enhanced.columns)
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=columns)

    print(f"特征工程后特征数量: {X_train_scaled.shape[1]}")

    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

    # 分割训练集和验证集
    train_size = int(0.85 * len(train_dataset))  # 增加训练集比例
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 使用更大的批次大小
    batch_size = 128
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_size = X_train_scaled.shape[1]
    output_size = len(columns)

    # 选择使用集成模型还是单个模型
    use_ensemble = True  # 设置为False使用单个模型

    if use_ensemble:
        print("使用集成模型...")
        ensemble = EnsembleModel(input_size, output_size, n_models=5)
        ensemble.create_models(device)

        # 训练每个模型
        for i, model in enumerate(ensemble.models):
            print(f"训练模型 {i + 1}/{len(ensemble.models)}")
            trainer = DNNTraining(model, device)
            trainer.train(train_loader, val_loader, epochs=200, patience=15)

        # 使用集成模型进行预测
        ensemble_predictions = []
        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_predictions = ensemble.predict(batch_features, device)
                ensemble_predictions.append(batch_predictions)

        predictions_scaled = np.vstack(ensemble_predictions)

    else:
        # 使用单个模型
        model = AdvancedDNNRegressor(
            input_size,
            output_size,
            hidden_layers=[1024, 512, 256, 128],
            dropout_rate=0.3
        ).to(device)

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练模型
        trainer = DNNTraining(model, device)
        trainer.train(train_loader, val_loader, epochs=300, patience=20)

        # 测试模型
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                all_predictions.append(outputs.cpu().numpy())

        predictions_scaled = np.vstack(all_predictions)

    # 合并目标值
    all_targets = []
    with torch.no_grad():
        for _, batch_targets in test_loader:
            all_targets.append(batch_targets.numpy())
    targets_scaled = np.vstack(all_targets)

    # 反标准化
    predictions = scaler_y.inverse_transform(predictions_scaled)
    targets = scaler_y.inverse_transform(targets_scaled)

    # 计算指标
    r2 = r2_score(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(targets - predictions))

    print(f"\nDNN模型性能:")
    print(f'R2 Score: {r2:.6f}')
    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE: {mae:.6f}')

    # 保存结果
    results = []
    for true_value, predicted_value in zip(targets, predictions):
        error = np.abs(true_value - predicted_value)
        formatted_true_value = ' '.join(map(str, true_value))
        formatted_predicted_value = ' '.join(map(str, predicted_value))
        formatted_error = ' '.join(map(str, error))
        results.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("result_DNN_optimized.csv", index=False)

    # 计算平均误差
    data = pd.read_csv("result_DNN_optimized.csv")
    column3 = data.iloc[:, 2]
    numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
    means = numbers.mean()

    print("<*>" * 50)
    print("6个数据的平均误差为：")
    for i, col in enumerate(columns):
        print(f"{col}: {means[i]:.6f}")
    print(f"总体平均误差: {means.mean():.6f}")

    # 绘制训练损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(trainer.train_losses, label='Training Loss')
        plt.plot(trainer.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(trainer.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        plt.tight_layout()
        plt.savefig('training_curves_optimized.png', dpi=300, bbox_inches='tight')
        plt.show()
    except:
        print("无法绘制训练曲线，请安装matplotlib")

    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.3f}秒")


if __name__ == "__main__":
    main()