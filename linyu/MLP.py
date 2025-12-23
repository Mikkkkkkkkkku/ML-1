import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib

warnings.filterwarnings('ignore')


# GPU配置
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except:
            return False
    return False


use_gpu = setup_gpu()
tf.random.set_seed(42)
np.random.seed(42)

BATCH_SIZE = 2048 if use_gpu else 512


class RobustMLP:
    """稳健的MLP模型"""

    def __init__(self, hidden_layers=[128, 64, 32], learning_rate=0.001,
                 batch_size=2048, epochs=100, l2_reg=0.0001, dropout_rate=0.1):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    def _create_model(self, input_dim, output_dim):
        model = Sequential()

        model.add(Dense(self.hidden_layers[0], activation='relu',
                        kernel_regularizer=l2(self.l2_reg),
                        input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu',
                            kernel_regularizer=l2(self.l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(output_dim, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, X, y):
        print("训练稳健MLP模型...")

        # 使用StandardScaler避免属性错误
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y)

        self.model = self._create_model(X_scaled.shape[1], y_scaled.shape[1])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
        ]

        history = self.model.fit(
            X_scaled, y_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
            validation_split=0.15,
            shuffle=True
        )

        best_val_loss = min(history.history['val_loss'])
        print(f"训练完成, 最佳val_loss: {best_val_loss:.4f}")

        return self

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0, batch_size=self.batch_size)
        return self.scaler_y.inverse_transform(y_pred_scaled)


class SafeHybridModel:
    """安全的混合模型 - 修复所有已知问题"""

    def __init__(self):
        self.models = []
        self.scaler_X = None
        self.scaler_y = None
        self.selected_features_mask = None

    def fit(self, X, y):
        print("训练安全混合模型...")

        # 确保X是numpy数组
        if hasattr(X, 'values'):
            X_data = X.values
        else:
            X_data = X

        # 特征选择和预处理
        X_processed, self.selected_features_mask = self._preprocess_features(X_data, is_training=True)

        # 使用StandardScaler避免属性错误
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X_processed)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y)

        self.models = []

        # 为每个目标变量训练模型
        for i in range(y_scaled.shape[1]):
            print(f"  目标变量 {i + 1}/{y_scaled.shape[1]} ({columns[i]})...")

            # 根据目标变量的特性选择模型
            if i in [1, 2]:  # CO2相关变量
                print("    使用随机森林...")
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42 + i,
                    n_jobs=-1
                )
                model.fit(X_scaled, y_scaled[:, i])
            else:  # 其他变量使用MLP
                print("    使用MLP...")
                if i == 0:  # T_SONIC
                    hidden_layers = [128, 64, 32]
                else:
                    hidden_layers = [64, 32, 16]

                model = RobustMLP(
                    hidden_layers=hidden_layers,
                    learning_rate=0.001,
                    batch_size=BATCH_SIZE,
                    epochs=100,
                    l2_reg=0.0001,
                    dropout_rate=0.1
                )
                model.fit(X_scaled, y_scaled[:, i].reshape(-1, 1))

            self.models.append(model)

            # 立即验证模型性能
            y_pred_temp = self._predict_single_model(i, X_scaled)
            mae = mean_absolute_error(y_scaled[:, i], y_pred_temp)
            print(f"    训练集MAE: {mae:.4f}")

        return self

    def _preprocess_features(self, X, is_training=False):
        """安全的特征预处理"""
        if is_training:
            # 训练时：选择特征并保存mask
            stds = np.std(X, axis=0)
            self.selected_features_mask = stds > 1e-6
            X_processed = X[:, self.selected_features_mask]
            print(f"特征选择: {X.shape[1]} -> {X_processed.shape[1]} 个特征")
            return X_processed, self.selected_features_mask
        else:
            # 测试时：使用训练时保存的mask
            if self.selected_features_mask is None:
                raise ValueError("必须先训练模型才能进行预测")
            X_processed = X[:, self.selected_features_mask]
            return X_processed

    def _predict_single_model(self, model_idx, X_scaled):
        """预测单个模型"""
        model = self.models[model_idx]
        if isinstance(model, RandomForestRegressor):
            return model.predict(X_scaled)
        else:
            pred_scaled = model.model.predict(X_scaled, verbose=0, batch_size=BATCH_SIZE)
            return pred_scaled.ravel()

    def predict(self, X):
        # 确保X是numpy数组
        if hasattr(X, 'values'):
            X_data = X.values
        else:
            X_data = X

        # 使用训练时的特征选择
        X_processed = self._preprocess_features(X_data, is_training=False)
        X_scaled = self.scaler_X.transform(X_processed)

        predictions = []
        for i, model in enumerate(self.models):
            pred_scaled = self._predict_single_model(i, X_scaled)

            # 安全的反标准化 - 使用StandardScaler的正确属性
            if hasattr(self.scaler_y, 'mean_'):
                pred = pred_scaled * self.scaler_y.scale_[i] + self.scaler_y.mean_[i]
            else:
                # 备用方案
                pred = pred_scaled * np.std(y_train[:, i]) + np.mean(y_train[:, i])

            predictions.append(pred.ravel())

        return np.column_stack(predictions)


def safe_feature_engineering(data, reference_columns=None):
    """安全的特征工程"""
    features = data.copy()

    # 确保只处理数值列
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()

    # 基础统计特征
    if numeric_columns:
        features['feature_mean'] = features[numeric_columns].mean(axis=1)
        features['feature_std'] = features[numeric_columns].std(axis=1)

    # 噪声相关特征
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                     'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    available_noise_columns = [col for col in noise_columns if col in numeric_columns]

    if available_noise_columns:
        features['noise_mean'] = features[available_noise_columns].mean(axis=1)
        features['noise_std'] = features[available_noise_columns].std(axis=1)

    # 处理缺失值
    features = features.fillna(0)

    # 移除无限大的值
    features = features.replace([np.inf, -np.inf], 0)

    # 如果提供了参考列，确保特征顺序一致
    if reference_columns is not None:
        # 添加缺失的列
        for col in reference_columns:
            if col not in features.columns:
                features[col] = 0
        # 按参考列排序
        features = features[reference_columns]

    print(f"特征工程后数据形状: {features.shape}")
    return features


def safe_preprocess_data(data):
    """安全的数据预处理"""
    data_processed = data.copy()

    # 转换数据类型
    for col in data_processed.columns:
        if data_processed[col].dtype == 'object':
            try:
                data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')
            except:
                data_processed = data_processed.drop(columns=[col])

    # 填充缺失值
    data_processed = data_processed.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"预处理后数据形状: {data_processed.shape}")
    return data_processed


# 主程序
start_time = time.time()

print("加载数据...")
try:
    train_dataSet = pd.read_csv('modified_数据集Time_Series661_detail.dat')
    test_dataSet = pd.read_csv('modified_数据集Time_Series662_detail.dat')
    print(f"训练集形状: {train_dataSet.shape}, 测试集形状: {test_dataSet.shape}")
except Exception as e:
    print(f"数据加载错误: {e}")
    exit()

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

print("数据预处理...")
train_data_processed = safe_preprocess_data(train_dataSet)
test_data_processed = safe_preprocess_data(test_dataSet)

print("特征工程...")
# 先处理训练集
X_train_enhanced = safe_feature_engineering(train_data_processed)

# 获取训练集的特征列作为参考
reference_columns = X_train_enhanced.columns.tolist()

# 处理测试集时使用相同的特征列
X_test_enhanced = safe_feature_engineering(test_data_processed, reference_columns)

# 提取目标变量
y_train = train_dataSet[columns].values
y_test = test_dataSet[columns].values

print(f"数据形状 - 训练集: {X_train_enhanced.shape}, 测试集: {X_test_enhanced.shape}")
print(f"目标变量形状 - 训练集: {y_train.shape}, 测试集: {y_test.shape}")

# 检查特征数量是否一致
if X_train_enhanced.shape[1] != X_test_enhanced.shape[1]:
    print(
        f"警告: 训练集和测试集特征数量不一致! 训练集: {X_train_enhanced.shape[1]}, 测试集: {X_test_enhanced.shape[1]}")
    # 强制对齐特征
    common_columns = list(set(X_train_enhanced.columns) & set(X_test_enhanced.columns))
    X_train_enhanced = X_train_enhanced[common_columns]
    X_test_enhanced = X_test_enhanced[common_columns]
    print(f"对齐后特征数量: {X_train_enhanced.shape[1]}")


# 数据采样函数
def sample_training_data(X, y, sample_ratio=0.8):
    n_samples = int(len(X) * sample_ratio)
    indices = np.random.choice(len(X), n_samples, replace=False)
    return X.iloc[indices], y[indices]


print("训练数据采样 (80%)...")
X_train_sampled, y_train_sampled = sample_training_data(X_train_enhanced, y_train, 0.8)
print(f"采样后形状 - 训练集: {X_train_sampled.shape}")

print("开始训练安全模型...")
model = SafeHybridModel()
model.fit(X_train_sampled.values, y_train_sampled)

print("测试集预测...")
batch_size = 10000
y_pred_parts = []
for i in range(0, len(X_test_enhanced), batch_size):
    end_idx = min(i + batch_size, len(X_test_enhanced))
    X_batch = X_test_enhanced.values[i:end_idx]
    y_pred_batch = model.predict(X_batch)
    y_pred_parts.append(y_pred_batch)
    print(f"  测试集预测进度: {end_idx}/{len(X_test_enhanced)}")

y_pred = np.vstack(y_pred_parts)

print("计算训练集预测...")
y_train_pred_parts = []
for i in range(0, len(X_train_sampled), batch_size):
    end_idx = min(i + batch_size, len(X_train_sampled))
    X_batch = X_train_sampled.values[i:end_idx]
    y_pred_batch = model.predict(X_batch)
    y_train_pred_parts.append(y_pred_batch)

y_train_pred = np.vstack(y_train_pred_parts)

# 计算误差
train_mae = np.mean(np.abs(y_train_sampled - y_train_pred), axis=0)
test_mae = np.mean(np.abs(y_test - y_pred), axis=0)

final_train_error = train_mae.mean()
final_test_error = test_mae.mean()

# 结果分析
print("\n" + "=" * 60)
print("安全模型性能分析")
print("=" * 60)
print(f"最终训练误差: {final_train_error:.6f}")
print(f"最终测试误差: {final_test_error:.6f}")

print("\n各特征详细误差:")
for i, col in enumerate(columns):
    status = "🎯" if test_mae[i] < 0.1 else "✅" if test_mae[i] < 0.5 else "⚠️" if test_mae[i] < 1.0 else "❌"
    print(f"  {status} {col}: {test_mae[i]:.6f}")

# 保存结果
print(f"\n保存结果...")
batch_size = 10000

for i in range(0, len(y_test), batch_size):
    end_idx = min(i + batch_size, len(y_test))

    batch_results = []
    for j in range(i, end_idx):
        True_Value = y_test[j]
        Predicted_Value = y_pred[j]
        error = np.abs(True_Value - Predicted_Value)

        formatted_true_value = ' '.join([f"{val:.6f}" for val in True_Value])
        formatted_predicted_value = ' '.join([f"{val:.6f}" for val in Predicted_Value])
        formatted_error = ' '.join([f"{val:.6f}" for val in error])

        batch_results.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(batch_results, columns=['True_Value', 'Predicted_Value', 'Error'])
    if i == 0:
        result_df.to_csv("result_SafeHybridModel.csv", index=False)
    else:
        result_df.to_csv("result_SafeHybridModel.csv", mode='a', header=False, index=False)

    print(f"  保存进度: {end_idx}/{len(y_test)}")

print(f"结果已保存到: result_SafeHybridModel.csv")

end_time = time.time()
total_time = end_time - start_time

print(f"\n总耗时: {total_time / 60:.2f} 分钟")

# 总结
print("\n" + "=" * 60)
print("模型训练总结")
print("=" * 60)

if final_test_error < 0.1:
    print("🎉 优秀！模型平均误差低于 0.1！")
elif final_test_error < 0.5:
    print("✅ 良好！模型平均误差低于 0.5！")
elif final_test_error < 1.0:
    print("⚠️ 一般！模型需要进一步优化")
else:
    print(f"❌ 当前误差 {final_test_error:.6f}，需要重大改进！")

# 内存清理
import gc

gc.collect()