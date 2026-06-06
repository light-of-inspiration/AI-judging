# -*- coding: utf-8 -*-

isJudgeTime = False # 预测罪名

isJudgeTime = True  # 预测刑期


import pandas as pd
import numpy as np
import time
import warnings

from PIL.ImageColor import colormap
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # 样本不平衡
import shap
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Heiti TC']  
plt.rcParams['axes.unicode_minus'] = False 

def load_data(file_path):
    """Excel"""
    df = pd.read_excel(file_path)
    print(f"df.shape: {df.shape}")
    print(f"df.head():\n{df.head()}")
    return df


def preprocess_data(df):
    df_processed = df.copy()

    X_features = [
        # '案由',
        '被告人人数',  '前科1名称',
        '文化程度编码',  '前科3名称', '是否再犯', '前科2名称',
        # '职业', '户籍地','民族', '性别', '年龄',
        '前科数量', 'A行为类型', 'C1次数', 'C2转出资金（千元）',
        'C3非法获利（千元）', 'C4介绍他人', 'C5联系上线', 'O上游犯罪',
        '退出+', '自首+', '坦白 +', '认罪认罚从宽', '从犯+', 'B参与时点',
        'D 竞合关系'
    ]

    actual_features = []
    for feature in X_features:
        possible_names = [
            feature,
            feature.strip(),
            feature.replace(' ', ''),
            feature.replace('+', ' +'),
            feature.replace('+', '+ ')
        ]

        found = False
        for name in possible_names:
            if name in df.columns:
                actual_features.append(name)
                found = True
                break

        if not found:
            print(f"  '{feature}' not found")


    # 处理NaN
    nan_count = df['罪名'].isna().sum()
    print(f"nan_count: {nan_count} ")

    # 删除NaN行
    if nan_count > 0:
        df_clean = df.dropna(subset=['罪名']).copy()
        print(f"df_clean.shape: {df_clean.shape}")
    else:
        df_clean = df.copy()

    X = df_clean[actual_features].copy()
    y = df_clean['罪名'].copy()
    if isJudgeTime:
        y = df_clean['刑期总和（月）'].copy()

    y_counts = y.value_counts()

    print(y_counts)
    print(f"总类别数: {y.nunique()}")

    # 缺失值
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('未知')
    for col in X.select_dtypes(include=['number']).columns:
        X[col] = X[col].fillna(X[col].median())

    # 分类变量编码
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)}类")

    # 目标变量编码
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y)

    flag ='罪名'
    if isJudgeTime:
        flag = '刑期'
    for i, class_name in enumerate(y_le.classes_):
        count = (y_encoded == i).sum()
        print(f"  类别{i}({class_name}): {count}个样本")

    return X, y_encoded, actual_features, label_encoders, y_le


# 样本不平衡
def handle_imbalance(X, y, min_samples=5):
    """SMOTE"""
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    # print(f"类别分布: {class_dist}")

    # 样本数少于指定阈值
    small_classes = [cls for cls, count in class_dist.items() if count < min_samples]
    if small_classes:
        mask = ~np.isin(y, small_classes)
        X_filtered = X[mask]
        y_filtered = y[mask]

        unique_filtered, counts_filtered = np.unique(y_filtered, return_counts=True)
        class_dist_filtered = dict(zip(unique_filtered, counts_filtered))
        print(f"Updated: {class_dist_filtered}")

        X, y = X_filtered, y_filtered

    # 过采样
    try:
        min_class_count = min(np.unique(y, return_counts=True)[1])
        k_neighbors = min(5, min_class_count - 1)
        if k_neighbors < 1:
            k_neighbors = 1

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
        class_dist_resampled = dict(zip(unique_resampled, counts_resampled))
        # print(f"SMOTE: {class_dist_resampled}")

        return X_resampled, y_resampled
    except Exception as e:
        # 随机过采样
        from sklearn.utils import resample

        X_resampled = []
        y_resampled = []

        max_samples = max(np.unique(y, return_counts=True)[1])

        for cls in np.unique(y):
            X_cls = X[y == cls]
            y_cls = y[y == cls]

            # 样本数少于最大样本数，进行过采样
            if len(X_cls) < max_samples:
                X_cls_resampled = resample(X_cls, replace=True, n_samples=max_samples, random_state=42)
                y_cls_resampled = np.array([cls] * max_samples)
            else:
                X_cls_resampled = X_cls
                y_cls_resampled = y_cls

            X_resampled.append(X_cls_resampled)
            y_resampled.append(y_cls_resampled)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
        class_dist_resampled = dict(zip(unique_resampled, counts_resampled))
        # print(f" {class_dist_resampled}")

        return X_resampled, y_resampled


# BP NN
def train_bp_neural_network(X, y, test_size=0.3, random_state=42):

    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练参数
    params = {
        'hidden_layer_sizes': (50,),  # 单隐藏层，50个神经元
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1.0,  # L2正则化
        'learning_rate_init': 0.1,
        'max_iter': 1000,
        'random_state': random_state,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'verbose': False
    }

    param_display = {
        '隐藏层结构': params['hidden_layer_sizes'],
        '激活函数': params['activation'],
        '求解器': params['solver'],
        'L2正则项': params['alpha'],
        '学习率': params['learning_rate_init'],
        '最大迭代次数': params['max_iter'],
        '早停': params['early_stopping']
    }

    for key, value in param_display.items():
        print(f"  {key}: {value}")

    model = MLPClassifier(**params)
    model.fit(X_train_scaled, y_train)

    training_time = time.time() - start_time

    print(f"\n training_time: {training_time:.3f} s")
    print(
        f"模型结构: 输入层({X_train.shape[1]}) -> 隐藏层({params['hidden_layer_sizes'][0]}) -> 输出层({len(np.unique(y))})")

    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler, training_time, params


# 评估
def evaluate_model(model, X_train, X_test, y_train, y_test, y_encoder):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(X_train)
        y_test_prob = model.predict_proba(X_test)

    # 训练集
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred, average='weighted')  # 使用加权平均
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # 测试集
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # AUC
    train_auc, test_auc = 0, 0
    if hasattr(model, 'predict_proba'):
        try:
            train_auc = roc_auc_score(y_train, y_train_prob, multi_class='ovr', average='weighted')
            test_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted')
        except:
            try:
                train_auc = roc_auc_score(y_train, y_train_prob, multi_class='ovo', average='weighted')
                test_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovo', average='weighted')
            except:
                print("Failed")
                train_auc = 0
                test_auc = 0

    # 表格
    evaluation_df = pd.DataFrame({
        '指标': ['准确率', '召回率', '精确率', 'F1', 'AUC'],
        '训练集': [train_accuracy, train_recall, train_precision, train_f1, train_auc],
        '测试集': [test_accuracy, test_recall, test_precision, test_f1, test_auc]
    })

    # 输出
    evaluation_df['训练集'] = evaluation_df['训练集'].apply(lambda x: f"{x:.3f}")
    evaluation_df['测试集'] = evaluation_df['测试集'].apply(lambda x: f"{x:.3f}")

    print("\n模型性能指标 (加权平均):")
    print("-" * 50)
    print(f"{'指标':<10}{'训练集':<10}{'测试集':<10}")
    print("-" * 50)
    for _, row in evaluation_df.iterrows():
        print(f"{row['指标']:<10}{row['训练集']:<10}{row['测试集']:<10}")
    print("-" * 50)

    print("\n测试集分类:")
    print("-" * 60)

    if hasattr(y_encoder, 'classes_'):
        target_names = [str(cls) for cls in y_encoder.classes_]
    else:
        target_names = [f'类别{i}' for i in range(len(np.unique(y_test)))]

    test_unique_classes = np.unique(y_test)
    if len(test_unique_classes) < len(target_names):
        target_names = [target_names[cls] for cls in test_unique_classes]
        label_map = {old: new for new, old in enumerate(test_unique_classes)}
        y_test_mapped = np.array([label_map[val] for val in y_test])
        y_test_pred_mapped = np.array([label_map[val] for val in y_test_pred])

        print(classification_report(y_test_mapped, y_test_pred_mapped,
                                    target_names=target_names, digits=3))
    else:
        print(classification_report(y_test, y_test_pred,
                                    target_names=target_names, digits=3))

    return evaluation_df


# SHAP
def shap_analysis(model, X_train, X_test, feature_names, sample_size=50):
    if len(X_train) > sample_size:
        print(f"sample_size: {sample_size}")
        background = X_train[:sample_size]
        test_sample = X_test[:min(sample_size, len(X_test))]
    else:
        background = X_train
        test_sample = X_test

    try:
        explainer = shap.KernelExplainer(model.predict_proba, background)

        # === C ===
        plt.rcParams['image.cmap'] = 'RdBu'

        # 计算SHAP值
        shap_values = explainer.shap_values(test_sample)

        # 可视化
        plt.figure(figsize=(12, 8))

        # === C ===
        blue_rgb = (255,0,149)
        purp_rgb = (106,0,255)
        pink_rgb = (0,149,255)

        colors = [blue_rgb, purp_rgb, pink_rgb]  
        n_bins = 100  # 分段数
        cmap_name = 'my_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # 绘制
        if isinstance(shap_values, list):
            # 多分类
            shap.summary_plot(shap_values, test_sample, feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.title("特征重要性排序（基于SHAP值，多分类）", fontsize=16)
        else:
            # 二分类
            shap.summary_plot(shap_values, test_sample, feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.title("特征重要性排序（基于SHAP值，二分类）", fontsize=16)


        plt.tight_layout()
        plt.savefig('SHAP特征重要性.png', dpi=300, bbox_inches='tight', facecolor='white')
        # plt.show()

        print("\n前10个最重要特征:")
        print("-" * 50)

        # 平均绝对SHAP值
        if isinstance(shap_values, list):
            # 多分类（取所有类别的平均值）
            mean_shap_values = np.abs(shap_values).mean(axis=0).mean(axis=0)
        else:
            # 二分类
            mean_shap_values = np.abs(shap_values).mean(axis=0)

        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            '特征': feature_names,
            '重要性': mean_shap_values
        }).sort_values('重要性', ascending=False)

        for i, row in feature_importance.head(10).iterrows():
            print(f"{i + 1:2d}. {row['特征']:<20} 重要性: {row['重要性']:.4f}")

        return feature_importance

    except Exception as e:
        print(f"Error with SHAP: {e}")

        try:
            # 权重作为替代
            if hasattr(model, 'coefs_'):
                # 第一层权重的绝对值均值
                weights = np.abs(model.coefs_[0])
                feature_importance_values = weights.mean(axis=1)

                feature_importance = pd.DataFrame({
                    '特征': feature_names,
                    '重要性': feature_importance_values
                }).sort_values('重要性', ascending=False)

                plt.figure(figsize=(14, 10))

                feature_color = (255 / 255, 0 / 255, 81 / 255, 0.8)  # RGB(255,0,81) 带透明度

                plt.barh(range(min(20, len(feature_names))),
                         feature_importance['重要性'].head(20)[::-1],
                         # color=feature_color,
                         edgecolor='black',
                         linewidth=0.5,
                         height=0.7)

                plt.yticks(range(min(20, len(feature_names))),
                           feature_importance['特征'].head(20)[::-1])
                plt.xlabel('特征重要性（基于权重）')
                plt.title('特征重要性排序')
                plt.tight_layout()
                plt.savefig('特征重要性_基于权重.png', dpi=300, bbox_inches='tight')
                # plt.show()

                print("\n前10个最重要特征（基于模型权重）:")
                print("-" * 50)
                for i, row in feature_importance.head(10).iterrows():
                    print(f"{i + 1:2d}. {row['特征']:<20} 重要性: {row['重要性']:.4f}")

                return feature_importance

        except Exception as e2:
            print(f"基于权重的特征重要性也失败: {e2}")

            # 最后使用简单的方法：特征的标准差作为重要性
            print("使用特征标准差作为重要性度量...")
            feature_std = np.std(X_train, axis=0)
            feature_importance = pd.DataFrame({
                '特征': feature_names,
                '重要性': feature_std
            }).sort_values('重要性', ascending=False)

            print("\n前10个最重要特征（基于标准差）:")
            print("-" * 50)
            for i, row in feature_importance.head(10).iterrows():
                print(f"{i + 1:2d}. {row['特征']:<20} 重要性: {row['重要性']:.4f}")

            return feature_importance


def print_model_params(training_time, params):
    """模型参数"""

    params_dict = {
        '参数名': ['训练用时', '数据切分比例', '数据洗牌', '交叉验证',
                   '激活函数', '求解器', '学习率', 'L2正则项',
                   '迭代次数', '隐藏第1层神经元数量', '早停策略', '验证集比例'],
        '参数值': [f"{training_time:.3f}s", '0.7', '是', '否',
                   params['activation'], params['solver'],
                   str(params['learning_rate_init']), str(params['alpha']),
                   str(params['max_iter']), str(params['hidden_layer_sizes'][0]),
                   str(params['early_stopping']), str(params['validation_fraction'])]
    }

    params_df = pd.DataFrame(params_dict)
    print(params_df.to_string(index=False))


# 非结构化数据
def predict_defendant_info(df, model, scaler, label_encoders, y_encoder, feature_names,
                           target_names=["唐鉴炎", "李双喜"]):

    if '被告人' not in df.columns:
        raise ValueError("数据中无'被告人'列，请确认列名正确性")

    target_df = df[df['被告人'].isin(target_names)].copy()
    if target_df.empty:
        print(f"未找到被告人：{', '.join(target_names)}")
        return pd.DataFrame()

    # 去重
    target_df = target_df.drop_duplicates(subset=['被告人'], keep='first')

    X_target = target_df[feature_names].copy()

    # 缺失值
    for col in X_target.select_dtypes(include=['object']).columns:
        X_target[col] = X_target[col].fillna('未知')
    for col in X_target.select_dtypes(include=['number']).columns:
        X_target[col] = X_target[col].fillna(X_target[col].median())

    # 特征编码
    for col in X_target.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            X_target[col] = X_target[col].astype(str)
            X_target[col] = np.where(
                X_target[col].isin(le.classes_),
                X_target[col].map(dict(zip(le.classes_, le.transform(le.classes_)))),
                0
            )
            X_target[col] = X_target[col].astype(int)

    # 数据标准化
    X_target_scaled = scaler.transform(X_target)

    # 实际值
    actual_vals = []
    if not isJudgeTime:
        actual_vals = target_df['罪名'].values
    else:
        actual_vals = target_df['刑期总和（月）'].values

    pred_encoded = model.predict(X_target_scaled)
    pred_vals = []

    if not isJudgeTime:
        pred_vals = y_encoder.inverse_transform(pred_encoded)
    else:
        pred_vals = y_encoder.inverse_transform(pred_encoded)

    result_df = pd.DataFrame({
        '被告人姓名': target_df['被告人'].values,
        f'实际{"罪名" if not isJudgeTime else "刑期（月）"}': actual_vals,
        f'预测{"罪名" if not isJudgeTime else "刑期（月）"}': pred_vals
    })

    result_df = result_df[result_df['被告人姓名'].isin(target_names)]
    result_df['被告人姓名'] = pd.Categorical(result_df['被告人姓名'], categories=target_names, ordered=True)
    result_df = result_df.sort_values('被告人姓名').reset_index(drop=True)

    return result_df


def main():
    try:
        df = load_data('LAIC.xlsx')
    except FileNotFoundError:
        print("LAIC.xlsx not found")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 数据预处理
    X, y, feature_names, label_encoders, y_encoder = preprocess_data(df)

    #样本不平衡
    X_resampled, y_resampled = handle_imbalance(X, y, min_samples=2)

    # 训练模型
    model, X_train_scaled, X_test_scaled, y_train, y_test, scaler, training_time, params = train_bp_neural_network(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )

    # 评估模型
    evaluation_df = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, y_encoder)

    # 模型参数
    print_model_params(training_time, params)

    # SHAP
    feature_importance = shap_analysis(model, X_train_scaled, X_test_scaled, feature_names)

    # 保存结果
    evaluation_df.to_csv('模型评估结果.csv', index=False, encoding='utf-8-sig')

    if feature_importance is not None:
        feature_importance.to_csv('特征重要性.csv', index=False, encoding='utf-8-sig')

    model_params_df = pd.DataFrame({
        '参数名': ['训练用时', '数据切分比例', '数据洗牌', '交叉验证',
                   '激活函数', '求解器', '学习率', 'L2正则项',
                   '迭代次数', '隐藏层神经元数量'],
        '参数值': [f"{training_time:.3f}s", '0.7', '是', '否',
                   params['activation'], params['solver'],
                   str(params['learning_rate_init']), str(params['alpha']),
                   str(params['max_iter']), str(params['hidden_layer_sizes'])]
    })
    model_params_df.to_csv('模型参数.csv', index=False, encoding='utf-8-sig')

    print(f"测试集准确率: {float(evaluation_df.loc[evaluation_df['指标'] == '准确率', '测试集'].values[0]):.3f}")

    # 非结构化数据
    target_defendants = ["唐鉴炎", "李双喜"]
    pred_result_df = predict_defendant_info(
        df=df,
        model=model,
        scaler=scaler,
        label_encoders=label_encoders,
        y_encoder=y_encoder,
        feature_names=feature_names,
        target_names=target_defendants
    )

    # 对比结果
    if not pred_result_df.empty:
        print("\n" + "=" * 60)
        print(f"被告人{'罪名' if not isJudgeTime else '刑期'} 实际值 vs 预测值")
        print("=" * 60)
        print(pred_result_df.to_string(index=False))

        pred_result_df.to_csv(f"被告人{'罪名' if not isJudgeTime else '刑期'}预测结果.csv",
                              index=False, encoding='utf-8-sig')
    else:
        print("\n未获取到被告人预测结果")


if __name__ == "__main__":
    main()