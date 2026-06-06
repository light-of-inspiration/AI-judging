import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# 中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Win
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel("LAIC.xlsx")

x_candidate_columns = [
    "案由", "被告人人数",
    "年龄",
    # "性别",
    # "民族",
    "前科数量", "前科1名称","前科2名称","前科3名称","是否再犯",
    "文化程度编码",
    "职业",
    "户籍地",
     "A行为类型", "B参与时点",
    "C1次数", "C2转出资金（千元）", "C3非法获利（千元）", "C4介绍他人","C5联系上线",
    "D 竞合关系", "O上游犯罪",
    "退出+", "自首+", "坦白 +", "认罪认罚从宽", "从犯+",
]

y_original_column = "刑期总和（月）"

# 检查
missing_x_cols = [col for col in x_candidate_columns if col not in df.columns]
missing_y_col = [] if y_original_column in df.columns else [y_original_column]
if missing_x_cols or missing_y_col:
    raise ValueError(f"缺失字段：{missing_x_cols + missing_y_col}")

X = df[x_candidate_columns].copy()

y_original = df[y_original_column].copy()
y_original = y_original.fillna(y_original.mean())  # 填充缺失值
y_original = np.where(y_original <= 0, 1e-6, y_original)  # 替换0或负值，避免log(0)报错

# 生成因变量Y：Ln_刑期总和（月）
Y = np.log(y_original)
df['Ln_刑期总和（月）'] = Y  

# 数据预处理
# 缺失值填充
for col in X.columns:
    if X[col].dtype == 'object':
        fill_val = X[col].mode()[0]  # 分类：填充众数
        X[col] = X[col].fillna(fill_val)
    else:
        fill_val = X[col].mean()  # 数值：填充均值
        X[col] = X[col].fillna(fill_val)

categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

for col in categorical_cols:
    X[col] = X[col].astype(str)


# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False,drop='first'), categorical_cols),
#         ('num', 'passthrough', numerical_cols)
#     ]
# )

# Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 分类：填充众数
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))  # 消除VIF=inf
])

# 对数：填充均值
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
])


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression(fit_intercept=True))
])



# 数据切分
test_size = 0.3 
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, shuffle=True, random_state=42
)

# LR
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression(fit_intercept=True))  # fit_intercept=True：拟合截距项
])

# 训练
lr_model.fit(X_train, Y_train)

# 评估
Y_train_pred = lr_model.predict(X_train)
Y_test_pred = lr_model.predict(X_test)

def calc_regression_metrics(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"===== {dataset_name} 模型评估指标 =====")
    print(f"R² 决定系数：{r2:.4f}")
    print(f"平均绝对误差（MAE）：{mae:.4f}")
    print(f"均方误差（MSE）：{mse:.4f}")
    print(f"均方根误差（RMSE）：{rmse:.4f}\n")
    return r2, mae, mse, rmse

train_r2, train_mae, train_mse, train_rmse = calc_regression_metrics(Y_train, Y_train_pred, "训练集")
test_r2, test_mae, test_mse, test_rmse = calc_regression_metrics(Y_test, Y_test_pred, "测试集")

X_processed_full = lr_model.named_steps['preprocessor'].transform(X)
feature_names_full = lr_model.named_steps['preprocessor'].get_feature_names_out()
X_processed_full_df = pd.DataFrame(X_processed_full, columns=feature_names_full)
X_processed_full_sm = sm.add_constant(X_processed_full_df)

# tatsmodels（t值、P值）
sm_model = sm.OLS(Y, X_processed_full_sm).fit()


# 标准化系数（Beta）
def get_standardized_coefficients(sm_model, X_processed_df):
    X_std = X_processed_df.std(axis=0)
    y_std = Y.std()
    coefs = sm_model.params[1:]
    standardized_coefs = coefs * (X_std / y_std)
    standardized_coefs = np.insert(standardized_coefs.values, 0, np.nan)
    return standardized_coefs

standardized_coefs = get_standardized_coefficients(sm_model, X_processed_full_df)


# VIF
def calculate_vif(X_processed_df, feature_names):
    vif_df = pd.DataFrame()
    vif_df["变量名称"] = feature_names
    vif_list = []
    for i in range(X_processed_df.shape[1]):
        try:
            vif_val = variance_inflation_factor(X_processed_df.values, i)
            vif_list.append(vif_val)
        except:
            vif_list.append(np.nan)
    vif_df["VIF"] = vif_list

    vif_index = ["const"] + feature_names.tolist()
    vif_values = [np.nan] + vif_df["VIF"].tolist()
    vif_series = pd.Series(vif_values, index=vif_index)
    return vif_series

vif_series = calculate_vif(X_processed_full_df, feature_names_full)

# ----------------------------------------------------
feature_names_sm = X_processed_full_sm.columns.tolist()
feature_names_sm = ["常数" if x == "const" else x for x in feature_names_sm]

regression_result_df = pd.DataFrame({
    '变量名称': feature_names_sm,
    'B（非标准化系数）': sm_model.params.values,
    '标准误': sm_model.bse.values,
    'Beta（标准化系数）': standardized_coefs,
    't': sm_model.tvalues.values,
    'P': sm_model.pvalues.values
})

regression_result_df['VIF'] = regression_result_df['变量名称'].map(vif_series)


# 显著性标记
def add_significance_marker(p_value):
    if p_value < 0.01:
        return f"{p_value:.3f}***"
    elif p_value < 0.05:
        return f"{p_value:.3f}**"
    elif p_value < 0.1:
        return f"{p_value:.3f}*"
    else:
        return f"{p_value:.3f}"


regression_result_df['P*'] = regression_result_df['P'].apply(add_significance_marker)

# R²、F值
r2 = sm_model.rsquared
adj_r2 = sm_model.rsquared_adj
f_stat = sm_model.fvalue
f_p = sm_model.f_pvalue

regression_result_df['R²'] = ""
regression_result_df['F'] = ""
regression_result_df.loc[regression_result_df['变量名称'] == "常数", 'R²'] = f"{r2:.3f}"
regression_result_df.loc[regression_result_df['变量名称'] == "常数", 'F'] = f"F={f_stat:.3f}\nP={f_p:.3f}***"

# --------------------------
column_order = [
    '变量名称', 'B（非标准化系数）', '标准误', 'Beta（标准化系数）',
    't', 'P*', 'VIF', 'R²', 'F'
]
regression_result_df = regression_result_df[column_order]

for col in ['B（非标准化系数）', '标准误', 'Beta（标准化系数）', 't', 'P']:
    if col in regression_result_df.columns:
        regression_result_df[col] = regression_result_df[col].apply(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "-"
        )

regression_result_df['VIF'] = regression_result_df['VIF'].apply(
    lambda x: f"{x:.3f}" if not pd.isna(x) else "-"
)

# -------------------------- 
print("=" * 150)
print("表 4 线性回归分析结果")
print(f"线性回归分析结果 n={len(df)}")
print("=" * 150)
print(regression_result_df.to_string(index=False))
print("=" * 150)
print(f"因变量：{y_original_column}（定量，Ln_刑期总和（月））")
print("其中，***、**、*分别代表1%、5%、10%的显著性水平。")

# 保存Excel
regression_result_df.to_excel("LR.xlsx", index=False)
print("\n 已保存至“LR.xlsx”")

# -------------------------- 
# 散点图
plt.figure(figsize=(12, 5))

# 子图1：训练集
plt.subplot(1, 2, 1)
plt.scatter(Y_train, Y_train_pred, alpha=0.6, color='blue', label='真实值vs预测值')
min_val = min(Y_train.min(), Y_train_pred.min())
max_val = max(Y_train.max(), Y_train_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合线（y=x）')
plt.xlabel('真实Ln_刑期总和（月）')
plt.ylabel('预测Ln_刑期总和（月）')
plt.title(f'真实值 vs 预测值（R²={train_r2:.4f}）')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：测试集
plt.subplot(1, 2, 2)
plt.scatter(Y_test, Y_test_pred, alpha=0.6, color='green', label='真实值vs预测值')
min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合线（y=x）')
plt.xlabel('真实Ln_刑期总和（月）')
plt.ylabel('预测Ln_刑期总和（月）')
plt.title(f'测试集 真实值 vs 预测值（R²={test_r2:.4f}）')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
# plt.show()

# 残差图
plt.figure(figsize=(12, 5))
# 训练集残差
plt.subplot(1, 2, 1)
residuals_train = Y_train - Y_train_pred
plt.scatter(Y_train_pred, residuals_train, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('预测Ln_刑期总和（月）')
plt.ylabel('残差（真实值-预测值）')
plt.title('训练集 残差图')
plt.grid(alpha=0.3)

# 测试集残差
plt.subplot(1, 2, 2)
residuals_test = Y_test - Y_test_pred
plt.scatter(Y_test_pred, residuals_test, alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('预测Ln_刑期总和（月）')
plt.ylabel('残差（真实值-预测值）')
plt.title('测试集 残差图')
plt.grid(alpha=0.3)

plt.tight_layout()
# plt.show()


def get_simplified_eq_original_vars(lr_model):

    feature_names = lr_model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = lr_model.named_steps['regressor'].coef_
    intercept = lr_model.named_steps['regressor'].intercept_
    dependent_var = "Ln_刑期总和（月）"

    original_var_terms = {}
    for name, coef in zip(feature_names, coefficients):
        if "__" in name:
            var_parts = name.split("__")
            if len(var_parts) >= 2:
                original_var_part = var_parts[1]
                original_var = original_var_part.split("_")[0]
            else:
                original_var = name
        else:
            original_var = name  

        if original_var not in original_var_terms:
            original_var_terms[original_var] = 0.0
        original_var_terms[original_var] += coef

    original_feature_terms = []
    for var_name, total_coef in original_var_terms.items():
        if total_coef >= 0:
            term = f"+ {total_coef:.4f} × {var_name}"
        else:
            term = f"- {abs(total_coef):.4f} × {var_name}"
        original_feature_terms.append(term)

    # 回归方程
    simplified_eq_original = f"{dependent_var} = {intercept:.4f} "
    simplified_eq_original += " ".join(original_feature_terms)

    return simplified_eq_original


# -------------------------- 
simplified_eq_original = get_simplified_eq_original_vars(lr_model)
print("\n===== 简化回归方程 =====")
print(simplified_eq_original)

import pandas as pd
import numpy as np


# --------------------------
def get_target_defendant_prediction(df, trained_model, name_list, id_col="被告人", y_original_col="刑期总和（月）"):
    if id_col not in df.columns:
        raise ValueError(f"数据中无'{id_col}'列")
    target_df = df[df[id_col].str.contains("|".join(name_list), na=False)].copy()
    if target_df.empty:
        print(f"未筛选到被告人：{name_list}")
        return pd.DataFrame()

    x_candidate_columns = [
        "案由", "被告人人数", "性别", "年龄", "民族", "前科1名称",
        "文化程度编码", "职业", "户籍地", "前科3名称", "是否再犯",
        "前科2名称", "前科数量", "A行为类型", "C1次数",
        "C2转出资金（千元）", "C3非法获利（千元）", "C4介绍他人",
        "C5联系上线", "O上游犯罪", "退出+", "自首+",
        "坦白 +", "认罪认罚从宽", "从犯+", "B参与时点", "D 竞合关系"
    ]
    X_target = target_df[x_candidate_columns].copy()

    actual_prison = target_df[y_original_col].values

    ln_pred_prison = trained_model.predict(X_target)

    # 原始刑期 = exp(对数预测刑期)
    pred_prison = np.exp(ln_pred_prison)
    pred_prison = np.where(pred_prison <= 0, 1e-3, pred_prison)

    result_df = pd.DataFrame({
        "被告人姓名": target_df[id_col].values,
        "实际刑期（月）": actual_prison,
        "模型对数预测刑期（Ln_月）": ln_pred_prison,
        "还原后预测刑期（月）": pred_prison
    })

    result_df = result_df.drop_duplicates(subset=["被告人姓名"], keep="first")
    result_df = result_df[result_df["被告人姓名"].isin(name_list)]
    return result_df


# -------------------------- 
target_names = ["唐鉴炎", "李双喜"]

pred_result_df = get_target_defendant_prediction(
    df=df,
    trained_model=lr_model,
    name_list=target_names,
    id_col="被告人",  # 若你的列名是“姓名”“被告人”等，此处修改
    y_original_col="刑期总和（月）"
)

print("===== 被告人唐、李 实际刑期 vs 模型预测刑期 =====")
if not pred_result_df.empty:
    pred_result_df_formatted = pred_result_df.round(2)
    print(pred_result_df_formatted.to_string(index=False))
else:
    print("未获取到结果")

pred_result_df.to_excel("被告人刑期预测结果.xlsx", index=False)
print("\n结果已保存至“被告人刑期预测结果.xlsx”")