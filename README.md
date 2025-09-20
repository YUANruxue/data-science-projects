# 电商用户价值与流失分析 (E-commerce Customer Value and Churn Analysis)

本项目通过 RFM 模型和 K-Means 聚类算法，对电商用户的消费行为进行分析，自动化地划分用户价值等级、生成可视化报告，并为每个用户匹配相应的营销策略，最终产出可直接用于业务决策的报告文件。

## 项目结构

```
ecommerce_analysis/
│
├── .gitignore
├── data/
│   ├── raw/
│   │   └── SampleSuperstore.csv
│   └── processed/
├── output/
│   └── customer_segmentation_report.csv  <-- 最终用户策略报告
├── reports/
│   └── figures/
│       ├── kmeans_elbow_plot.png         <-- K-Means 肘部图
│       └── cluster_visualizations.png    <-- 用户分群可视化图
├── prepare_data_for_analysis.py
├── rfm_churn_analysis.py
├── requirements.txt
└── README.md
```

## 环境要求

* Python 3.8+
* 依赖库见 `requirements.txt`

## 使用方法

1.  **克隆项目并进入目录**
2.  **创建并激活虚拟环境**
3.  **安装依赖**: `pip install -r requirements.txt`
4.  **准备数据**: 将原始数据文件 `SampleSuperstore.csv` 放入 `data/raw/` 文件夹。
5.  **运行项目**:
    * 首先，运行数据预处理脚本：
        ```bash
        python prepare_data_for_analysis.py
        ```
    * 然后，运行主分析脚本：
        ```bash
        python rfm_churn_analysis.py
        ```

## 产出文件说明

脚本运行成功后，会自动生成以下文件：

1.  **用户分层策略报告 (`output/customer_segmentation_report.csv`)**
    * 一个包含所有用户的详细表格。
    * 关键列包括：`user_id`, `R`, `F`, `M`, `Cluster`, `Segment` (业务标签), `Strategy` (建议策略)。
    * 此文件可直接交付给市场或运营团队使用。

2.  **可视化报告 (`reports/figures/`)**
    * `kmeans_elbow_plot.png`: 用于判断最佳聚类数量 K 值的技术图表。
    * `cluster_visualizations.png`: 一组散点图，直观展示不同用户群在 R, F, M 维度上的分布情况，帮助理解分群结果。

## 分析流程

1.  **数据预处理**: 清洗原始数据，提取 RFM 分析所需字段。
2.  **RFM 与聚类分析**:
    * 计算用户的 RFM 指标。
    * 通过 K-Means 算法对用户进行自动分群。
3.  **用户分层与策略映射**:
    * 基于每个群组的 RFM 均值，自动为群组赋予业务标签（如“高价值核心用户”、“需要唤醒的客户”等）。
    * 为不同标签的用户匹配预设的营销策略。
4.  **报告生成**:
    * 将带有业务标签和策略的完整用户列表导出为 CSV 文件。
    * 生成并保存用户分群的可视化图表。

---

## 📦 环境依赖
安装依赖：
```bash
pip install -r requirements.txt
```

依赖包包括：
- pandas
- scikit-learn
- matplotlib
- seaborn
- surprise
- pyspark
