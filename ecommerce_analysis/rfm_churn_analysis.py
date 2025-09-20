import pandas as pd
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题

# --- 数据计算与准备函数 (无变化) ---
def calculate_rfm(df):
    print("--- 步骤1: 计算 RFM 指标 ---")
    snapshot_date = df['order_date'].max() + dt.timedelta(days=1)
    rfm = df.groupby('user_id').agg({
        'order_date': lambda date: (snapshot_date - date.max()).days,
        'order_id': 'count',
        'amount': 'sum'
    })
    rfm.rename(columns={'order_date': 'R', 'order_id': 'F', 'amount': 'M'}, inplace=True)
    print("✅ RFM 指标计算完成。")
    return rfm


def find_optimal_k_and_save_plot(rfm_df, output_path):
    print("--- 步骤2: 使用肘部法则寻找最佳 K 值 ---")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('K-Means Clustering - Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ 肘部法则图已保存至: {output_path}")


def perform_kmeans_clustering(rfm_df, n_clusters):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['R', 'F', 'M']])
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    print(f"✅ 已成功将用户分为 {n_clusters} 个群组。")
    return rfm_df


# --- 用户分层与策略制定函数 ---
def map_clusters_to_segments(rfm_df):
    """根据聚类结果的RFM均值，为每个簇定义业务标签和策略。"""
    print("--- 步骤3: 定义用户分层与营销策略 ---")
    cluster_profiles = rfm_df.groupby('Cluster')[['R', 'F', 'M']].mean().reset_index()

    sorted_clusters = cluster_profiles.sort_values(by=['M', 'F', 'R'], ascending=[False, False, True]).reset_index(
        drop=True)

    segment_map = {
        sorted_clusters.loc[0, 'Cluster']: ("高价值核心用户", "提供VIP服务, 新品优先体验, 定制化关怀"),
        sorted_clusters.loc[1, 'Cluster']: ("潜力增长用户", "发放优惠券, 推荐相关产品, 激励其提高消费频率"),
        sorted_clusters.loc[2, 'Cluster']: ("需要唤醒的客户", "通过邮件/短信进行召回, 提供限时优惠"),
        sorted_clusters.loc[3, 'Cluster']: ("低价值或流失客户", "维持基本服务, 减少营销投入")
    }

    rfm_df['Segment'] = rfm_df['Cluster'].map(lambda x: segment_map.get(x, ("未知", "无"))[0])
    rfm_df['Strategy'] = rfm_df['Cluster'].map(lambda x: segment_map.get(x, ("未知", "无"))[1])

    print("✅ 已为所有用户分配业务标签和营销策略。")
    print("\n--- 用户分层概览 ---")
    # --- 这里是修改的地方 ---
    print(rfm_df.groupby('Segment').size())

    return rfm_df


# --- 可视化报告生成函数 (无变化) ---
def save_cluster_visualizations(rfm_df, output_path):
    """生成并保用户分群的可视化图表。"""
    print("--- 步骤4: 生成并保存可视化报告 ---")

    palette = sns.color_palette("coolwarm", n_colors=rfm_df['Cluster'].nunique())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('用户分群可视化报告', fontsize=16)

    sns.scatterplot(data=rfm_df, x='R', y='M', hue='Segment', palette=palette, ax=axes[0])
    axes[0].set_title('消费近度(R) vs 消费金额(M)')

    sns.scatterplot(data=rfm_df, x='F', y='M', hue='Segment', palette=palette, ax=axes[1])
    axes[1].set_title('消费频率(F) vs 消费金额(M)')

    sns.scatterplot(data=rfm_df, x='R', y='F', hue='Segment', palette=palette, ax=axes[2])
    axes[2].set_title('消费近度(R) vs 消费频率(F)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ 用户分群可视化图表已保存至: {output_path}")


# --- 主程序 (无变化) ---
if __name__ == '__main__':
    PROCESSED_DATA_FILE = 'data/processed/ecommerce_orders.csv'
    ELBOW_PLOT_FILE = 'reports/figures/kmeans_elbow_plot.png'
    VISUALIZATION_REPORT_FILE = 'reports/figures/cluster_visualizations.png'
    STRATEGY_OUTPUT_FILE = 'output/customer_segmentation_report.csv'

    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"❌ 错误: 未找到分析所需的文件 '{PROCESSED_DATA_FILE}'。")
        print("   请先运行 'prepare_data_for_analysis.py' 脚本。")
    else:
        df_orders = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['order_date'])

        rfm_table = calculate_rfm(df_orders)

        find_optimal_k_and_save_plot(rfm_table[['R', 'F', 'M']], ELBOW_PLOT_FILE)
        OPTIMAL_K = 4
        rfm_with_clusters = perform_kmeans_clustering(rfm_table, OPTIMAL_K)

        rfm_final_report = map_clusters_to_segments(rfm_with_clusters)

        save_cluster_visualizations(rfm_final_report, VISUALIZATION_REPORT_FILE)

        try:
            os.makedirs(os.path.dirname(STRATEGY_OUTPUT_FILE), exist_ok=True)
            rfm_final_report.to_csv(STRATEGY_OUTPUT_FILE, index=True)
            print(f"--- 步骤5: 产出最终报告文件 ---")
            print(f"✅ 包含用户分层和营销策略的完整报告已保存至: {STRATEGY_OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ 错误: 保存策略文件失败. {e}")

        churn_risk_users = rfm_final_report[rfm_final_report['Segment'] == '需要唤醒的客户']
        print("\n--- 摘要：需要重点关注的'需要唤醒的客户'列表 (前10名) ---")
        print(churn_risk_users.head(10))