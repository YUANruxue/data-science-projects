import pandas as pd
import os


def prepare_superstore_for_rfm_analysis(raw_file_path, output_file_path):
    """
    加载原始的 Superstore 数据集，进行预处理，并将其转换为
    适用于 RFM 和流失用户分析的格式。
    """
    print("--- 开始加载和处理 Superstore 数据 ---")

    # 1. 加载数据
    try:
        df = pd.read_csv(raw_file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 at '{raw_file_path}'")
        print("   请确保您的原始数据 'SampleSuperstore.csv' 已放置在 'data/raw/' 文件夹下。")
        return False

    # 2. 定义列映射并筛选
    column_mapping = {
        'Customer ID': 'user_id',
        'Order ID': 'order_id',
        'Order Date': 'order_date',
        'Sales': 'amount'
    }

    df_processed = df[list(column_mapping.keys())].copy()
    df_processed.rename(columns=column_mapping, inplace=True)
    print("✅ 1. 已筛选并重命名列")

    # 3. 数据清洗和类型转换
    df_processed['order_date'] = pd.to_datetime(df_processed['order_date'], errors='coerce')
    original_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    cleaned_rows = len(df_processed)
    print(f"✅ 2. 已处理数据类型并清理了 {original_rows - cleaned_rows} 行无效数据")

    # 4. 保存处理后的文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_processed.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"✅ 3. Superstore数据处理完成！")
        print(f"   干净的数据已保存至: {output_file_path}")
        return True
    except Exception as e:
        print(f"❌ 错误: 保存文件失败. {e}")
        return False


if __name__ == '__main__':
    # 定义原始文件和目标文件的路径
    RAW_DATA_FILE = 'data/raw/SampleSuperstore.csv'
    PROCESSED_DATA_FILE = 'data/processed/ecommerce_orders.csv'

    # 执行处理函数
    prepare_superstore_for_rfm_analysis(RAW_DATA_FILE, PROCESSED_DATA_FILE)