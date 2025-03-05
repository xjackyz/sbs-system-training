import sqlite3
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_database(db_connection):
    """初始化数据库，创建所需的表"""
    try:
        with sqlite3.connect(db_connection['db_name']) as conn:
            cursor = conn.cursor()
            
            # 创建 training_data 表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    column1 TEXT NOT NULL,
                    column2 REAL NOT NULL,
                    column3 INTEGER
                )
            ''')
            
            # 创建 training_results 表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL
                )
            ''')
            
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"数据库初始化错误: {e}")


def save_training_data(dataset, training_results, db_connection):
    """
    保存训练数据到数据库
    :param dataset: 训练数据集，必须是 Pandas DataFrame
    :param training_results: 训练结果，包含最佳指标和对应值
    :param db_connection: 数据库连接参数，包含 db_name
    """
    # 数据验证
    if not isinstance(dataset, pd.DataFrame):
        logger.error("dataset 必须是 Pandas DataFrame")
        return
    if 'best_metric' not in training_results or 'value' not in training_results:
        logger.error("training_results 必须包含 'best_metric' 和 'value' 字段")
        return
    
    try:
        with sqlite3.connect(db_connection['db_name']) as conn:
            cursor = conn.cursor()
            
            # 保存数据集
            dataset.to_sql('training_data', conn, if_exists='replace', index=False)
            
            # 保存训练结果
            cursor.execute('''INSERT INTO training_results (metric, value) VALUES (?, ?)''',
                           (training_results['best_metric'], training_results['value']))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"数据库错误: {e}")
    except Exception as e:
        logger.error(f"发生错误: {e}") 