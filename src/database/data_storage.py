import sqlite3


def initialize_database(db_connection):
    """初始化数据库，创建所需的表"""
    try:
        conn = sqlite3.connect(db_connection['db_name'])
        cursor = conn.cursor()
        
        # 创建 training_data 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column1 TEXT,
                column2 REAL,
                -- 添加其他列
            )
        ''')
        
        # 创建 training_results 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT,
                value REAL
            )
        ''')
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"数据库初始化错误: {e}")
    finally:
        if conn:
            conn.close()


def save_training_data(dataset, training_results, db_connection):
    """
    保存训练数据到数据库
    :param dataset: 训练数据集
    :param training_results: 训练结果
    :param db_connection: 数据库连接参数
    """
    try:
        conn = sqlite3.connect(db_connection['db_name'])
        cursor = conn.cursor()
        
        # 保存数据集
        dataset.to_sql('training_data', conn, if_exists='replace', index=False)
        
        # 保存训练结果
        cursor.execute('''INSERT INTO training_results (metric, value) VALUES (?, ?)''',
                       (training_results['best_metric'], training_results['value']))
        conn.commit()
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if conn:
            conn.close()  # 确保连接被关闭 