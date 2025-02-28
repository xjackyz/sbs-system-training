import pytest
import pandas as pd
import os
from src.self_supervised.trainer.self_supervised_trainer import DataIncrementalLoader

@pytest.fixture
def setup_data_file(tmpdir):
    data = "date,price\n2023-01-01,100\n2023-01-02,200\n"
    data_file = tmpdir.join("test_data.csv")
    with open(data_file, 'w') as f:
        f.write(data)
    return str(data_file)

def test_data_incremental_loader(setup_data_file):
    loader = DataIncrementalLoader(setup_data_file, batch_size=1)
    batch, pos = loader.get_next_batch()
    assert batch is not None, "数据加载失败"
    assert len(batch) == 1, "批次大小不正确"
    assert 'date' in batch.columns, "缺少日期列"
    assert 'price' in batch.columns, "缺少价格列"

    # 测试增量加载
    batch, pos = loader.get_next_batch()
    assert batch is not None, "数据加载失败"
    assert len(batch) == 1, "批次大小不正确"

    # 测试没有更多数据
    for _ in range(2):
        loader.get_next_batch()
    batch, pos = loader.get_next_batch()
    assert batch is None, "应无更多数据" 