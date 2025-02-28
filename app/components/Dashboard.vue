<template>
  <div class="dashboard">
    <div class="grid-container">
      <!-- 系统状态卡片 -->
      <div class="card system-status">
        <h2>系统状态</h2>
        <div class="status-grid">
          <div class="status-item">
            <span class="label">训练系统</span>
            <span class="status running">运行中</span>
          </div>
          <div class="status-item">
            <span class="label">交易分析</span>
            <span class="status running">运行中</span>
          </div>
          <div class="status-item">
            <span class="label">监控系统</span>
            <span class="status running">运行中</span>
          </div>
          <div class="status-item">
            <span class="label">Discord Bot</span>
            <span class="status running">运行中</span>
          </div>
        </div>
      </div>

      <!-- 资源使用卡片 -->
      <div class="card resource-usage">
        <h2>资源使用</h2>
        <div class="resource-grid">
          <div class="resource-item">
            <h3>GPU 使用率</h3>
            <div class="progress-bar">
              <div class="progress" style="width: 75%"></div>
            </div>
            <span>75%</span>
          </div>
          <div class="resource-item">
            <h3>内存使用率</h3>
            <div class="progress-bar">
              <div class="progress" style="width: 60%"></div>
            </div>
            <span>60%</span>
          </div>
          <div class="resource-item">
            <h3>CPU 使用率</h3>
            <div class="progress-bar">
              <div class="progress" style="width: 45%"></div>
            </div>
            <span>45%</span>
          </div>
          <div class="resource-item">
            <h3>存储使用率</h3>
            <div class="progress-bar">
              <div class="progress" style="width: 30%"></div>
            </div>
            <span>30%</span>
          </div>
        </div>
      </div>

      <!-- 训练状态卡片 -->
      <div class="card training-status">
        <h2>训练状态</h2>
        <div class="training-info">
          <div class="info-item">
            <span class="label">当前Epoch</span>
            <span class="value">23/100</span>
          </div>
          <div class="info-item">
            <span class="label">Loss</span>
            <span class="value">0.0234</span>
          </div>
          <div class="info-item">
            <span class="label">准确率</span>
            <span class="value">94.5%</span>
          </div>
          <div class="info-item">
            <span class="label">已训练时间</span>
            <span class="value">12h 34m</span>
          </div>
        </div>
      </div>

      <!-- 最近告警卡片 -->
      <div class="card recent-alerts">
        <h2>最近告警</h2>
        <div class="alert-list">
          <div class="alert-item warning">
            <span class="time">10:23</span>
            <span class="message">GPU温度超过阈值(85°C)</span>
          </div>
          <div class="alert-item info">
            <span class="time">09:45</span>
            <span class="message">新的模型检查点已保存</span>
          </div>
          <div class="alert-item success">
            <span class="time">09:30</span>
            <span class="message">训练阶段1完成</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dashboard {
  padding: 20px;
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.card {
  background: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.card h2 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #333;
}

/* 系统状态样式 */
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.9em;
}

.status.running {
  background: #e3f2fd;
  color: #1976d2;
}

/* 资源使用样式 */
.resource-grid {
  display: grid;
  gap: 15px;
}

.progress-bar {
  background: #eee;
  height: 8px;
  border-radius: 4px;
  margin: 8px 0;
}

.progress {
  height: 100%;
  background: #2196f3;
  border-radius: 4px;
}

/* 训练状态样式 */
.training-info {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}

.info-item {
  display: flex;
  flex-direction: column;
}

.info-item .label {
  color: #666;
  font-size: 0.9em;
}

.info-item .value {
  font-size: 1.2em;
  font-weight: bold;
  color: #333;
}

/* 告警列表样式 */
.alert-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.alert-item {
  display: flex;
  align-items: center;
  padding: 10px;
  border-radius: 4px;
  font-size: 0.9em;
}

.alert-item .time {
  margin-right: 10px;
  color: #666;
}

.alert-item.warning {
  background: #fff3e0;
  color: #e65100;
}

.alert-item.info {
  background: #e3f2fd;
  color: #1976d2;
}

.alert-item.success {
  background: #e8f5e9;
  color: #2e7d32;
}
</style> 