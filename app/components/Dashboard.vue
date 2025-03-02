<template>
  <div class="dashboard">
    <div class="grid-container">
      <!-- 系统状态卡片 -->
      <div class="card system-status">
        <h2>系统状态</h2>
        <div class="status-grid">
          <div class="status-item">
            <span class="label">训练系统</span>
            <span class="status" :class="systemStatus.training_system ? 'running' : 'stopped'">
              {{ systemStatus.training_system ? '运行中' : '已停止' }}
            </span>
          </div>
          <div class="status-item">
            <span class="label">交易分析</span>
            <span class="status" :class="systemStatus.trading_analysis ? 'running' : 'stopped'">
              {{ systemStatus.trading_analysis ? '运行中' : '已停止' }}
            </span>
          </div>
          <div class="status-item">
            <span class="label">监控系统</span>
            <span class="status" :class="systemStatus.monitoring_system ? 'running' : 'stopped'">
              {{ systemStatus.monitoring_system ? '运行中' : '已停止' }}
            </span>
          </div>
          <div class="status-item">
            <span class="label">Discord Bot</span>
            <span class="status" :class="systemStatus.discord_bot ? 'running' : 'stopped'">
              {{ systemStatus.discord_bot ? '运行中' : '已停止' }}
            </span>
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
              <div class="progress" :style="{ width: resourceUsage.gpu_usage + '%' }"></div>
            </div>
            <span>{{ Math.round(resourceUsage.gpu_usage) }}%</span>
          </div>
          <div class="resource-item">
            <h3>内存使用率</h3>
            <div class="progress-bar">
              <div class="progress" :style="{ width: resourceUsage.memory_usage + '%' }"></div>
            </div>
            <span>{{ Math.round(resourceUsage.memory_usage) }}%</span>
          </div>
          <div class="resource-item">
            <h3>CPU 使用率</h3>
            <div class="progress-bar">
              <div class="progress" :style="{ width: resourceUsage.cpu_usage + '%' }"></div>
            </div>
            <span>{{ Math.round(resourceUsage.cpu_usage) }}%</span>
          </div>
          <div class="resource-item">
            <h3>存储使用率</h3>
            <div class="progress-bar">
              <div class="progress" :style="{ width: resourceUsage.storage_usage + '%' }"></div>
            </div>
            <span>{{ Math.round(resourceUsage.storage_usage) }}%</span>
          </div>
        </div>
      </div>

      <!-- 训练状态卡片 -->
      <div class="card training-status">
        <h2>训练状态</h2>
        <div class="training-info">
          <div class="info-item">
            <span class="label">当前Epoch</span>
            <span class="value">{{ trainingMetrics.current_epoch }}/{{ trainingMetrics.total_epochs }}</span>
          </div>
          <div class="info-item">
            <span class="label">Loss</span>
            <span class="value">{{ trainingMetrics.loss }}</span>
          </div>
          <div class="info-item">
            <span class="label">准确率</span>
            <span class="value">{{ trainingMetrics.accuracy }}%</span>
          </div>
          <div class="info-item">
            <span class="label">已训练时间</span>
            <span class="value">{{ trainingMetrics.training_time }}</span>
          </div>
        </div>
      </div>

      <!-- 最近告警卡片 -->
      <div class="card recent-alerts">
        <h2>最近告警</h2>
        <div class="alert-list">
          <div v-for="(alert, index) in alerts" :key="index" class="alert-item" :class="alert.level">
            <span class="time">{{ alert.time }}</span>
            <span class="message">{{ alert.message }}</span>
          </div>
          <div v-if="alerts.length === 0" class="no-alerts">
            暂无告警信息
          </div>
        </div>
      </div>

      <!-- 连接状态 -->
      <div class="card connection-status">
        <h2>WebSocket连接状态</h2>
        <div class="status-indicator">
          <span class="status-dot" :class="{ connected: websocketConnected }"></span>
          <span>{{ websocketConnected ? '已连接' : '未连接' }}</span>
        </div>
        <div class="last-update">
          <span>最后更新时间: {{ lastUpdateTime }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Dashboard',
  data() {
    return {
      systemStatus: {
        training_system: false,
        trading_analysis: false,
        monitoring_system: false,
        discord_bot: false
      },
      resourceUsage: {
        gpu_usage: 0,
        memory_usage: 0,
        cpu_usage: 0,
        storage_usage: 0
      },
      trainingMetrics: {
        current_epoch: 0,
        total_epochs: 100,
        loss: 0,
        accuracy: 0,
        training_time: '0h 0m'
      },
      alerts: [],
      websocket: null,
      websocketConnected: false,
      lastUpdateTime: '未更新',
      retryCount: 0,
      maxRetryCount: 5,
      retryInterval: 3000 // 3秒后重试
    }
  },
  created() {
    this.fetchInitialData();
    this.connectWebSocket();
  },
  beforeUnmount() {
    this.disconnectWebSocket();
  },
  methods: {
    async fetchInitialData() {
      try {
        // 获取系统状态
        const statusResponse = await fetch('/api/system/status');
        if (statusResponse.ok) {
          this.systemStatus = await statusResponse.json();
        }
        
        // 获取资源使用情况
        const resourcesResponse = await fetch('/api/system/resources');
        if (resourcesResponse.ok) {
          this.resourceUsage = await resourcesResponse.json();
        }
        
        // 获取训练指标
        const metricsResponse = await fetch('/api/training/metrics');
        if (metricsResponse.ok) {
          this.trainingMetrics = await metricsResponse.json();
        }
        
        // 获取告警信息
        const alertsResponse = await fetch('/api/alerts/recent');
        if (alertsResponse.ok) {
          this.alerts = await alertsResponse.json();
        }
        
        this.updateLastUpdateTime();
      } catch (error) {
        console.error('获取初始数据失败:', error);
      }
    },
    
    connectWebSocket() {
      // 根据当前环境确定WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      
      // 创建WebSocket连接
      this.websocket = new WebSocket(wsUrl);
      
      // 连接建立时
      this.websocket.onopen = () => {
        this.websocketConnected = true;
        this.retryCount = 0;
        console.log('WebSocket连接已建立');
      };
      
      // 接收消息时
      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // 根据消息类型处理不同的数据
          if (data.type === 'resources') {
            this.resourceUsage = data.data;
          } else if (data.type === 'system_status') {
            this.systemStatus = data.data;
          } else if (data.type === 'training_metrics') {
            this.trainingMetrics = data.data;
          } else if (data.type === 'alert') {
            // 新告警放在最前面
            this.alerts.unshift(data.data);
            // 限制显示的告警数量
            if (this.alerts.length > 10) {
              this.alerts = this.alerts.slice(0, 10);
            }
          }
          
          this.updateLastUpdateTime();
        } catch (error) {
          console.error('处理WebSocket消息失败:', error);
        }
      };
      
      // 连接关闭时
      this.websocket.onclose = () => {
        this.websocketConnected = false;
        console.log('WebSocket连接已关闭');
        
        // 尝试重新连接
        if (this.retryCount < this.maxRetryCount) {
          this.retryCount++;
          console.log(`${this.retryInterval / 1000}秒后尝试重新连接 (${this.retryCount}/${this.maxRetryCount})`);
          setTimeout(() => this.connectWebSocket(), this.retryInterval);
        }
      };
      
      // 连接出错时
      this.websocket.onerror = (error) => {
        this.websocketConnected = false;
        console.error('WebSocket连接错误:', error);
      };
    },
    
    disconnectWebSocket() {
      if (this.websocket) {
        this.websocket.close();
        this.websocket = null;
      }
    },
    
    updateLastUpdateTime() {
      const now = new Date();
      const hours = now.getHours().toString().padStart(2, '0');
      const minutes = now.getMinutes().toString().padStart(2, '0');
      const seconds = now.getSeconds().toString().padStart(2, '0');
      this.lastUpdateTime = `${hours}:${minutes}:${seconds}`;
    }
  }
}
</script>

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

.status.stopped {
  background: #ffebee;
  color: #d32f2f;
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
  transition: width 0.3s ease;
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

.alert-item.error {
  background: #ffebee;
  color: #d32f2f;
}

.no-alerts {
  color: #757575;
  font-style: italic;
  text-align: center;
  padding: 20px 0;
}

/* 连接状态样式 */
.connection-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #f44336;
}

.status-dot.connected {
  background-color: #4caf50;
}

.last-update {
  font-size: 0.8em;
  color: #757575;
}
</style> 