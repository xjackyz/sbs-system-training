<template>
  <div class="layout">
    <!-- 顶部导航栏 -->
    <nav class="top-nav">
      <div class="logo">
        <h1>SBS Trading System</h1>
      </div>
      <div class="user-info">
        <span>系统状态: <span class="status-active">运行中</span></span>
      </div>
    </nav>

    <!-- 侧边栏 -->
    <aside class="sidebar">
      <ul class="nav-menu">
        <li>
          <router-link to="/dashboard">
            <i class="fas fa-home"></i>
            <span>系统概览</span>
          </router-link>
        </li>
        <li>
          <router-link to="/training">
            <i class="fas fa-brain"></i>
            <span>模型训练</span>
          </router-link>
          <ul class="submenu">
            <li><router-link to="/training/status">训练状态</router-link></li>
            <li><router-link to="/training/config">训练配置</router-link></li>
            <li><router-link to="/training/metrics">训练指标</router-link></li>
          </ul>
        </li>
        <li>
          <router-link to="/analysis">
            <i class="fas fa-chart-line"></i>
            <span>交易分析</span>
          </router-link>
          <ul class="submenu">
            <li><router-link to="/analysis/market">市场结构</router-link></li>
            <li><router-link to="/analysis/signals">交易信号</router-link></li>
            <li><router-link to="/analysis/performance">绩效分析</router-link></li>
          </ul>
        </li>
        <li>
          <router-link to="/monitor">
            <i class="fas fa-desktop"></i>
            <span>系统监控</span>
          </router-link>
          <ul class="submenu">
            <li><router-link to="/monitor/resources">资源使用</router-link></li>
            <li><router-link to="/monitor/logs">系统日志</router-link></li>
            <li><router-link to="/monitor/alerts">告警信息</router-link></li>
          </ul>
        </li>
        <li>
          <router-link to="/settings">
            <i class="fas fa-cog"></i>
            <span>系统设置</span>
          </router-link>
        </li>
      </ul>
    </aside>

    <!-- 主要内容区域 -->
    <main class="main-content">
      <router-view></router-view>
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-areas:
    "nav nav"
    "sidebar main";
  grid-template-columns: 250px 1fr;
  grid-template-rows: 60px 1fr;
  min-height: 100vh;
}

.top-nav {
  grid-area: nav;
  background: #1a1a1a;
  color: white;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.sidebar {
  grid-area: sidebar;
  background: #2a2a2a;
  color: white;
  padding: 20px 0;
}

.main-content {
  grid-area: main;
  background: #f5f5f5;
  padding: 20px;
}

.nav-menu {
  list-style: none;
  padding: 0;
  margin: 0;
}

.nav-menu li {
  margin: 8px 0;
}

.nav-menu a {
  display: flex;
  align-items: center;
  padding: 10px 20px;
  color: #fff;
  text-decoration: none;
  transition: background 0.3s;
}

.nav-menu a:hover {
  background: #3a3a3a;
}

.nav-menu i {
  margin-right: 10px;
}

.submenu {
  list-style: none;
  padding-left: 40px;
  margin: 5px 0;
}

.submenu a {
  padding: 5px 20px;
  font-size: 0.9em;
}

.status-active {
  color: #4caf50;
  font-weight: bold;
}
</style> 