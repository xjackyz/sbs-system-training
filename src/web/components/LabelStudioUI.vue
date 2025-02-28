<!-- Label Studio集成界面组件 -->
<template>
  <div class="label-studio-container">
    <!-- 项目管理 -->
    <div class="project-section">
      <h2>项目管理</h2>
      
      <!-- 创建项目表单 -->
      <el-form :model="projectForm" label-width="100px">
        <el-form-item label="项目名称">
          <el-input v-model="projectForm.name" placeholder="请输入项目名称"></el-input>
        </el-form-item>
        <el-form-item label="项目描述">
          <el-input type="textarea" v-model="projectForm.description" placeholder="请输入项目描述"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="createProject">创建项目</el-button>
        </el-form-item>
      </el-form>
      
      <!-- 项目列表 -->
      <div class="project-list">
        <h3>项目列表</h3>
        <el-table :data="projects" style="width: 100%">
          <el-table-column prop="id" label="ID" width="80"></el-table-column>
          <el-table-column prop="name" label="名称"></el-table-column>
          <el-table-column prop="description" label="描述"></el-table-column>
          <el-table-column label="操作" width="200">
            <template #default="scope">
              <el-button size="small" @click="viewProject(scope.row)">查看</el-button>
              <el-button size="small" type="danger" @click="deleteProject(scope.row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
    
    <!-- 标注任务管理 -->
    <div class="task-section" v-if="currentProject">
      <h2>标注任务管理</h2>
      
      <!-- 导入任务 -->
      <div class="import-tasks">
        <el-upload
          class="upload-demo"
          action="#"
          :on-change="handleFileChange"
          :auto-upload="false">
          <el-button type="primary">选择文件</el-button>
        </el-upload>
        <el-button type="success" @click="importTasks">导入任务</el-button>
      </div>
      
      <!-- 任务列表 -->
      <div class="task-list">
        <h3>任务列表</h3>
        <el-table :data="tasks" style="width: 100%">
          <el-table-column prop="id" label="ID" width="80"></el-table-column>
          <el-table-column prop="data" label="数据"></el-table-column>
          <el-table-column prop="status" label="状态"></el-table-column>
          <el-table-column label="操作" width="200">
            <template #default="scope">
              <el-button size="small" @click="annotateTask(scope.row)">标注</el-button>
              <el-button size="small" type="danger" @click="deleteTask(scope.row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
    
    <!-- 统计信息 -->
    <div class="stats-section" v-if="currentProject">
      <h2>统计信息</h2>
      <el-row :gutter="20">
        <el-col :span="8">
          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <span>总任务数</span>
              </div>
            </template>
            <div class="item">
              {{ stats.totalTasks }}
            </div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <span>已完成标注</span>
              </div>
            </template>
            <div class="item">
              {{ stats.completedTasks }}
            </div>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <span>标注准确率</span>
              </div>
            </template>
            <div class="item">
              {{ stats.accuracy }}%
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script>
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'

export default {
  name: 'LabelStudioUI',
  
  setup() {
    // 项目表单数据
    const projectForm = reactive({
      name: '',
      description: ''
    })
    
    // 状态数据
    const projects = ref([])
    const tasks = ref([])
    const currentProject = ref(null)
    const stats = ref({
      totalTasks: 0,
      completedTasks: 0,
      accuracy: 0
    })
    
    // 创建项目
    const createProject = async () => {
      try {
        // TODO: 调用后端API创建项目
        ElMessage.success('项目创建成功')
      } catch (error) {
        ElMessage.error('项目创建失败')
      }
    }
    
    // 查看项目
    const viewProject = (project) => {
      currentProject.value = project
      // TODO: 加载项目任务和统计信息
    }
    
    // 删除项目
    const deleteProject = async (project) => {
      try {
        // TODO: 调用后端API删除项目
        ElMessage.success('项目删除成功')
      } catch (error) {
        ElMessage.error('项目删除失败')
      }
    }
    
    // 处理文件选择
    const handleFileChange = (file) => {
      // TODO: 处理选择的文件
    }
    
    // 导入任务
    const importTasks = async () => {
      try {
        // TODO: 调用后端API导入任务
        ElMessage.success('任务导入成功')
      } catch (error) {
        ElMessage.error('任务导入失败')
      }
    }
    
    // 标注任务
    const annotateTask = (task) => {
      // TODO: 打开标注界面
    }
    
    // 删除任务
    const deleteTask = async (task) => {
      try {
        // TODO: 调用后端API删除任务
        ElMessage.success('任务删除成功')
      } catch (error) {
        ElMessage.error('任务删除失败')
      }
    }
    
    return {
      projectForm,
      projects,
      tasks,
      currentProject,
      stats,
      createProject,
      viewProject,
      deleteProject,
      handleFileChange,
      importTasks,
      annotateTask,
      deleteTask
    }
  }
}
</script>

<style scoped>
.label-studio-container {
  padding: 20px;
}

.project-section,
.task-section,
.stats-section {
  margin-bottom: 30px;
}

.project-list,
.task-list {
  margin-top: 20px;
}

.import-tasks {
  margin: 20px 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.box-card {
  margin-bottom: 20px;
}

.item {
  font-size: 24px;
  text-align: center;
  color: #409EFF;
}
</style> 