<template>
  <div class="container">
    <div class="header">
      <h1>{{ playerName }}</h1>
      <a href="/" class="back-button">返回首页</a>
    </div>
    <div v-if="loading" class="loading">
      正在加载球员数据...
    </div>
    <div v-else>
      <div class="player-info">
        <h2>生涯数据</h2>
        <div class="stats-grid">
          <div class="stat-item" v-for="stat in basicStats" :key="stat.label">
            <div class="stat-label">{{ stat.label }}</div>
            <div class="stat-value">{{ stat.value }}</div>
          </div>
        </div>
      </div>
      <div class="charts-grid">
        <!-- 显示基础数据趋势图 -->
        <BasicStatsChart :data="playerData" />
        <!-- 显示雷达图 -->
        <RadarChart :data="playerData" />
      </div>
      <!-- 预测信息区域 -->
      <PredictChart v-if="playerData.predictions" :predictions="playerData.predictions" />
    </div>
  </div>
</template>

<script>
import BasicStatsChart from '../components/BasicStatsChart.vue'
import RadarChart from '../components/RadarChart.vue'
import PredictChart from '../components/PredictChart.vue'

export default {
  name: 'PlayerDetail',
  components: {
    BasicStatsChart,
    RadarChart,
    PredictChart
  },
  data() {
    return {
      playerData: null,
      loading: true,
      error: null,
      playerName: ''
    }
  },
  computed: {
    basicStats() {
      if (!this.playerData || !this.playerData.career_stats) return [];
      const cs = this.playerData.career_stats;
      // 处理位置字段，将“-”替换为“/”
      let positions = cs.Pos || 'N/A';
      if (typeof positions === 'string') {
        positions = positions.replace(/-/g, '/');
      }
      return [
        { label: '位置', value: positions },
        { label: '生涯场均得分', value: cs.PTS ? cs.PTS.toFixed(1) : 'N/A' },
        { label: '生涯场均篮板', value: cs.TRB ? cs.TRB.toFixed(1) : 'N/A' },
        { label: '生涯场均助攻', value: cs.AST ? cs.AST.toFixed(1) : 'N/A' },
        { label: '生涯场均抢断', value: cs.STL ? cs.STL.toFixed(1) : 'N/A' },
        { label: '生涯场均盖帽', value: cs.BLK ? cs.BLK.toFixed(1) : 'N/A' },
        { label: '生涯比赛', value: cs.G || 'N/A' },
        { label: '生涯赛季', value: cs.Seasons || 'N/A' },
        { label: '职业生涯', value: (cs.Year_Start && cs.Year_End) ? `${cs.Year_Start}-${cs.Year_End}` : 'N/A' }
      ];
    }
  },
  methods: {
    async loadPlayerData() {
      try {
        // 从 URL 中提取球员名称，例如 /player/<playerName>
        const pathParts = window.location.pathname.split('/');
        this.playerName = decodeURIComponent(pathParts[pathParts.length - 1]);
        document.title = `${this.playerName} - 球员数据`;
        const response = await fetch('/api/players/evaluate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          },
          body: JSON.stringify({ player_name: this.playerName })
        });
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.error) {
          throw new Error(data.error);
        }
        this.playerData = data;
      } catch (error) {
        console.error('获取数据失败:', error);
        this.error = error.message || '获取数据失败';
      } finally {
        this.loading = false;
      }
    }
  },
  mounted() {
    this.loadPlayerData();
  }
}
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
  background-color: #f5f5f5;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}
.back-button {
  padding: 8px 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  text-decoration: none;
}
.loading {
  text-align: center;
  padding: 20px;
  font-size: 18px;
  color: #666;
}
.player-info {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 20px;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 15px;
  margin-top: 15px;
}
.stat-item {
  text-align: center;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 4px;
}
.stat-label {
  font-size: 14px;
  color: #666;
}
.stat-value {
  font-size: 20px;
  font-weight: bold;
  color: #333;
  margin-top: 5px;
}
.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  margin-top: 20px;
}
</style>
