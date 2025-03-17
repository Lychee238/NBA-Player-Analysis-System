<template>
  <div class="charts-container">
    <div class="chart-container">
      <canvas ref="basicStatsCanvas"></canvas>
    </div>
    <div class="chart-container">
      <canvas ref="shootingCanvas"></canvas>
    </div>
    <div class="chart-container">
      <canvas ref="defenseCanvas"></canvas>
    </div>
  </div>
</template>

<script>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import Chart from 'chart.js/auto'

export default {
  name: 'BasicStatsChart',
  props: {
    data: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    const basicStatsCanvas = ref(null)
    const shootingCanvas = ref(null)
    const defenseCanvas = ref(null)
    let chartBasic = null
    let chartShooting = null
    let chartDefense = null

    // 定义一个辅助函数，用于创建时间序列图
    function createTimeSeriesChart(canvas, metrics, title, yAxisLabel) {
      // 提取年份数组（拷贝一份，防止后续修改影响其他图表）
      let years = props.data.yearly_stats.map(stat => stat.Year)
      const datasets = []
      
      metrics.forEach(({ key, label, color, isPerGame = false }) => {
        // 提取每个赛季的实际数据
        const actualData = props.data.yearly_stats.map(stat => {
          if (isPerGame) {
            return stat[key] && stat.G ? stat[key] / stat.G : null
          } else {
            let value = stat[key]
            // 如果数据为百分比（key中包含'%'）则乘以100
            if (value !== null && value !== undefined && !isNaN(value) && key.includes('%')) {
              value = parseFloat(value) * 100
            }
            return value
          }
        })
        datasets.push({
          label: label,
          data: actualData,
          borderColor: color,
          backgroundColor: 'transparent',
          fill: false,
          spanGaps: true
        })

        // 如果存在预测数据，则添加预测系列
        if (props.data.predictions && props.data.predictions[key]) {
          const prediction = props.data.predictions[key].value
          const lastYear = parseInt(years[years.length - 1])
          // 构造预测数据：前面用null补位，最后三个点中保留实际数据点，后两点使用预测值
          const predictionData = Array(actualData.length - 1).fill(null)
          predictionData.push(actualData[actualData.length - 1])
          predictionData.push(prediction)
          predictionData.push(prediction)
          // 更新年份：为预测增加两个赛季
          years = years.concat([lastYear + 1, lastYear + 2])
          datasets.push({
            label: `${label}预测`,
            data: predictionData,
            borderColor: color,
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            fill: false,
            spanGaps: true
          })
        }
      })

      return new Chart(canvas, {
        type: 'line',
        data: {
          labels: years,
          datasets: datasets
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: title,
              font: { size: 16 }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              callbacks: {
                label: context => {
                  const value = context.raw
                  return `${context.dataset.label}: ${value !== null ? value.toFixed(1) : 'N/A'}`
                }
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: '赛季' }
            },
            y: {
              title: { display: true, text: yAxisLabel },
              beginAtZero: true,
              ticks: { callback: value => value.toFixed(1) }
            }
          }
        }
      })
    }

    function createAllCharts() {
      // 如果已存在则先销毁
      if (chartBasic) { chartBasic.destroy(); chartBasic = null }
      if (chartShooting) { chartShooting.destroy(); chartShooting = null }
      if (chartDefense) { chartDefense.destroy(); chartDefense = null }

      // 基础数据图表：场均得分、篮板、助攻
      chartBasic = createTimeSeriesChart(
        basicStatsCanvas.value,
        [
          { key: 'PTS', label: '场均得分', color: '#4CAF50', isPerGame: true },
          { key: 'TRB', label: '场均篮板', color: '#2196F3', isPerGame: true },
          { key: 'AST', label: '场均助攻', color: '#FF9800', isPerGame: true }
        ],
        '基础数据趋势及预测',
        '数值'
      )

      // 投篮命中率图表
      chartShooting = createTimeSeriesChart(
        shootingCanvas.value,
        [
          { key: 'FG%', label: '投篮命中率', color: '#E91E63' },
          { key: '3P%', label: '三分命中率', color: '#9C27B0' },
          { key: 'FT%', label: '罚球命中率', color: '#673AB7' }
        ],
        '投篮命中率趋势',
        '百分比'
      )

      // 防守数据图表：场均抢断、盖帽
      chartDefense = createTimeSeriesChart(
        defenseCanvas.value,
        [
          { key: 'STL', label: '场均抢断', color: '#795548', isPerGame: true },
          { key: 'BLK', label: '场均盖帽', color: '#607D8B', isPerGame: true }
        ],
        '防守数据趋势及预测',
        '数值'
      )
    }

    onMounted(() => {
      if (props.data && props.data.yearly_stats) {
        createAllCharts()
      }
    })

    watch(() => props.data, () => {
      if (props.data && props.data.yearly_stats) {
        createAllCharts()
      }
    })

    onBeforeUnmount(() => {
      if (chartBasic) chartBasic.destroy()
      if (chartShooting) chartShooting.destroy()
      if (chartDefense) chartDefense.destroy()
    })

    return {
      basicStatsCanvas,
      shootingCanvas,
      defenseCanvas
    }
  }
}
</script>

<style scoped>
.charts-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.chart-container {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
