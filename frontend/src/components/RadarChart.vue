<template>
  <div class="chart-container">
    <canvas ref="chartCanvas"></canvas>
  </div>
</template>

<script>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import Chart from 'chart.js/auto'

export default {
  name: 'RadarChart',
  props: {
    data: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    const chartCanvas = ref(null)
    let chartInstance = null

    const createRadarChart = () => {
      if (!props.data || !props.data.career_stats) {
        console.warn('雷达图：缺少 career_stats 数据')
        return
      }

      const metrics = [
        { key: 'PTS', label: '场均得分', color: '#4CAF50', maxValue: 35 },
        { key: 'TRB', label: '场均篮板', color: '#2196F3', maxValue: 16 },
        { key: 'AST', label: '场均助攻', color: '#FF9800', maxValue: 12 },
        { key: 'STL', label: '场均抢断', color: '#795548', maxValue: 3.5 },
        { key: 'BLK', label: '场均盖帽', color: '#607D8B', maxValue: 4 }
      ]

      const careerStats = props.data.career_stats
      const positionAverages = props.data.position_averages || {}

      // 检查并打印接收到的原始数据
      console.log('接收到的原始数据:', props.data)
      console.log('生涯数据:', careerStats)

      let playerPosition = careerStats.Pos || 'Unknown'
      if (typeof playerPosition === 'string') {
        playerPosition = playerPosition.replace(/-/g, '/')
      }

      // 计算每个指标的场均值
      const playerValues = metrics.map(metric => {
        // 获取原始统计数据（已经是场均值）
        let avgValue = parseFloat(careerStats[metric.key]) || 0
        
        // 标准化到0-100的范围
        let normalizedValue = Math.min(100, (avgValue / metric.maxValue) * 100)
        
        console.log(`${metric.label} - 场均值: ${avgValue.toFixed(2)}, 标准化值: ${normalizedValue.toFixed(2)}%`)
        
        return normalizedValue
      })

      // 计算位置平均值
      const positionValues = metrics.map(metric => {
        let value = parseFloat(positionAverages[metric.key]) || 0
        let normalizedValue = Math.min(100, (value / metric.maxValue) * 100)
        
        console.log(`${metric.label} 位置平均值: ${value.toFixed(2)}, 标准化值: ${normalizedValue.toFixed(2)}%`)
        
        return normalizedValue
      })

      console.log('雷达图 - 球员数据:', playerValues)
      console.log('雷达图 - 位置平均:', positionValues)

      if (chartInstance) {
        chartInstance.destroy()
      }

      chartInstance = new Chart(chartCanvas.value, {
        type: 'radar',
        data: {
          labels: metrics.map(m => m.label),
          datasets: [
            {
              label: '球员数据',
              data: playerValues,
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              borderColor: 'rgb(255, 99, 132)',
              pointBackgroundColor: metrics.map(m => m.color),
              pointBorderColor: '#fff',
              pointHoverBackgroundColor: '#fff',
              pointHoverBorderColor: metrics.map(m => m.color)
            },
            {
              label: `${playerPosition}位置平均值`,
              data: positionValues,
              backgroundColor: 'rgba(54, 162, 235, 0.2)',
              borderColor: 'rgb(54, 162, 235)',
              pointBackgroundColor: 'rgb(54, 162, 235)',
              pointBorderColor: '#fff',
              pointHoverBackgroundColor: '#fff',
              pointHoverBorderColor: 'rgb(54, 162, 235)'
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: `球员能力雷达图（与${playerPosition}位置平均值对比）`,
              font: { size: 16 }
            },
            tooltip: {
              callbacks: {
                label: function (context) {
                  const value = context.raw
                  const maxValue = metrics[context.dataIndex].maxValue
                  const originalValue = ((value / 100) * maxValue).toFixed(1)
                  return `${context.dataset.label}: ${originalValue}（${value.toFixed(1)}%）`
                }
              }
            }
          },
          scales: {
            r: {
              angleLines: { display: true },
              suggestedMin: 0,
              suggestedMax: 100,
              ticks: {
                callback: value => value + '%'
              }
            }
          }
        }
      })
    }

    onMounted(() => {
      createRadarChart()
    })

    watch(() => props.data, () => {
      createRadarChart()
    })

    onBeforeUnmount(() => {
      if (chartInstance) chartInstance.destroy()
    })

    return { chartCanvas }
  }
}
</script>

<style scoped>
.chart-container {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
