<template>
  <div class="prediction-info">
    <h3>下赛季预测</h3>
    <div class="prediction-details">
      <div class="prediction-item" v-for="(prediction, key) in predictions" :key="key">
        <span class="label">{{ getMetricLabel(key) }}:</span>
        <span class="value">
          {{ formatValue(prediction.value, key) }}
          <small>(置信区间: {{ formatBounds(prediction.lower_bound, prediction.upper_bound, key) }})</small>
        </span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PredictChart',
  props: {
    predictions: {
      type: Object,
      required: true
    }
  },
  methods: {
    getMetricLabel(metric) {
      const labels = {
        'PTS': '场均得分',
        'TRB': '场均篮板',
        'AST': '场均助攻',
        'STL': '场均抢断',
        'BLK': '场均盖帽',
        'PER': 'PER效率值',
        'BPM': '正负值',
        'VORP': '替换价值'
      }
      return labels[metric] || metric
    },
    formatValue(value, metric) {
      return metric === 'BPM' ? value.toFixed(1) : Math.max(0, value).toFixed(1)
    },
    formatBounds(lower, upper, metric) {
      if (metric === 'BPM') {
        return `${lower.toFixed(1)} - ${upper.toFixed(1)}`
      } else {
        return `${Math.max(0, lower).toFixed(1)} - ${Math.max(0, upper).toFixed(1)}`
      }
    }
  }
}
</script>

<style scoped>
.prediction-info {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 20px;
}
.prediction-info h3 {
  margin: 0 0 15px 0;
  color: #333;
}
.prediction-details {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}
.prediction-item {
  background: #f8f9fa;
  padding: 10px;
  border-radius: 4px;
}
.prediction-item .label {
  color: #666;
  font-size: 14px;
}
.prediction-item .value {
  display: block;
  font-size: 20px;
  font-weight: bold;
  color: #333;
  margin-top: 5px;
}
</style>
