<template>
  <div class="home">
    <h1>球员评估系统</h1>
    <!-- 将获取到的球员名称传递给搜索框组件 -->
    <SearchBox :players="players" />
  </div>
</template>

<script>
import SearchBox from '../components/searchBox.vue'

export default {
  name: 'Home',
  components: { SearchBox },
  data() {
    return {
      players: []
    }
  },
  mounted() {
    // 从后端接口获取球员名称
    fetch('/api/players/names')
      .then(response => response.json())
      .then(data => {
        if (data.success && Array.isArray(data.players)) {
          this.players = data.players
        } else {
          console.error('获取球员名称失败：', data.error)
        }
      })
      .catch(err => {
        console.error('请求球员名称失败：', err)
      })

    // 保留原有的文件上传代码（如后续需要使用）
    const formData = new FormData()
    fetch('/upload', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          console.log('文件上传成功：', data.message)
        } else {
          console.error('上传失败：', data.error)
        }
      })
      .catch(error => console.error('错误：', error))
  }
}
</script>

<style scoped>
.home {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
h1 {
  color: #2c3e50;
  margin-bottom: 30px;
}
</style>
