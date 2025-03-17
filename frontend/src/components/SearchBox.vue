<template>
  <div class="search-container">
    <input 
      type="text" 
      class="search-input" 
      placeholder="输入球员姓名搜索..."
      v-model="query"
      @input="handleInput"
      @keydown="handleKeyDown"
      autocomplete="off"
    />
    <div class="suggestions" v-show="showSuggestions && filteredPlayers.length > 0">
      <div 
        class="suggestion-item" 
        v-for="(player, index) in filteredPlayers" 
        :key="index"
        :class="{ selected: index === selectedIndex }"
        @click="selectPlayer(player)"
      >
        {{ player }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SearchBox',
  props: {
    players: {
      type: Array,
      default: () => []
    }
  },
  data() {
    return {
      query: '',
      selectedIndex: -1,
      showSuggestions: false
    }
  },
  computed: {
    filteredPlayers() {
      if (!this.query) {
        return []
      }
      // 不区分大小写过滤球员名称
      return this.players.filter(player =>
        player.toLowerCase().includes(this.query.toLowerCase())
      )
    }
  },
  methods: {
    handleInput() {
      this.selectedIndex = -1
      this.showSuggestions = this.query ? true : false
    },
    handleKeyDown(e) {
      const itemsLength = this.filteredPlayers.length
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        this.selectedIndex = Math.min(this.selectedIndex + 1, itemsLength - 1)
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        this.selectedIndex = Math.max(this.selectedIndex - 1, -1)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (this.selectedIndex >= 0 && this.selectedIndex < itemsLength) {
          this.selectPlayer(this.filteredPlayers[this.selectedIndex])
        }
      }
    },
    selectPlayer(player) {
      this.query = player
      this.showSuggestions = false
      // 跳转到球员详情页，使用后端 /player/<player_name> 接口渲染详情页面
      window.location.href = `/player/${encodeURIComponent(player)}`
    },
    handleClickOutside(e) {
      if (!this.$el.contains(e.target)) {
        this.showSuggestions = false
      }
    }
  },
  mounted() {
    document.addEventListener('click', this.handleClickOutside)
  },
  beforeUnmount() {
    document.removeEventListener('click', this.handleClickOutside)
  }
}
</script>

<style scoped>
.search-container {
  background-color: #f5f7fa;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  width: 300px;
  position: relative;
}

.search-input {
  width: 100%;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  outline: none;
  box-sizing: border-box;
}

.search-input:focus {
  border-color: #409eff;
}

.suggestions {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background-color: white;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  margin-top: 5px;
  max-height: 200px;
  overflow-y: auto;
  z-index: 1000;
}

.suggestion-item {
  padding: 8px 12px;
  cursor: pointer;
}

.suggestion-item:hover {
  background-color: #f5f7fa;
}

.selected {
  background-color: #e6f1fc;
}
</style>
