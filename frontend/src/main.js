import { createApp } from 'vue';
import App from './App.vue';
import router from './router'; // 导入路由配置
import axios from 'axios'; // 引入 axios
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const app = createApp(App);

// 全局挂载 axios 和 Chart.js
app.config.globalProperties.$http = axios;
app.config.globalProperties.$chart = Chart;

// 配置全局 axios 默认值
axios.defaults.baseURL = 'http://localhost:5000'; // 修改为本地Flask后端地址
axios.defaults.headers.common['Content-Type'] = 'application/json';

// 添加请求拦截器
axios.interceptors.request.use(
  config => {
    console.log('发送请求:', config.url, config);
    return config;
  },
  error => {
    console.error('请求错误:', error);
    return Promise.reject(error);
  }
);

// 添加响应拦截器
axios.interceptors.response.use(
  response => {
    console.log('收到响应:', response.data);
    return response;
  },
  error => {
    console.error('响应错误:', error.response || error);
    return Promise.reject(error);
  }
);

app.use(router); // 使用路由
app.mount('#app');