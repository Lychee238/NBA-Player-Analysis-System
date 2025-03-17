import { createRouter, createWebHistory } from 'vue-router';
import Home from './views/Home.vue'; // 导入首页组件
import PlayerDetail from './views/PlayerDetail.vue'; // 导入球员详情页组件

const routes = [
  {
    path: '/', // 首页路由
    name: 'Home',
    component: Home,
  },
  {
    path: '/player/:name', // 球员详情页路由，动态路由参数
    name: 'PlayerDetail',
    component: PlayerDetail,
    props: true, // 将路由参数作为 props 传递给组件
  },
];

const router = createRouter({
  history: createWebHistory(), // 使用 history 模式（去掉 URL 中的 #）
  routes,
});

export default router;