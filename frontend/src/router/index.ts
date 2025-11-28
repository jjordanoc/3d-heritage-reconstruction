import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Viewer from '../views/Viewer.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: Home },
    { 
      path: '/viewer', 
      name: 'viewer', 
      component: Viewer,
      props: { viewType: 'ply' }
    },
    { 
      path: '/splatViewer', 
      name: 'splatViewer', 
      component: Viewer,
      props: { viewType: 'gsplat' }
    },
  ],
})
