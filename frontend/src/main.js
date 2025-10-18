import { createApp } from 'vue'
import App from './App.vue'
import router from './router'   // ← resolvdrá a ./router/index.ts

createApp(App)
  .use(router)
  .mount('#app')
