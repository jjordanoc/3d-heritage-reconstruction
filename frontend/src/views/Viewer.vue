<template>
  <div class="page">
    <Suspense>
      <template #default>
        <AsyncPLYViewer />
      </template>
      <template #fallback>
        <div class="loading">Cargando visor…</div>
      </template>
    </Suspense>
  </div>
</template>

<script setup>
import { defineAsyncComponent } from 'vue'
const AsyncPLYViewer = defineAsyncComponent(() => import('@/components/PLYViewer.vue'))
</script>

<style scoped>
.page {
  width: 100%;
  height: 100vh;        /* ocupa todo el viewport */
  background: #0b0d12;  /* consistente con el visor */
  display: grid;
  place-items: center;
}
.loading {
  color: #e6e9ef;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
</style>
