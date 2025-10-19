<!-- src/views/Viewer.vue -->
<template>
  <div class="page">
    <Suspense>
      <template #default>
        <AsyncPLYViewer />
      </template>
      <template #fallback>Cargando visor…</template>
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
  height: 100vh;     /* <-- clave: que el contenedor del canvas tenga altura real */
  display: flex;     /* opcional, por si quieres meter más cosas */
  flex-direction: column;
}
</style>
