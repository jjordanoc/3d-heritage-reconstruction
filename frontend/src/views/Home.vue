<!-- Home.vue -->
<template>
  <div>
    <NavBar />
    <section class="hero">
      <h1>Welcome to Our Landing Page</h1>
      <p>Recorre el auditorio en 3D (WASD + mouse)</p>
      <button @click="open = true">Abrir visor 3D</button>
    </section>

    <!-- Modal a pantalla completa -->
    <Teleport to="body">
      <div
        v-if="open"
        style="position:fixed; inset:0; display:flex; flex-direction:column;
               background:rgba(0,0,0,0.6); z-index:9999;">
        <button
          @click="open=false"
          style="position:absolute; top:16px; right:16px; padding:8px 12px; z-index:1;">
          Cerrar
        </button>

        <Suspense>
          <template #default>
            <!-- Contenedor que **sí** ocupa toda la pantalla -->
            <div style="flex:1; min-height:0;">
              <AsyncPLYViewer />
            </div>
          </template>
          <template #fallback>
            <div style="margin:auto; color:#fff; font-family:sans-serif;">Cargando visor…</div>
          </template>
        </Suspense>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, defineAsyncComponent } from 'vue'
import NavBar from '@/components/NavBar.vue' // o ruta relativa
const open = ref(false)

// Si NO tienes alias "@", usa ruta relativa: () => import('../components/PLYViewer.vue')
const AsyncPLYViewer = defineAsyncComponent({
  loader: () => import('@/components/PLYViewer.vue'),
  // opcional: manejo de errores y reintento
  onError(error, retry, fail, attempts) {
    if (attempts <= 2) retry()
    else fail()
    // mira la consola del navegador si falla el import
    console.error('Error cargando PLYViewer:', error)
  }
})
</script>
