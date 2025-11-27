<template>
  <div class="page">
    <!-- Update Feed (replaces Viewer Toggle) -->
    <UpdateFeed ref="updateFeedRef" v-if="projectId" :project-id="projectId" />

    <Suspense>
      <template #default>
        <component 
          :is="currentViewerType === 'ply' ? AsyncPLYViewer : AsyncGSplatViewer"
          ref="viewerRef" 
          @loadComplete="onLoadComplete"
          @model-updated="onModelUpdated"
        />
      </template>
      <template #fallback>
        <div class="loading">Cargando visor…</div>
      </template>
    </Suspense>

    <!-- Loading overlay during re-fetch -->
    <div v-if="reloading" class="loading-overlay">
      <div class="spinner"></div>
      <p>Cargando...</p>
    </div>

    <!-- Overlay inferior -->
    <div class="upload-bar">
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        @change="onFileSelected"
        style="display: none"
      />
      <div v-if="selectedFile" class="file-info">
        <p>{{ selectedFile.name }}</p>
        <button @click="discardFile" class="btn small danger">Descartar</button>
      </div>
      <div v-else>
        <button @click="triggerFileInput" class="btn small">Cargar imagen</button>
      </div>
      <button
        v-if="selectedFile"
        @click="uploadFile"
        class="btn small primary"
        :disabled="uploading"
      >
        {{ uploading ? "Subiendo..." : "Subir imagen" }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, defineAsyncComponent, computed } from 'vue'
import { useRoute } from 'vue-router'
import UpdateFeed from '@/components/UpdateFeed.vue'

const AsyncPLYViewer = defineAsyncComponent(() => import('@/components/PLYViewer.vue'))
const AsyncGSplatViewer = defineAsyncComponent(() => import('@/components/GSplatViewer.vue'))
const viewerRef = ref(null)
const updateFeedRef = ref(null)
const fileInput = ref(null)
const selectedFile = ref(null)
const uploading = ref(false)
const reloading = ref(false)

const route = useRoute()
const projectId = computed(() => (route.query.id || route.params.id)?.toString())
const currentViewerType = computed(() => {
  return route.query.view === 'splat' ? 'gsplat' : 'ply'
})

onMounted(() => {
  document.body.classList.add('no-scroll')
})
onUnmounted(() => {
  document.body.classList.remove('no-scroll')
})

// API base
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

function triggerFileInput() {
  fileInput.value?.click()
}

function onFileSelected(e) {
  const file = e.target.files?.[0]
  if (file) selectedFile.value = file
}

function discardFile() {
  selectedFile.value = null
  fileInput.value.value = ''
}

function onLoadComplete() {
  uploading.value = false
  reloading.value = false
}

function onModelUpdated(metadata) {
  if (updateFeedRef.value && metadata) {
    // Normalize metadata: might be an array or an object
    const updates = Array.isArray(metadata) ? metadata : [metadata]
    
    updates.forEach(meta => {
      const userId = meta.user_id || 'System'
      const imageId = meta.image_id || 'Update'
      updateFeedRef.value.addMessage(userId, imageId)
    })
  }
}

async function uploadFile() {
  if (!selectedFile.value) return
  uploading.value = true

  const id = projectId.value
  if (!id) {
    alert('No se encontró ID del modelo.')
    uploading.value = false
    return
  }

  const form = new FormData()
  form.append('file', selectedFile.value) // backend espera 'file', no 'image'
  
  // Add user_id from localStorage
  const userId = localStorage.getItem('heritage_user') || 'Guest'
  form.append('user_id', userId)

  try {
    const res = await fetch(`${API_BASE}/pointcloud/${encodeURIComponent(id)}`, {
      method: 'POST',
      body: form,
    })

    if (!res.ok) throw new Error(`Error ${res.status}`)
    
    // Expect JSON response {"success": true}
    const data = await res.json()
    
    if (data.success) {
      // NOTE: We don't trigger reload here anymore because the update
      // comes asynchronously via WebSocket when reconstruction finishes.
      // We just reset the UI state.
      // reloading.value = true 
      // await viewerRef.value?.reloadPointCloud(id)
      console.log("Upload successful, waiting for WebSocket update...")
      uploading.value = false
    }
    
    discardFile()
  } catch (err) {
    console.error(err)
    alert('Error al subir imagen.')
    uploading.value = false
    reloading.value = false
  }
}
</script>

<style scoped>
.page {
  position: fixed;      /* ocupa la ventana, sin scroll de documento */
  inset: 0;             /* top/right/bottom/left = 0 */
  width: 100vw;
  height: 100vh;        /* puedes usar 100dvh si quieres manejar barras móviles */
  background: #0b0d12;
  display: grid;
  place-items: center;
  overflow: hidden;      /* evita scroll dentro del contenedor */
}

/* Mantén tu loading como estaba */
.loading {
  color: #e6e9ef;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

/* Barra inferior sin provocar scroll */
.upload-bar {
  position: fixed;        /* fija respecto al viewport */
  left: 50%;
  bottom: max(20px, env(safe-area-inset-bottom)); /* notch-friendly */
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(255,255,255,0.06);
  padding: 10px 16px;
  border-radius: 12px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.12);
  z-index: 10;
}

.file-info p {
  margin: 0;
  color: #e6e9ef;
  font-size: 14px;
}

.btn {
  color: #e6e9ef;
  background: rgba(124,172,248,0.15);
  border: 1px solid rgba(124,172,248,0.4);
  padding: 6px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}
.btn:hover { background: rgba(124,172,248,0.25); }
.btn.small { font-size: 14px; }
.btn.primary { background: #2563eb; border-color: #2563eb; }
.btn.primary:hover { background: #1e4fba; }
.btn.danger { background: #ef4444; border-color: #ef4444; }
.btn.danger:hover { background: #b91c1c; }

/* Loading overlay */
.loading-overlay {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(11, 13, 18, 0.85);
  backdrop-filter: blur(8px);
  padding: 24px 32px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.15);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  z-index: 100;
}

.loading-overlay p {
  margin: 0;
  color: #e6e9ef;
  font-size: 16px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(124,172,248,0.2);
  border-top-color: #7cacf8;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
