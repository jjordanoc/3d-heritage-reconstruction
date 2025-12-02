<template>
  <div class="page">
    <!-- Update Feed (replaces Viewer Toggle) -->
    <UpdateFeed ref="updateFeedRef" v-if="projectId" :project-id="projectId" />

    <Suspense>
      <template #default>
        <component 
          :is="AsyncPLYViewer"
          ref="viewerRef" 
          @loadComplete="onLoadComplete"
          @model-updated="onModelUpdated"
          @requestUpload="triggerFileInput"
          @noPointCloud="onNoPointCloud"
          @hasPointCloud="onHasPointCloud"
        />
      </template>
      <template #fallback>
        <div class="loading">Cargando visor…</div>
      </template>
    </Suspense>

    <!-- 🔹 Input SIEMPRE presente -->
    <input
      ref="fileInput"
      type="file"
      accept="image/*"
      @change="onFileSelected"
      style="display: none"
    />

    <!-- Overlay inferior: cuando hay nube O cuando hay archivo seleccionado -->
    <div v-if="!isEmptyScene || selectedFile" class="upload-bar">
      <!-- Cuando hay imagen seleccionada -->
      <div v-if="selectedFile" class="upload-bar__content">
        <div class="upload-bar__file">
          <span class="upload-bar__label">Imagen seleccionada</span>
          <span class="upload-bar__name" :title="selectedFile.name">
            {{ selectedFile.name }}
          </span>
        </div>

        <div class="upload-bar__actions">
          <button
            @click="discardFile"
            class="btn btn-ghost small"
            type="button"
          >
            Cancelar
          </button>

          <button
            @click="uploadFile"
            class="btn primary small"
            type="button"
            :disabled="uploading"
          >
            {{ uploading ? "Subiendo…" : "Subir" }}
          </button>
        </div>
      </div>

      <!-- Cuando NO hay imagen seleccionada -->
      <div v-else class="upload-bar__content upload-bar__content--center">
        <button
          @click="triggerFileInput"
          class="btn small"
          type="button"
        >
          Cargar imagen
        </button>
      </div>
    </div>

  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, defineAsyncComponent, computed } from 'vue'
import { useRoute } from 'vue-router'
import UpdateFeed from '@/components/UpdateFeed.vue'

const AsyncPLYViewer = defineAsyncComponent(() => import('@/components/PLYViewer.vue'))
// const AsyncGSplatViewer = defineAsyncComponent(() => import('@/components/GSplatViewer.vue')) // Unused
const viewerRef = ref(null)
const updateFeedRef = ref(null)
const fileInput = ref(null)
const selectedFile = ref(null)
const uploading = ref(false)
// Estado: si no hay nube de puntos, ocultamos la barra inferior
const isEmptyScene = ref(false)
// const currentViewerType = ref('ply') // Defaulting to PLY, no toggle needed

const route = useRoute()
const projectId = computed(() => (route.query.id || route.params.id)?.toString())

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
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function onLoadComplete() {
  uploading.value = false
  // reloading.value = false
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

// Recibimos eventos desde PLYViewer sobre si hay o no nube
function onNoPointCloud() {
  isEmptyScene.value = true
}

function onHasPointCloud() {
  isEmptyScene.value = false
}

async function uploadFile() {
  if (!selectedFile.value) return

  // Al empezar subida: activar spinner
  uploading.value = true

  const id = projectId.value
  if (!id) {
    alert('No se encontró ID del modelo.')
    uploading.value = false
    return
  }

  const form = new FormData()
  form.append('file', selectedFile.value)
  const userId = localStorage.getItem('heritage_user') || 'Guest'
  form.append('user_id', userId)

  try {
    const res = await fetch(`${API_BASE}/pointcloud/${encodeURIComponent(id)}`, {
      method: 'POST',
      body: form,
    })

    if (!res.ok) throw new Error(`Error ${res.status}`)
    
    const data = await res.json()
    
    if (data.success) {
      console.log('Upload successful, waiting for WebSocket update...')
    }

    discardFile()
  } catch (err) {
    console.error(err)
    alert('Error al subir imagen.')
    uploading.value = false
  }
}
</script>

<style scoped>
.page {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;

  /* Tokens claros (alineado con Home.vue) */
  --bg: #f3f4ff;
  --bg-soft: #fbfdff;
  --text: #0f172a;
  --muted: #475569;
  --accent: #2563eb;
  --accent-2: #7c3aed;
  --ring: rgba(37, 99, 235, 0.25);
  --shadow: 0 16px 40px rgba(15, 23, 42, 0.12);

  background:
    radial-gradient(140% 140% at 0% 0%, rgba(124,172,248,0.36), transparent 55%),
    radial-gradient(140% 140% at 100% 0%, rgba(155,140,242,0.32), transparent 55%),
    linear-gradient(180deg, #f9fafb, #ffffff);
  color: var(--text);
  display: grid;
  place-items: center;
  overflow: hidden;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

/* Texto de carga (fallback Suspense) */
.loading {
  color: var(--muted);
  font-size: 14px;
}

/* =========================
   Contenedor de la pill
   (solo posiciona, sin ancho)
   ========================= */
.upload-bar {
  position: fixed;
  left: 50%;
  bottom: max(12px, env(safe-area-inset-bottom));
  transform: translateX(-50%);
  z-index: 10;

  /* que el contenedor NO tenga fondo ni ancho extra */
  background: transparent;
  border: none;
  box-shadow: none;
  padding: 0;
  width: auto;
}

/* Pill real: se ajusta al contenido */
.upload-bar__content {
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;

  max-width: min(420px, 90vw);
  padding: 6px 10px;
  border-radius: 999px;

  background:
    radial-gradient(120% 180% at 0% 0%, rgba(124,172,248,0.14), transparent 55%),
    radial-gradient(120% 180% at 100% 0%, rgba(155,140,242,0.14), transparent 55%),
    linear-gradient(180deg, #f9fafb, #ffffff);
  border: 1px solid rgba(148,163,184,0.35);
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px) saturate(150%);
}

.upload-bar__content--center {
  justify-content: center;
}

/* Info de archivo */
.upload-bar__file {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  min-width: 0;
}

.upload-bar__label {
  font-size: 11px;
  color: var(--muted);
  opacity: 0.9;
}

.upload-bar__name {
  font-size: 12px;
  color: var(--text);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 140px; /* evita que agrande demasiado la pill */
}

/* Acciones: botones a la derecha, pegaditos */
.upload-bar__actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

/* =========================
   Botones claros
   ========================= */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;

  padding: 5px 10px;
  border-radius: 10px;

  border: 1px solid rgba(148,163,184,0.55);
  background:
    linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.9));
  color: var(--text);

  font-size: 12px;
  line-height: 1.2;
  cursor: pointer;
  outline: none;

  transition:
    transform 0.14s ease,
    box-shadow 0.14s ease,
    border-color 0.14s ease,
    background 0.14s ease;
}

.btn.small {
  padding-inline: 9px;
}

.btn:hover,
.btn:focus-visible {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px var(--ring);
  transform: translateY(-1px);
}

/* Primario */
.btn.primary {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border-color: transparent;
  color: #f9fafb;
}

.btn.primary:disabled {
  opacity: 0.7;
  cursor: default;
  transform: none;
  box-shadow: none;
}

.btn.primary:hover:not(:disabled),
.btn.primary:focus-visible:not(:disabled) {
  box-shadow: 0 0 0 3px var(--ring);
}

/* Ghost: para Cancelar */
.btn.btn-ghost {
  background: transparent;
  border-color: transparent;
  color: var(--muted);
}

.btn.btn-ghost:hover,
.btn.btn-ghost:focus-visible {
  background: rgba(148,163,184,0.12);
  border-color: rgba(148,163,184,0.3);
  box-shadow: 0 0 0 3px rgba(148,163,184,0.25);
}

/* =========================
   Mobile: aún más compacto
   ========================= */
@media (max-width: 600px) {
  .upload-bar__content {
    padding: 4px 8px;
    gap: 6px;
  }

  .upload-bar__file {
    flex-direction: row;
    align-items: center;
    gap: 4px;
  }

  .upload-bar__label {
    font-size: 10px;
  }

  .upload-bar__name {
    font-size: 11px;
    max-width: 110px;
  }

  .upload-bar__actions {
    gap: 4px;
  }

  .btn {
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 9px;
  }
}
</style>
