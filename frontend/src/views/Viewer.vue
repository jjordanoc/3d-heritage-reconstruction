<template>
  <div class="page">
    <!-- Update Feed -->
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
          :first-scene-processing="firstSceneProcessing"
        />
      </template>
      <template #fallback>
        <div class="loading">Cargando visor…</div>
      </template>
    </Suspense>

    <!-- Input SIEMPRE presente -->
    <input
      ref="fileInput"
      type="file"
      accept="image/*"
      :capture="captureAttr"
      @change="onFileSelected"
      style="display: none"
    />

    <!-- Barra inferior:
         - hay nube (!isEmptyScene)
         - o hay archivo seleccionado
         - o se está procesando la primera imagen -->
    <div
      v-if="!isEmptyScene || selectedFile || firstSceneProcessing"
      class="upload-bar"
    >
      <!-- Modo: archivo seleccionado -->
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

      <!-- Modo: sin archivo seleccionado -->
      <div
        v-else
        class="upload-bar__content upload-bar__content--compact"
      >
        <!-- Estado: procesando PRIMERA imagen -->
        <template v-if="firstSceneProcessing">
          <div class="upload-bar__processing">
            <div class="upload-bar__spinner"></div>
            <span class="upload-bar__processing-title">
              Procesando la primera imagen…
            </span>
          </div>

          <button
            @click="triggerFileInput"
            class="btn small secondary"
            type="button"
          >
            Cargar imagen
          </button>
        </template>

        <!-- Estado normal: solo botón -->
        <template v-else>
          <button
            @click="triggerFileInput"
            class="btn small"
            type="button"
          >
            Cargar imagen
          </button>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, defineAsyncComponent, computed } from 'vue'
import { useRoute } from 'vue-router'
import UpdateFeed from '@/components/UpdateFeed.vue'

const AsyncPLYViewer = defineAsyncComponent(() => import('@/components/PLYViewer.vue'))

const viewerRef = ref(null)
const updateFeedRef = ref(null)
const fileInput = ref(null)
const selectedFile = ref(null)
const uploading = ref(false)

const isEmptyScene = ref(false)
const firstSceneProcessing = ref(false)
const hasPointCloudEver = ref(false)

const route = useRoute()
const projectId = computed(() => (route.query.id || route.params.id)?.toString())

const isAndroid = /Android/i.test(navigator.userAgent || '')
const captureAttr = isAndroid ? 'environment' : undefined

onMounted(() => {
  document.body.classList.add('no-scroll')
})
onUnmounted(() => {
  document.body.classList.remove('no-scroll')
})

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
}

function onModelUpdated(metadata) {
  if (updateFeedRef.value && metadata) {
    const updates = Array.isArray(metadata) ? metadata : [metadata]
    updates.forEach(meta => {
      const userId = meta.user_id || 'System'
      const imageId = meta.image_id || 'Update'
      updateFeedRef.value.addMessage(userId, imageId)
    })
  }
}

function onNoPointCloud() {
  isEmptyScene.value = true
  // firstSceneProcessing se controla solo desde el flujo de subida
}

function onHasPointCloud() {
  isEmptyScene.value = false
  firstSceneProcessing.value = false
  hasPointCloudEver.value = true
}

async function uploadFile() {
  if (!selectedFile.value || uploading.value) return

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

    // Si es la primera vez que esta escena tendrá nube,
    // activamos el mensaje de "procesando primera imagen"
    if (data.success && !hasPointCloudEver.value) {
      firstSceneProcessing.value = true
      isEmptyScene.value = true
    }

    if (data.success) {
      console.log('Upload successful, waiting for WebSocket update...')
    }

    discardFile()
  } catch (err) {
    console.error(err)
    alert('Error al subir imagen.')
  } finally {
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

.loading {
  color: var(--muted);
  font-size: 14px;
}

/* Contenedor de la pill inferior */
.upload-bar {
  position: fixed;
  left: 50%;
  bottom: max(10px, env(safe-area-inset-bottom));
  transform: translateX(-50%);
  z-index: 10;
  background: transparent;
  pointer-events: none; /* el contenido interior sí tiene pointer-events */
}

/* Pill real */
.upload-bar__content {
  pointer-events: auto;
  display: inline-flex;
  align-items: center;
  gap: 8px;

  max-width: min(360px, 92vw);
  padding: 6px 10px;
  border-radius: 999px;

  background: rgba(15, 23, 42, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.65);
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.45);
  color: #e5e7eb;
  backdrop-filter: blur(10px);
  font-size: 12px;
}

/* versión más compacta cuando no mostramos info de archivo */
.upload-bar__content--compact {
  padding: 5px 10px;
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
  font-size: 10px;
  color: #94a3b8;
}

.upload-bar__name {
  font-size: 11px;
  color: #e5e7eb;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 150px;
}

.upload-bar__actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

/* Botones */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.7);
  background: rgba(15, 23, 42, 0.9);
  color: #e5e7eb;
  font-size: 11px;
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
  box-shadow: 0 0 0 2px var(--ring);
  transform: translateY(-0.5px);
}

/* Primario */
.btn.primary {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border-color: transparent;
  color: #f9fafb;
}

/* Secundario (borde claro, fondo oscuro) */
.btn.secondary {
  background: transparent;
  border-color: rgba(148, 163, 184, 0.7);
}

/* Ghost */
.btn.btn-ghost {
  background: transparent;
  border-color: transparent;
  color: #cbd5f5;
}

.btn.btn-ghost:hover,
.btn.btn-ghost:focus-visible {
  background: rgba(148, 163, 184, 0.18);
  border-color: rgba(148, 163, 184, 0.4);
}

/* Estado "procesando primera imagen" */
.upload-bar__processing {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-right: 4px;
}

.upload-bar__spinner {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 2px solid rgba(148, 163, 184, 0.45);
  border-top-color: #38bdf8;
  border-right-color: #a855f7;
  animation: upload-spin 0.8s linear infinite;
}

.upload-bar__processing-title {
  font-size: 11px;
  white-space: nowrap;
}

@keyframes upload-spin {
  to {
    transform: rotate(360deg);
  }
}

/* Mobile */
@media (max-width: 600px) {
  .upload-bar__content {
    max-width: 94vw;
    padding-inline: 8px;
  }

  .upload-bar__name {
    max-width: 120px;
  }

  .btn {
    font-size: 10px;
    padding-inline: 7px;
  }
}
</style>
