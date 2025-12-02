<!-- src/components/SceneModal.vue -->
<template>
  <div class="modal-backdrop" @click.self="handleClose">
    <div class="modal-sheet">
      <header class="modal-header">
        <h2 class="modal-title">Nueva Escena</h2>
        <button
          type="button"
          class="icon-btn"
          @click="handleClose"
          aria-label="Cerrar"
        >
          ✕
        </button>
      </header>

      <form class="modal-body" @submit.prevent="handleSubmit">
        <!-- Nombre de la escena -->
        <label class="field">
          <span class="field-label">Nombre de la escena</span>
          <input
            v-model="sceneName"
            class="field-input"
            type="text"
            required
            placeholder="ej: mi-proyecto"
          />
          <span class="field-hint">
            Se usará como identificador único.
          </span>
        </label>

        <!-- Imagen miniatura (OPCIONAL) -->
        <div class="field">
          <div class="field-header">
            <span class="field-label">Imagen miniatura (obligatoria)</span>
            <button
              type="button"
              class="dropzone"
              :class="{ 'dropzone--filled': !!previewUrl }"
              @click="triggerFileInput"
            >
              Seleccionar imagen
            </button>
          </div>

          <!-- input nativo escondido -->
          <input
            ref="fileInput"
            type="file"
            class="file-input-native"
            accept="image/*"
            :capture="captureAttr"
            @change="onFileChange"
          />

          <!-- Dropzone solo si el usuario activó la miniatura -->
          <div
            v-if="showThumbnail"
            class="dropzone"
            :class="{ 'dropzone--filled': !!previewUrl }"
            @click="triggerFileInput"
          >
            <div class="dropzone-main">
              <div v-if="previewUrl" class="dropzone-preview">
                <img :src="previewUrl" alt="Previsualización miniatura" />
              </div>

              <div class="dropzone-text">
                <span class="dropzone-title">
                  {{ previewUrl ? 'Cambiar imagen' : 'Haz clic para seleccionar una imagen' }}
                </span>
                <span class="dropzone-subtitle">
                  Se reescalará automáticamente a 256 × 256 píxeles
                </span>
              </div>
            </div>

            <!-- Tacho a la derecha -->
            <button
              v-if="previewUrl"
              type="button"
              class="preview-clear-btn"
              @click.stop="clearImage"
              aria-label="Quitar imagen seleccionada"
            >
              <svg
                class="trash-icon"
                viewBox="0 0 20 20"
                aria-hidden="true"
              >
                <path
                  d="M7.5 4.5h5a.75.75 0 0 1 .74.62l.26 1.63h2.25a.75.75 0 0 1 0 1.5h-.63l-.56 7.03A1.75 1.75 0 0 1 12.82 17H7.18a1.75 1.75 0 0 1-1.74-1.72L4.88 8.25h-.63a.75.75 0 0 1 0-1.5H6.5l.26-1.63a.75.75 0 0 1 .74-.62Zm.27 3.75a.75.75 0 0 0-1.5.06l.25 5a.75.75 0 1 0 1.5-.06l-.25-5Zm2.73.06a.75.75 0 0 0-1.5 0v5a.75.75 0 0 0 1.5 0v-5Zm2.48-.06a.75.75 0 0 0-1.5-.06l-.25 5a.75.75 0 0 0 1.5.06l.25-5ZM8.25 3a1 1 0 0 1 1-1h1.5a1 1 0 0 1 1 1v.5h-3.5V3Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
        </div>

        <!-- Error -->
        <p v-if="errorMessage" class="error-msg">
          {{ errorMessage }}
        </p>

        <!-- Footer -->
        <footer class="modal-footer">
          <button
            type="button"
            class="btn btn-ghost"
            @click="handleClose"
            :disabled="isSubmitting"
          >
            Cancelar
          </button>
          <button
            type="submit"
            class="btn"
            :disabled="isSubmitting || !sceneName || !file"
          >
            <span v-if="!isSubmitting">Crear</span>
            <span v-else>Creando…</span>
          </button>
        </footer>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const isAndroid = typeof navigator !== 'undefined' && /Android/i.test(navigator.userAgent || '')
const captureAttr = isAndroid ? 'environment' : undefined

const emit = defineEmits(['close', 'created'])

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

const sceneName = ref('')
const file = ref(null)
const previewUrl = ref('')
const isSubmitting = ref(false)
const errorMessage = ref('')

const fileInput = ref(null)
const showThumbnail = ref(false)

async function resizeImageTo256(inputFile) {
  const SIZE = 512

  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (ev) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = SIZE
        canvas.height = SIZE
        const ctx = canvas.getContext('2d')

        if (!ctx) {
          reject(new Error('No se pudo obtener el contexto 2D del canvas'))
          return
        }

        const scale = Math.max(SIZE / img.width, SIZE / img.height)
        const newWidth = img.width * scale
        const newHeight = img.height * scale

        const dx = (SIZE - newWidth) / 2
        const dy = (SIZE - newHeight) / 2

        ctx.fillStyle = '#f3f4f6'
        ctx.fillRect(0, 0, SIZE, SIZE)

        ctx.drawImage(img, dx, dy, newWidth, newHeight)

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error('No se pudo generar el blob de la imagen'))
              return
            }
            resolve(blob)
          },
          'image/png',
          0.95
        )
      }

      img.onerror = (err) => reject(err)
      img.src = ev.target?.result
    }

    reader.onerror = (err) => reject(err)
    reader.readAsDataURL(inputFile)
  })
}

function handleClose() {
  if (isSubmitting.value) return
  emit('close')
}

function triggerFileInput() {
  fileInput.value?.click()
}

function clearImage() {
  file.value = null
  if (previewUrl.value) {
    window.URL.revokeObjectURL(previewUrl.value)
  }
  previewUrl.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
  showThumbnail.value = false
}

async function onFileChange(event) {
  const selected = event.target.files?.[0]
  if (!selected) return

  try {
    const resizedBlob = await resizeImageTo256(selected)

    // guardamos el blob 256x256
    file.value = resizedBlob

    // preview de la versión reescalada
    if (previewUrl.value) {
      window.URL.revokeObjectURL(previewUrl.value)
    }
    previewUrl.value = window.URL.createObjectURL(resizedBlob)

    // mostramos el bloque de preview si estaba oculto
    showThumbnail.value = true
  } catch (err) {
    console.error('Error al procesar la imagen:', err)
    errorMessage.value = 'No se pudo procesar la imagen seleccionada.'
    file.value = null
  }
}

async function handleSubmit() {
  if (!sceneName.value) return

  if (!file.value) {
    errorMessage.value = 'Debes seleccionar una imagen miniatura antes de crear la escena.'
    return
  }

  isSubmitting.value = true
  errorMessage.value = ''

  try {
    const formData = new FormData()
    formData.append('name', sceneName.value)

    const filename = `${sceneName.value || 'thumbnail'}.png`
    formData.append('thumbnail', file.value, filename)

    const res = await fetch(
      `${API_BASE}/scene/${encodeURIComponent(sceneName.value)}`,
      {
        method: 'POST',
        body: formData,
      }
    )

    if (!res.ok) {
      throw new Error(`Error HTTP ${res.status}`)
    }

    emit('created', sceneName.value)
    emit('close')
  } catch (err) {
    console.error('Error creando escena:', err)
    errorMessage.value = 'No se pudo crear la escena. Inténtalo nuevamente.'
  } finally {
    isSubmitting.value = false
  }
}
</script>

<style scoped>
:root {
  --accent: #7cacf8;
  --accent-2: #9b8cf2;
  --muted: #6b7280;
  --border-soft: rgba(148, 163, 184, 0.5);
}

/* BACKDROP: centrado SIEMPRE, sin scroll interno */
.modal-backdrop {
  position: fixed;
  inset: 0;
  min-height: 100vh;
  background: rgba(15, 23, 42, 0.25);
  backdrop-filter: blur(16px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  z-index: 50;
}

/* Tarjeta: altura limitada a ~560px -> no necesita scroll en pantallas normales */
.modal-sheet {
  width: min(640px, 100% - 32px);
  max-height: 560px; /* contenido real cabe de sobra */
  border-radius: 22px;
  background:
    radial-gradient(120% 140% at 0% 0%, rgba(124, 172, 248, 0.35), transparent 55%),
    radial-gradient(110% 140% at 100% 0%, rgba(155, 140, 242, 0.3), transparent 55%),
    linear-gradient(180deg, #f9fafb, #ffffff);
  border: 1px solid var(--border-soft);
  box-shadow: 0 26px 80px rgba(15, 23, 42, 0.30);
  color: #020617;
  padding: 22px 26px 18px;
}

/* Un poco más compacto en pantallas pequeñas */
@media (max-width: 640px) {
  .modal-sheet {
    width: min(520px, 100% - 24px);
    padding: 18px 18px 14px;
    border-radius: 18px;
  }
}

/* HEADER */
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.modal-title {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.01em;
}

/* Botón cerrar */
.icon-btn {
  border: none;
  background: transparent;
  color: #9ca3af;
  border-radius: 999px;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  font-size: 15px;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
}

.icon-btn:hover {
  background: rgba(148, 163, 184, 0.2);
  color: #111827;
  transform: translateY(-1px);
}

/* BODY – SIN max-height, SIN overflow -> no scroll interno */
.modal-body {
  display: grid;
  gap: 14px;
}

/* FIELDS */
.field {
  display: grid;
  gap: 6px;
}

.field-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.field-label {
  font-size: 14px;
  font-weight: 500;
}

.field-input {
  width: 90%;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: #f9fafb;
  padding: 9px 14px;
  color: inherit;
  font-size: 14px;
  outline: none;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, transform 0.18s ease;
}

.field-input::placeholder {
  color: #9ca3af;
}

.field-input:focus-visible {
  border-color: rgba(37, 99, 235, 0.75);
  box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.55);
  background: #ffffff;
  transform: translateY(-1px);
}

.field-hint {
  font-size: 11px;
  color: var(--muted);
}

/* LINK OPCIONAL */
.link-btn {
  border: none;
  background: none;
  padding: 0;
  margin: 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--accent);
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
}

/* FILE INPUT NATIVO OCULTO */
.file-input-native {
  display: none;
}

/* Dropzone */
.dropzone {
  border-radius: 16px;
  border: 1px dashed rgba(148, 163, 184, 0.9);
  background: #f3f4f6;
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition:
    border-color 0.18s ease,
    background 0.18s ease,
    box-shadow 0.18s ease,
    transform 0.18s ease;
}

.dropzone-main {
  display: flex;
  align-items: center;
  gap: 10px;
}

.dropzone--filled {
  border-style: solid;
  background: #eef2ff;
}

.dropzone:hover {
  border-color: rgba(124, 172, 248, 0.9);
  background: #e5f3ff;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
  transform: translateY(-1px);
}

/* Preview */
.dropzone-preview {
  width: 64px;
  height: 64px;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.7);
  flex-shrink: 0;
  background: #e5e7eb;
}

.dropzone-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* Botón tacho a la derecha del dropzone */
.preview-clear-btn {
  margin-left: auto;
  width: 30px;
  height: 30px;
  border-radius: 999px;
  border: none;
  background: #0f172a;
  color: #f9fafb;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 10px rgba(15, 23, 42, 0.5);
  transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}

.preview-clear-btn:hover {
  transform: translateY(-1px);
  background: #dc2626;
  box-shadow: 0 6px 16px rgba(127, 29, 29, 0.55);
}

.trash-icon {
  width: 15px;
  height: 15px;
}

.dropzone-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.dropzone-title {
  font-size: 13px;
  font-weight: 500;
}

.dropzone-subtitle {
  font-size: 11px;
  color: var(--muted);
}

/* ERROR */
.error-msg {
  margin: 2px 0 0;
  font-size: 12px;
  color: #b91c1c;
}

/* FOOTER */
.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 6px;
}

/* BOTONES */
.btn {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.7);
  padding: 8px 18px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
  color: #111827;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  outline: none;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease, opacity 0.15s ease;
}

.btn:hover:not(:disabled),
.btn:focus-visible:not(:disabled) {
  border-color: rgba(37, 99, 235, 0.75);
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.28);
  transform: translateY(-1px);
}

.btn:disabled {
  opacity: 0.55;
  cursor: default;
  box-shadow: none;
}

/* Botón fantasma */
.btn.btn-ghost {
  background: #e5e7eb;
}

.btn.btn-ghost:hover:not(:disabled),
.btn.btn-ghost:focus-visible:not(:disabled) {
  background: #d1d5db;
}

/* Motion */
@media (prefers-reduced-motion: reduce) {
  .modal-sheet,
  .btn,
  .icon-btn,
  .dropzone,
  .preview-clear-btn {
    transition: none;
  }
}
</style>
