<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-card">
      <h2>Nueva Escena</h2>
      
      <form @submit.prevent="handleSubmit">
        <div class="form-group">
          <label for="sceneName">Nombre de la escena</label>
          <input
            id="sceneName"
            v-model="sceneName"
            type="text"
            placeholder="ej: mi-proyecto"
            required
            :disabled="isSubmitting"
          />
          <p class="hint">Se usará como identificador único</p>
        </div>

        <div class="form-group">
          <label for="thumbnail">Imagen miniatura</label>
          <input
            id="thumbnail"
            ref="fileInputRef"
            type="file"
            accept="image/*"
            required
            :disabled="isSubmitting"
            @change="onFileChange"
          />
          <p v-if="selectedFileName" class="file-name">{{ selectedFileName }}</p>
        </div>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

        <div class="button-group">
          <button
            type="button"
            class="btn secondary"
            :disabled="isSubmitting"
            @click="$emit('close')"
          >
            Cancelar
          </button>
          <button
            type="submit"
            class="btn primary"
            :disabled="isSubmitting || !sceneName || !thumbnailFile"
          >
            {{ isSubmitting ? 'Creando...' : 'Crear' }}
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const emit = defineEmits(['close', 'created'])
const router = useRouter()

const sceneName = ref('')
const thumbnailFile = ref(null)
const selectedFileName = ref('')
const isSubmitting = ref(false)
const errorMessage = ref('')
const fileInputRef = ref(null)

// API base
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

function onFileChange(e) {
  const file = e.target.files?.[0]
  if (file) {
    thumbnailFile.value = file
    selectedFileName.value = file.name
  } else {
    thumbnailFile.value = null
    selectedFileName.value = ''
  }
}

async function handleSubmit() {
  if (!sceneName.value || !thumbnailFile.value) return
  
  errorMessage.value = ''
  isSubmitting.value = true

  try {
    const formData = new FormData()
    formData.append('thumbnail', thumbnailFile.value)

    const res = await fetch(`${API_BASE}/scene/${encodeURIComponent(sceneName.value)}`, {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Error desconocido' }))
      throw new Error(error.detail || `Error ${res.status}`)
    }

    // Success - emit event and navigate
    emit('created', sceneName.value)
    emit('close')
    
    // Navigate to viewer for the new scene
    router.push({ name: 'viewer', query: { id: sceneName.value } })
  } catch (err) {
    console.error('Error creating scene:', err)
    errorMessage.value = err.message || 'Error al crear la escena. Intenta de nuevo.'
  } finally {
    isSubmitting.value = false
  }
}
</script>

<style scoped>
/* Theme tokens matching Home.vue */
:root {
  --bg: #0b0d12;
  --bg-soft: #0f131a;
  --card: rgba(255,255,255,0.06);
  --card-stroke: rgba(255,255,255,0.12);
  --text: #e6e9ef;
  --muted: #9aa4b2;
  --accent: #7cacf8;
  --ring: rgba(124,172,248,0.45);
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #ffffff;
    --bg-soft: #fbfdff;
    --card: rgba(16,24,40,0.04);
    --card-stroke: rgba(16,24,40,0.08);
    --text: #0f172a;
    --muted: #475569;
    --accent: #2563eb;
    --ring: rgba(37,99,235,.25);
  }
}

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  display: grid;
  place-items: center;
  z-index: 1000;
  padding: 20px;
}

.modal-card {
  background: var(--card);
  border: 1px solid var(--card-stroke);
  border-radius: 16px;
  padding: 24px;
  width: min(500px, 100%);
  backdrop-filter: saturate(140%) blur(10px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
}

.modal-card h2 {
  margin: 0 0 20px;
  color: var(--text);
  font-size: 24px;
  letter-spacing: -0.01em;
}

form {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

label {
  color: var(--text);
  font-size: 14px;
  font-weight: 500;
}

input[type="text"],
input[type="file"] {
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--card-stroke);
  border-radius: 8px;
  color: var(--text);
  font-size: 14px;
  font-family: inherit;
  transition: border-color 0.2s, background 0.2s;
}

input[type="text"]:focus {
  outline: none;
  border-color: var(--accent);
  background: rgba(255, 255, 255, 0.08);
}

input[type="text"]:disabled,
input[type="file"]:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

input[type="file"] {
  cursor: pointer;
}

.hint,
.file-name {
  margin: 0;
  font-size: 12px;
  color: var(--muted);
}

.file-name {
  font-weight: 500;
}

.error {
  margin: 0;
  padding: 10px;
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.4);
  border-radius: 8px;
  color: #ef4444;
  font-size: 14px;
}

.button-group {
  display: flex;
  gap: 10px;
  margin-top: 6px;
}

.btn {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid var(--card-stroke);
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.secondary {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text);
}

.btn.secondary:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.1);
}

.btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}

.btn.primary:hover:not(:disabled) {
  background: #6399e5;
  transform: translateY(-1px);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

@media (prefers-reduced-motion: reduce) {
  .btn,
  input { transition: none; }
}
</style>

