<template>
  <div class="modal-overlay">
    <div class="modal-content">
      <h2>¡Bienvenido!</h2>
      <p>Ingresa un nombre de usuario para continuar.</p>
      
      <input 
        v-model="username" 
        type="text" 
        placeholder="Nombre de usuario" 
        class="username-input"
        @keyup.enter="saveUser"
      />
      
      <div class="actions">
        <button
          @click="saveUser"
          class="btn primary"
          :disabled="!username.trim()"
        >
          Continuar
        </button>
        <button @click="enterAsGuest" class="btn secondary">
          Entrar como invitado
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['close'])
const username = ref('')

function saveUser() {
  if (!username.value.trim()) return
  localStorage.setItem('heritage_user', username.value.trim())
  emit('close')
}

function enterAsGuest() {
  // Mantengo "Guest" en localStorage para no romper nada en el resto del código
  localStorage.setItem('heritage_user', 'Guest')
  emit('close')
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.75);
  backdrop-filter: blur(8px);
  display: grid;
  place-items: center;
  z-index: 9999;
}

.modal-content {
  width: min(400px, 90vw);
  padding: 24px 22px 22px;
  border-radius: 18px;

  background:
    radial-gradient(
      140% 160% at 0% 0%,
      rgba(124, 172, 248, 0.22),
      transparent 55%
    ),
    radial-gradient(
      140% 160% at 100% 0%,
      rgba(155, 140, 242, 0.22),
      transparent 55%
    ),
    linear-gradient(180deg, #f9fafb, #ffffff);
  border: 1px solid rgba(148, 163, 184, 0.28);
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
  backdrop-filter: blur(10px) saturate(150%);

  display: flex;
  flex-direction: column;
  gap: 18px;

  color: #0f172a;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

h2 {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.01em;
}

p {
  margin: 0;
  color: #475569;
  font-size: 14px;
}

.username-input {
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.7);
  background: #f1f5f9;
  color: #0f172a;
  font-size: 14px;
  outline: none;
  transition:
    border-color 0.18s ease,
    box-shadow 0.18s ease,
    background 0.18s ease;
}

.username-input::placeholder {
  color: #94a3b8;
}

.username-input:focus {
  border-color: #2563eb;
  background: #ffffff;
  box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.4);
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 4px;
}

.btn {
  padding: 10px 12px;
  border-radius: 999px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
  transition:
    transform 0.15s ease,
    box-shadow 0.15s ease,
    filter 0.15s ease,
    background 0.15s ease,
    border-color 0.15s ease;
}

/* Primario: mismo gradiente azul-violeta del app */
.btn.primary {
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  color: #f9fafb;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.45);
}

.btn.primary:hover:not(:disabled),
.btn.primary:focus-visible:not(:disabled) {
  transform: translateY(-0.5px);
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.55);
  filter: brightness(1.03);
}

.btn.primary:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}

/* Secundario: borde suave, tono gris-azulado */
.btn.secondary {
  background: rgba(248, 250, 252, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.8);
  color: #475569;
}

.btn.secondary:hover,
.btn.secondary:focus-visible {
  background: #e5edf7;
  border-color: rgba(148, 163, 184, 1);
  transform: translateY(-0.5px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.18);
}

/* Mobile tweaks */
@media (max-width: 640px) {
  .modal-content {
    padding: 22px 18px 18px;
    border-radius: 16px;
  }

  h2 {
    font-size: 20px;
  }

  p {
    font-size: 13px;
  }

  .username-input {
    font-size: 13px;
    padding: 9px 10px;
  }

  .btn {
    font-size: 12px;
    padding: 9px 10px;
  }
}
</style>
