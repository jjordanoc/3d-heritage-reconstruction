<template>
  <div class="modal-overlay">
    <div class="modal-content">
      <h2>Welcome!</h2>
      <p>Please enter a username to continue.</p>
      
      <input 
        v-model="username" 
        type="text" 
        placeholder="Username" 
        class="username-input"
        @keyup.enter="saveUser"
      />
      
      <div class="actions">
        <button @click="saveUser" class="btn primary" :disabled="!username.trim()">
          Enter
        </button>
        <button @click="enterAsGuest" class="btn secondary">
          Enter as Guest
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
  localStorage.setItem('heritage_user', 'Guest')
  emit('close')
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  display: grid;
  place-items: center;
  z-index: 9999;
}

.modal-content {
  background: #1e293b;
  padding: 32px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  width: min(400px, 90vw);
  display: flex;
  flex-direction: column;
  gap: 20px;
  color: #e2e8f0;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
}

h2 {
  margin: 0;
  font-size: 24px;
  color: white;
}

p {
  margin: 0;
  color: #94a3b8;
}

.username-input {
  padding: 12px;
  border-radius: 8px;
  border: 1px solid #334155;
  background: #0f172a;
  color: white;
  font-size: 16px;
  outline: none;
  transition: border-color 0.2s;
}

.username-input:focus {
  border-color: #3b82f6;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.btn {
  padding: 12px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.2s;
}

.btn.primary {
  background: #3b82f6;
  color: white;
}

.btn.primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.secondary {
  background: transparent;
  border: 1px solid #334155;
  color: #94a3b8;
}

.btn.secondary:hover {
  border-color: #475569;
  color: #cbd5e1;
}
</style>

