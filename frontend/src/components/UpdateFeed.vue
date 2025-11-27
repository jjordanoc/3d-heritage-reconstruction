<template>
  <div class="update-feed-container" v-if="messages.length > 0">
    <transition-group name="feed-anim" tag="div" class="feed-list">
      <div v-for="msg in messages" :key="msg.id" class="feed-item">
        <span class="feed-text">{{ msg.userId }} contributed {{ msg.imageId }}!</span>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { ref, watch, onUnmounted } from 'vue'
import { useWebSocket } from '@vueuse/core'

const props = defineProps({
  projectId: {
    type: String,
    required: true
  }
})

const messages = ref([])
let messageCounter = 0

// --- WebSocket Setup ---
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')
const WS_API_BASE = (import.meta.env.VITE_WS_API_URL || '').replace(/\/+$/, '')

let wsUrl = WS_API_BASE.replace(/^http/, 'ws')
if (!wsUrl || wsUrl.startsWith('/')) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  wsUrl = `${protocol}//${host}${API_BASE}`
}

const fullWsUrl = `${wsUrl}/ws/${props.projectId}`

const { data, close } = useWebSocket(fullWsUrl, {
  autoReconnect: {
    retries: -1,
    delay: 1000,
  },
  heartbeat: {
    message: JSON.stringify({ type: 'ping' }),
    interval: 30000,
    pongTimeout: 10000,
  },
})

watch(data, (newData) => {
  if (!newData) return
  try {
    const msg = JSON.parse(newData)
    // Look for metadata with user_id and image_id on any event
    if (msg.metadata && msg.metadata.user_id && msg.metadata.image_id) {
      addMessage(msg.metadata.user_id, msg.metadata.image_id)
    }
  } catch (e) {
    console.error('[UpdateFeed] Error parsing message:', e)
  }
})

function addMessage(userId, imageId) {
  const id = ++messageCounter
  const message = {
    id,
    userId,
    imageId,
    timestamp: Date.now()
  }
  
  // Stack order: Oldest at top, newest at bottom (push to end)
  messages.value.push(message)
  
  // Remove after 10 seconds
  setTimeout(() => {
    removeMessage(id)
  }, 10000)
}

function removeMessage(id) {
  const index = messages.value.findIndex(m => m.id === id)
  if (index !== -1) {
    messages.value.splice(index, 1)
  }
}

onUnmounted(() => {
  close()
})
</script>

<style scoped>
.update-feed-container {
  position: fixed;
  top: 20px; /* Replaces top: 20px from viewer-toggle */
  right: 20px;
  width: 50vw; /* Half of width */
  max-height: 50vh; /* Max half of height */
  background-color: rgba(0, 0, 0, 0.5); /* Black transparent */
  z-index: 10; /* Same z-index as previous toggle */
  border-radius: 12px;
  padding: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  pointer-events: none; /* Allow clicks to pass through */
  transition: height 0.3s ease;
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.1);
}

.feed-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.feed-item {
  width: 100%;
}

.feed-text {
  color: white;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  font-size: 14px;
  font-weight: 500;
  text-shadow: 0 1px 2px rgba(0,0,0,0.8);
  display: block;
  word-break: break-word;
}

/* Transitions */
.feed-anim-enter-active,
.feed-anim-leave-active {
  transition: all 0.5s ease;
}

.feed-anim-enter-from {
  opacity: 0;
  transform: translateY(20px); /* Enter from bottom */
}

.feed-anim-leave-to {
  opacity: 0;
  transform: translateY(-20px); /* Fade away to top */
}

.feed-anim-move {
  transition: transform 0.5s ease;
}
</style>
