<template>
  <div class="update-feed" v-if="messages.length > 0">
    <transition-group
      name="feed"
      tag="div"
      class="update-feed__list"
    >
      <article
        v-for="msg in messages"
        :key="msg.id"
        class="update-feed__item"
      >
        <div class="update-feed__title">
          Nueva imagen aportada
        </div>
        <div class="update-feed__meta">
          <span class="chip">{{ msg.userId }}</span>
          <span class="sep">·</span>
          <span class="id">ID imagen: {{ msg.imageId }}</span>
        </div>
      </article>
    </transition-group>
  </div>
</template>

<script setup>
import { ref, defineProps, defineExpose } from 'vue'

defineProps({
  projectId: {
    type: String,
    required: false,
  },
})

const messages = ref([])
let messageCounter = 0

function addMessage(userId, imageId) {
  console.log(`[UpdateFeed] Adding message: ${userId} - ${imageId}`)

  const id = ++messageCounter
  const message = {
    id,
    userId,
    imageId,
  }

  messages.value.push(message)

  // Se mantiene 5s en pantalla
  setTimeout(() => {
    removeMessage(id)
  }, 5000)
}

function removeMessage(id) {
  const index = messages.value.findIndex(m => m.id === id)
  if (index !== -1) {
    messages.value.splice(index, 1)
  }
}

defineExpose({
  addMessage,
})
</script>

<style scoped>
.update-feed {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 30;
  pointer-events: none; /* no bloquea el visor */
}

.update-feed__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* Tarjeta de mensaje clara */
.update-feed__item {
  pointer-events: auto;
  min-width: 220px;
  max-width: 320px;

  padding: 10px 12px;
  border-radius: 14px;

  background:
    radial-gradient(140% 160% at 0% 0%, rgba(124, 172, 248, 0.18), transparent 55%),
    radial-gradient(140% 160% at 100% 0%, rgba(155, 140, 242, 0.16), transparent 55%),
    linear-gradient(180deg, #f9fafb, #ffffff);
  border: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);

  color: #0f172a;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  font-size: 13px;

  /* Desvanecerse en 5s */
  animation: feed-fade-out 5s linear forwards;
}

.update-feed__title {
  font-weight: 600;
  margin-bottom: 2px;
}

.update-feed__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  color: #475569;
}

.chip {
  padding: 2px 6px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.15);
  font-size: 11px;
}

.sep {
  opacity: 0.7;
}

.id {
  font-size: 11px;
  opacity: 0.9;
}

/* Animación de desvanecimiento continuo */
@keyframes feed-fade-out {
  0% {
    opacity: 1;
    transform: translateY(0);
  }
  100% {
    opacity: 0;
    transform: translateY(-4px);
  }
}

/* Animaciones del transition-group:
   al entrar / salir y cuando los demás suben */
.feed-enter-from,
.feed-leave-to {
  opacity: 0;
  transform: translateY(4px);
}

.feed-enter-active,
.feed-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}

/* Cuando un ítem se mueve porque otro fue eliminado */
.feed-move {
  transition: transform 0.25s ease;
}

@media (max-width: 600px) {
  .update-feed {
    top: 10px;
    right: 10px;
  }

  .update-feed__item {
    min-width: 180px;
    max-width: 260px;
    padding: 8px 10px;
  }
}
</style>
<!-- 
a -->
