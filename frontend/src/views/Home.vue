<!-- src/views/Home.vue -->
<template>
  <div class="home">
    <NavBar />

    <!-- HERO -->
    <section class="hero">
      <div class="hero__bg"></div>
      <div class="hero__content">
        <h1>Reconstrucciones <span>3D</span></h1>
        <p class="lead">
          Explora colecciones de reconstrucciones 3D con nubes de puntos.
        </p>
      </div>
    </section>

    <!-- CATEGORIES -->
    <section class="categories container">
      <!-- Sección: Crear nueva escena -->
      <div class="category">
        <div class="category__head">
          <h2>Crear Nueva Escena Compartida</h2>
        </div>

        <div class="grid">
          <!-- Add Scene Card (SOLO aquí) -->
          <article
            class="card add-scene-card"
            tabindex="0"
            @click="openModal"
            @keydown.enter="openModal"
            @keydown.space.prevent="openModal"
          >
            <div class="thumb add-thumb">
              <svg class="plus-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M12 5v14M5 12h14"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </div>

            <div class="card__body">
              <h3 class="card__title">Crear Nueva Escena Compartida</h3>
              <p class="card__sub">Sube imágenes para reconstruir</p>
            </div>
          </article>
        </div>
      </div>

      <!-- Sección: Escenas compartidas (3 estados en un solo bloque) -->
      <div class="category">
        <div class="category__head">
          <h2>Escenas Compartidas</h2>
        </div>

        <!-- Estado: cargando -->
        <div v-if="isLoadingScenes" class="status-wrapper">
          <article class="card card--status">
            <div class="status-icon status-icon--spinner"></div>
            <div class="status-copy">
              <p class="status-title">Buscando escenas compartidas…</p>
              <p class="status-sub">
                Cargando escenas compartidas. Esto puede tardar unos segundos.
              </p>
            </div>
          </article>
        </div>

        <!-- Estado: con escenas -->
        <template v-else-if="userScenes.length > 0">
          <div class="grid">
            <!-- User Scene Cards -->
            <article
              v-for="scene in userScenes"
              :key="scene.id"
              class="card"
              tabindex="0"
            >
              <div class="thumb">
                <img
                  :src="scene.img"
                  :alt="scene.title"
                  loading="lazy"
                  decoding="async"
                  @error="onImgError($event)"
                />
              </div>

              <div class="card__body">
                <h3 class="card__title">{{ scene.title }}</h3>
                <p class="card__sub">{{ scene.subtitle }}</p>

                <router-link
                  class="btn"
                  :to="{ name: 'viewer', query: { id: scene.id } }"
                  :aria-label="`Ver ${scene.title} en el visor`"
                >
                  <svg class="btn__icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      d="M5 12h12M13 5l7 7-7 7"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                  View
                </router-link>
              </div>
            </article>
          </div>
        </template>

        <!-- Estado: sin escenas -->
        <div v-else class="status-wrapper">
          <article class="card card--status card--empty">
            <div class="status-icon status-icon--ghost">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M4 18.5V10a8 8 0 0 1 16 0v8.5l-2.2-1.4a2 2 0 0 0-2.2.1L14 19l-1.6-1.2a2 2 0 0 0-2.4 0L8 19l-1.8-1.3a2 2 0 0 0-2.2-.1Z"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1.5"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
                <circle cx="10" cy="11" r="1" />
                <circle cx="14" cy="11" r="1" />
              </svg>
            </div>
            <div class="status-copy">
              <p class="status-title">Aún no tienes escenas compartidas</p>
              <p class="status-sub">
                Crea tu primera escena desde la sección superior
                <span class="status-sub__hint">“Crear Nueva Escena Compartida”</span>.
              </p>
            </div>
          </article>
        </div>
      </div>
    </section>

    <!-- Scene Creation Modal -->
    <SceneModal
      v-if="showModal"
      @close="closeModal"
      @created="handleSceneCreated"
    />
  </div>
</template>

<script setup>
import NavBar from '@/components/NavBar.vue'
import SceneModal from '@/components/SceneModal.vue'
import { ref, onMounted } from 'vue'

// API base
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

// Modal state
const showModal = ref(false)

// User scenes from backend
const userScenes = ref([])

// Loading state for scenes
const isLoadingScenes = ref(true)

// Fetch user scenes from backend
async function fetchUserScenes() {
  isLoadingScenes.value = true
  try {
    const res = await fetch(`${API_BASE}/scenes`)
    if (!res.ok) throw new Error(`Error ${res.status}`)

    const data = await res.json()

    // Map backend response to card format
    userScenes.value = (data.scenes || []).map(scene => ({
      id: scene.name,
      title: scene.name,
      subtitle: 'Escena personalizada',
      img: `data:image/png;base64,${scene.thumbnail}`,
    }))
  } catch (err) {
    console.error('Failed to fetch user scenes:', err)
    // Silent fail - just show empty user scenes
    userScenes.value = []
  } finally {
    isLoadingScenes.value = false
  }
}

// Modal handlers
function openModal() {
  showModal.value = true
}

function closeModal() {
  showModal.value = false
}

async function handleSceneCreated(sceneName) {
  // Refetch scenes to include the new one
  await fetchUserScenes()
}

// Load user scenes on mount
onMounted(() => {
  fetchUserScenes()
})

function onImgError(e) {
  const svg = encodeURIComponent(`
    <svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'>
      <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='#e5e7eb'/>
          <stop offset='100%' stop-color='#f1f5f9'/>
        </linearGradient>
      </defs>
      <rect width='100%' height='100%' fill='url(#g)'/>
      <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
            fill='#64748b' font-family='system-ui, -apple-system, Segoe UI, Roboto, sans-serif' font-size='12'>
        200×200
      </text>
    </svg>
  `)
  e.target.src = `data:image/svg+xml;charset=utf-8,${svg}`
}
</script>

<style scoped>
/* ====== Theme Tokens ====== */
:root {
  --bg: #f3f4ff;
  --bg-soft: #fbfdff;
  --text: #0f172a;
  --muted: #475569;
  --accent: #2563eb;
  --accent-2: #7c3aed;
  --ring: rgba(37, 99, 235, 0.25);
  --shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
}

/* ====== Layout ====== */
.home {
  display: flex;
  flex-direction: column;
  gap: 24px;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}

.container {
  width: min(1200px, 92vw);
  margin: 0 auto;
}

/* ====== Hero ====== */
.hero {
  position: relative;
  overflow: clip;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
}
.hero__bg {
  position: absolute;
  inset: -2px;
  filter: blur(40px);
  background:
    radial-gradient(
      1200px 400px at 10% -10%,
      rgba(124, 172, 248, 0.25),
      transparent 60%
    ),
    radial-gradient(
      1000px 500px at 90% -20%,
      rgba(155, 140, 242, 0.2),
      transparent 60%
    ),
    linear-gradient(180deg, rgba(255, 255, 255, 0.9), #ffffff);
  animation: float 12s ease-in-out infinite alternate;
}
@keyframes float {
  to {
    transform: translateY(-12px);
  }
}
.hero__content {
  position: relative;
  padding: clamp(36px, 6vw, 72px) 16px;
  width: min(1200px, 92vw);
  margin: 0 auto;
}
.hero h1 {
  font-size: clamp(28px, 4.5vw, 56px);
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 10px;
}
.hero h1 span {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.lead {
  margin: 0;
  max-width: 60ch;
  color: var(--muted);
  font-size: clamp(14px, 1.8vw, 18px);
}

/* ====== Category ====== */
.categories {
  display: flex;
  flex-direction: column;
  gap: 36px;
  padding: 8px 0 48px;
}
.category {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.category__head {
  display: flex;
  align-items: center;
  gap: 12px;
}
.category h2 {
  font-size: clamp(18px, 2.2vw, 24px);
  margin: 0;
  letter-spacing: -0.01em;
}
.pill {
  font-size: 12px;
  color: var(--accent);
  border: 1px solid var(--accent);
  padding: 4px 8px;
  border-radius: 999px;
  background: linear-gradient(
    180deg,
    rgba(124, 172, 248, 0.12),
    rgba(124, 172, 248, 0.05)
  );
}

/* ====== Grid & Cards ====== */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 18px;
}

/* Cards SIEMPRE claras, como el modal */
.card {
  background:
    radial-gradient(
      140% 160% at 0% 0%,
      rgba(124, 172, 248, 0.22),
      transparent 55%
    ),
    radial-gradient(
      140% 160% at 100% 0%,
      rgba(155, 140, 242, 0.2),
      transparent 55%
    ),
    linear-gradient(180deg, #f9fafb, #ffffff);
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px) saturate(150%);
  transform: translateZ(0);
  transition:
    transform 0.25s ease,
    box-shadow 0.25s ease,
    border-color 0.25s ease,
    background 0.25s ease;
  will-change: transform;
}

.card:focus-within,
.card:hover {
  transform: translateY(-4px);
  border-color: var(--ring);
  box-shadow: 0 20px 50px rgba(15, 23, 42, 0.16);
}

/* Área de imagen cuadrada: la foto escala con el cuadrado */
.thumb {
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), #f3f4ff);
  aspect-ratio: 1 / 1;
  min-height: 200px; /* asegura que no sea muy chica en desktop */
}

/* La imagen ocupa un % del lado del cuadrado */
.thumb img {
  width: 72%;
  height: 72%;
  max-width: 400px;
  max-height: 400px;
  object-fit: cover;
  border-radius: 14px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
}

/* En mobile la hacemos aún más grande dentro del cuadrado */
@media (max-width: 768px) {
  .thumb img {
    width: 90%;
    height: 90%;
  }
}

/* Cuerpo de la card */
.card__body {
  padding: 14px 14px 16px;
  display: grid;
  gap: 8px;
}
.card__title {
  margin: 0;
  font-size: 16px;
  color: #0f172a;
}
.card__sub {
  margin: 0 0 6px;
  color: var(--muted);
  font-size: 13px;
}

/* ====== Button ====== */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 12px;
  text-decoration: none;
  color: #0f172a;
  background: linear-gradient(
    180deg,
    rgba(255, 255, 255, 0.95),
    rgba(248, 250, 252, 0.9)
  );
  border: 1px solid rgba(148, 163, 184, 0.45);
  outline: none;
  transition:
    transform 0.18s ease,
    background 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}
.btn__icon {
  width: 18px;
  height: 18px;
}
.btn:hover,
.btn:focus-visible {
  border-color: var(--ring);
  box-shadow: 0 0 0 6px var(--ring);
  transform: translateY(-1px);
}

/* ====== Add Scene Card ====== */
.add-scene-card {
  cursor: pointer;
  border-style: dashed;
  border-width: 2px;
  border-color: rgba(148, 163, 184, 0.5);
  background:
    radial-gradient(
      140% 160% at 0% 0%,
      rgba(124, 172, 248, 0.16),
      transparent 55%
    ),
    radial-gradient(
      140% 160% at 100% 0%,
      rgba(155, 140, 242, 0.16),
      transparent 55%
    ),
    linear-gradient(180deg, #f9fafb, #ffffff);
}

.add-scene-card:hover,
.add-scene-card:focus-visible {
  border-color: var(--accent);
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.16);
}

.add-thumb {
  background: transparent;
  display: grid;
  place-items: center;
}

.plus-icon {
  width: 60px;
  height: 60px;
  color: var(--muted);
  transition: color 0.2s, transform 0.2s;
}

.add-scene-card:hover .plus-icon,
.add-scene-card:focus-visible .plus-icon {
  color: var(--accent);
  transform: scale(1.1);
}

.add-scene-card .card__title {
  color: var(--muted);
}

.add-scene-card:hover .card__title,
.add-scene-card:focus-visible .card__title {
  color: #0f172a;
}

/* ====== Estados (loading / empty) ====== */
.status-wrapper {
  max-width: 480px;
}

.card--status {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 12px 14px 13px;
}

.status-icon {
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  background: radial-gradient(
      circle at 0% 0%,
      rgba(124, 172, 248, 0.35),
      transparent 60%
    ),
    radial-gradient(
      circle at 100% 100%,
      rgba(155, 140, 242, 0.35),
      transparent 60%
    );
  border: 1px solid rgba(148, 163, 184, 0.6);
}

/* Spinner animado para "cargando" */
.status-icon--spinner {
  border-radius: 999px;
  border: 2px solid rgba(148, 163, 184, 0.4);
  border-top-color: var(--accent);
  border-right-color: var(--accent-2);
  background: transparent;
  animation: spin 0.9s linear infinite;
}

/* Icono para vacío */
.status-icon--ghost svg {
  width: 20px;
  height: 20px;
  color: #4b5563;
}

/* Texto dentro del status card */
.status-copy {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.status-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #0f172a;
}

.status-sub {
  margin: 0;
  font-size: 12px;
  color: var(--muted);
}

.status-sub__hint {
  font-weight: 500;
  color: var(--accent);
}

/* Spin animation */
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ====== Motion preferences ====== */
@media (prefers-reduced-motion: reduce) {
  .hero__bg,
  .card,
  .btn,
  .plus-icon,
  .status-icon--spinner {
    transition: none;
    animation: none;
  }
}
</style>
