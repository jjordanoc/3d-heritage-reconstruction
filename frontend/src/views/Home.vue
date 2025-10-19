<!-- src/views/Home.vue -->
<template>
  <div class="home">
    <NavBar />

    <!-- HERO -->
    <section class="hero">
      <div class="hero__bg"></div>
      <div class="hero__content">
        <h1>Reconstrucciones <span>3D</span></h1>
        <p class="lead">Explora colecciones de reconstrucciones (demo) con modelos, nubes de puntos y mallas texturizadas.</p>
      </div>
    </section>

    <!-- CATEGORIES -->
    <section class="categories container">
      <div
        v-for="cat in categories"
        :key="cat.key"
        class="category"
      >
        <div class="category__head">
          <h2>{{ cat.title }}</h2>
          <div class="pill">Demo</div>
        </div>

        <div class="grid">
          <article
            v-for="item in cat.items"
            :key="item.id"
            class="card"
            tabindex="0"
          >
            <div class="thumb">
              <img
                :src="item.img"
                :alt="item.title"
                loading="lazy"
                decoding="async"
                @error="onImgError($event)"
              />
            </div>

            <div class="card__body">
              <h3 class="card__title">{{ item.title }}</h3>
              <p class="card__sub">{{ item.subtitle }}</p>

              <router-link
                class="btn"
                :to="{ name: 'viewer', query: { id: item.id } }"
                aria-label="Ver {{ item.title }} en el visor"
              >
                <svg class="btn__icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M5 12h12M13 5l7 7-7 7" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                View
              </router-link>
            </div>
          </article>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import NavBar from '@/components/NavBar.vue'
import { reactive } from 'vue'

const categories = reactive([
  {
    key: 'patrimonio',
    title: 'Reconstrucciones de Patrimonio',
    items: [
      { id: 'huaca-pucllana', title: 'Huaca Pucallana', subtitle: 'Miraflores, Lima', img: '/img/patrimonio/pucllana.jpg' },
    ],
  },
  // {
  //   key: 'estructuras',
  //   title: 'Reconstrucciones de Estructuras',
  //   items: [
  //     { id: 'puente-a', title: 'Puente A', subtitle: 'Modelo FEM', img: '/img/estructuras/bridge-a.jpg' },
  //   ],
  // },
  {
    key: 'entornos',
    title: 'Reconstrucciones de Entornos',
    items: [
      { id: 'auditorio-utec', title: 'Auditorio UTEC', subtitle: 'UTEC, Barranco', img: '/img/entornos/auditorio-utec.jpg' },
    ],
  },
])

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
  --bg: #0b0d12;
  --bg-soft: #0f131a;
  --card: rgba(255,255,255,0.06);
  --card-stroke: rgba(255,255,255,0.12);
  --text: #e6e9ef;
  --muted: #9aa4b2;
  --accent: #7cacf8;
  --accent-2: #9b8cf2;
  --ring: rgba(124,172,248,0.45);
  --shadow: 0 10px 30px rgba(0,0,0,.25);
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
    --accent-2: #7c3aed;
    --ring: rgba(37,99,235,.25);
    --shadow: 0 12px 28px rgba(2,6,23,.12);
  }
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
  border-bottom: 1px solid var(--card-stroke);
}
.hero__bg {
  position: absolute;
  inset: -2px;
  filter: blur(40px);
  background:
    radial-gradient(1200px 400px at 10% -10%, rgba(124,172,248,.25), transparent 60%),
    radial-gradient(1000px 500px at 90% -20%, rgba(155,140,242,.22), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,.06), transparent 40%);
  animation: float 12s ease-in-out infinite alternate;
}
@keyframes float {
  to { transform: translateY(-12px); }
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
  background: linear-gradient(180deg, rgba(124,172,248,.12), rgba(124,172,248,.05));
}

/* ====== Grid & Cards ====== */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 18px;
}

.card {
  background: var(--card);
  border: 1px solid var(--card-stroke);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: var(--shadow);
  backdrop-filter: saturate(140%) blur(6px);
  transform: translateZ(0);
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  will-change: transform;
}
.card:focus-within,
.card:hover {
  transform: translateY(-4px);
  border-color: var(--ring);
  box-shadow: 0 16px 40px rgba(0,0,0,.28);
}

/* Área de imagen cuadrada (≈200×200, responsivo) */
.thumb {
  display: grid;
  place-items: center;
  background: linear-gradient(180deg, rgba(255,255,255,.06), transparent);
  aspect-ratio: 1 / 1;
  min-height: 200px; /* asegura ~200px en pantallas pequeñas */
}
.thumb img {
  width: clamp(180px, 20vw, 220px);
  height: clamp(180px, 20vw, 220px);
  object-fit: cover;
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0,0,0,.25);
}

.card__body {
  padding: 14px 14px 16px;
  display: grid;
  gap: 8px;
}
.card__title {
  margin: 0;
  font-size: 16px;
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
  color: var(--text);
  background:
    linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
  border: 1px solid var(--card-stroke);
  outline: none;
  transition: transform .18s ease, background .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.btn__icon {
  width: 18px; height: 18px;
}
.btn:hover, .btn:focus-visible {
  border-color: var(--ring);
  box-shadow: 0 0 0 6px var(--ring);
  transform: translateY(-1px);
}

/* ====== Motion preferences ====== */
@media (prefers-reduced-motion: reduce) {
  .hero__bg,
  .card,
  .btn { transition: none; animation: none; }
}
</style>
