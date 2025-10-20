<!-- src/components/NavBar.vue -->
<template>
  <header class="nav-wrap">
    <nav class="nav container" role="navigation" aria-label="Main">
      <!-- Logo / Brand -->
      <router-link to="/" class="brand" aria-label="UT3C Heritage - Inicio">
        <svg class="brand__icon" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- Outer ring with heritage symbolism -->
          <circle cx="20" cy="20" r="18" stroke="url(#gradient1)" stroke-width="1.5" opacity="0.6"/>
          <circle cx="20" cy="20" r="15" stroke="url(#gradient1)" stroke-width="1" opacity="0.4"/>
          
          <!-- 3D cube/monument symbol in center -->
          <path d="M20 8 L28 13 L28 20 L20 25 L12 20 L12 13 Z" 
                fill="url(#gradient2)" opacity="0.2"/>
          <path d="M20 8 L28 13 L20 18 Z" 
                fill="url(#gradient3)" opacity="0.8"/>
          <path d="M20 18 L28 13 L28 20 L20 25 Z" 
                fill="url(#gradient2)" opacity="0.5"/>
          <path d="M20 18 L12 13 L12 20 L20 25 Z" 
                fill="url(#gradient4)" opacity="0.6"/>
          
          <!-- Camera viewfinder corners -->
          <path d="M8 8 L11 8 L11 9 L9 9 L9 11 L8 11 Z" fill="var(--accent)"/>
          <path d="M32 8 L29 8 L29 9 L31 9 L31 11 L32 11 Z" fill="var(--accent)"/>
          <path d="M8 32 L11 32 L11 31 L9 31 L9 29 L8 29 Z" fill="var(--accent)"/>
          <path d="M32 32 L29 32 L29 31 L31 31 L31 29 L32 29 Z" fill="var(--accent)"/>
          
          <defs>
            <linearGradient id="gradient1" x1="0" y1="0" x2="40" y2="40">
              <stop offset="0%" stop-color="var(--accent)"/>
              <stop offset="100%" stop-color="var(--accent-2)"/>
            </linearGradient>
            <linearGradient id="gradient2" x1="20" y1="8" x2="20" y2="25">
              <stop offset="0%" stop-color="var(--accent)"/>
              <stop offset="100%" stop-color="var(--accent-2)"/>
            </linearGradient>
            <linearGradient id="gradient3" x1="20" y1="8" x2="28" y2="18">
              <stop offset="0%" stop-color="var(--accent)" stop-opacity="1"/>
              <stop offset="100%" stop-color="var(--accent-2)" stop-opacity="0.6"/>
            </linearGradient>
            <linearGradient id="gradient4" x1="12" y1="13" x2="20" y2="25">
              <stop offset="0%" stop-color="var(--accent-2)"/>
              <stop offset="100%" stop-color="var(--accent)"/>
            </linearGradient>
          </defs>
        </svg>
        <div class="brand__text">
          <div class="brand__title">UT3C Heritage</div>
          <div class="brand__subtitle">3D Reconstruction</div>
        </div>
      </router-link>

      <ul class="links" :class="{ open }">
        <!-- Home como texto, ya no redirige -->
        <li>
          <router-link
            to="/"
            class="link"
            aria-label="Ir a la página de inicio"
          >
            Inicio
          </router-link>
        </li>
      </ul>
    </nav>
  </header>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'

const open = ref(false)
const route = useRoute()
// Cierra el menú al navegar
watch(() => route.fullPath, () => (open.value = false))
</script>

<style scoped>
/* ===== Tokens (match con tu Home bello) ===== */
:root {
  --bg: #0b0d12;
  --glass: rgba(255,255,255,0.06);
  --stroke: rgba(255,255,255,0.10);
  --text: #e6e9ef;
  --muted: #9aa4b2;
  --accent: #7cacf8;
  --accent-2: #9b8cf2;
  --ring: rgba(124,172,248,0.45);
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #fff;
    --glass: rgba(16,24,40,0.04);
    --stroke: rgba(16,24,40,0.12);
    --text: #0f172a;
    --muted: #475569;
    --accent: #2563eb;
    --accent-2: #7c3aed;
    --ring: rgba(37,99,235,.25);
  }
}

/* ===== Layout ===== */
.nav-wrap {
  position: sticky;
  top: 0;
  z-index: 1000;
  backdrop-filter: saturate(140%) blur(10px);
  background:
    linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,0)) ,
    var(--glass);
  border-bottom: 1px solid var(--stroke);
}
.container {
  width: min(1200px, 92vw);
  margin: 0 auto;
}
.nav {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 12px;
  min-height: 64px;
  color: var(--text);
}

/* ===== Brand ===== */
.brand {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  color: var(--text);
  font-weight: 600;
  letter-spacing: .2px;
  transition: transform .2s ease, filter .2s ease;
}
.brand:hover {
  transform: translateY(-1px);
  filter: brightness(1.1);
}
.brand__icon {
  width: 40px;
  height: 40px;
  flex-shrink: 0;
  filter: drop-shadow(0 2px 8px rgba(124,172,248,.35));
  transition: filter .2s ease;
}
.brand:hover .brand__icon {
  filter: drop-shadow(0 4px 12px rgba(124,172,248,.5));
}
.brand__text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  line-height: 1.2;
}
.brand__title {
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
  letter-spacing: 0.3px;
  transition: color .2s ease;
}
.brand:hover .brand__title {
  color: var(--accent);
}
.brand__subtitle {
  font-size: 10px;
  font-weight: 500;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  transition: color .2s ease;
}
.brand:hover .brand__subtitle {
  color: #475569;
}

@media (prefers-color-scheme: light) {
  .brand__title {
    color: #0f172a;
  }
  .brand__subtitle {
    color: #64748b;
  }
}

/* ===== Links ===== */
.links {
  margin: 0;
  padding: 0;
  list-style: none;
  justify-self: end;

  display: flex;
  align-items: center;
  gap: 8px;
}
.link {
  display: inline-flex;
  align-items: center;
  height: 36px;
  padding: 0 12px;
  border-radius: 10px;
  color: var(--muted);
  text-decoration: none;
  border: 1px solid transparent;
  transition: color .15s ease, border-color .2s ease, background .2s ease, transform .15s ease;
}
.link:hover,
.link:focus-visible {
  color: var(--text);
  background: rgba(255,255,255,.05);
  border-color: var(--stroke);
  outline: none;
}
.router-link-active.link {
  color: var(--text);
  background: linear-gradient(180deg, rgba(124,172,248,.12), rgba(124,172,248,.05));
  border-color: var(--ring);
  box-shadow: 0 0 0 6px var(--ring);
}

/* ===== CTA ===== */
.cta {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 40px;
  padding: 0 14px;
  border-radius: 12px;
  text-decoration: none;
  color: #fff;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border: 1px solid transparent;
  transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
  will-change: transform;
}
.cta:hover,
.cta:focus-visible {
  transform: translateY(-1px);
  filter: brightness(1.05);
  box-shadow: 0 8px 28px rgba(0,0,0,.25);
  outline: none;
}
.cta__icon { width: 18px; height: 18px; }

/* ===== Hamburger (mobile) ===== */
.hamburger {
  justify-self: end;
  width: 40px; height: 40px;
  border-radius: 10px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,.04);
  display: none; /* visible sólo en móvil */
  align-items: center;
  justify-content: center;
  gap: 4px;
  cursor: pointer;
  transition: background .2s ease, border-color .2s ease;
}
.hamburger:hover { background: rgba(255,255,255,.08); }
.hamburger span {
  display: block;
  width: 18px; height: 2px;
  background: var(--text);
  border-radius: 2px;
}
.hamburger span:nth-child(2) { width: 14px; }
.hamburger span:nth-child(3) { width: 10px; }

/* ===== Responsive ===== */
@media (max-width: 800px) {
  .hamburger { display: inline-flex; }
  
  .brand__icon {
    width: 36px;
    height: 36px;
  }
  .brand__title {
    font-size: 15px;
  }
  .brand__subtitle {
    font-size: 9px;
  }
  
  .links {
    position: absolute;
    top: 64px; right: 0; left: 0;
    display: grid;
    gap: 8px;
    padding: 10px 14px 14px;
    background: var(--glass);
    border-bottom: 1px solid var(--stroke);
    transform-origin: top;
    transform: scaleY(0.96);
    opacity: 0;
    pointer-events: none;
    transition: transform .2s ease, opacity .2s ease;
  }
  .links.open {
    opacity: 1;
    pointer-events: auto;
    transform: scaleY(1);
  }
  .link,
  .cta {
    justify-content: center;
    width: 100%;
  }
}

@media (max-width: 500px) {
  .brand__text {
    display: none;
  }
  .brand__icon {
    width: 40px;
    height: 40px;
  }
}

/* Motion prefs */
@media (prefers-reduced-motion: reduce) {
  .cta, .link, .hamburger, .links { transition: none; }
}
</style>
