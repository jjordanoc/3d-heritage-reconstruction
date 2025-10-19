<!-- src/components/NavBar.vue -->
<template>
  <header class="nav-wrap">
    <nav class="nav container" role="navigation" aria-label="Main">

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
  gap: 10px;
  text-decoration: none;
  color: var(--text);
  font-weight: 600;
  letter-spacing: .2px;
}
.brand__dot {
  width: 16px; height: 16px;
  border-radius: 50%;
  background: radial-gradient(120% 120% at 30% 30%, var(--accent), var(--accent-2));
  box-shadow: 0 0 18px rgba(124,172,248,.55);
}
.brand__text {
  font-size: 15px;
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

/* Motion prefs */
@media (prefers-reduced-motion: reduce) {
  .cta, .link, .hamburger, .links { transition: none; }
}
</style>
