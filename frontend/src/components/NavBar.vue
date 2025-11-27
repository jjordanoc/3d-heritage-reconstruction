<!-- src/components/NavBar.vue -->
<template>
  <header class="nav-wrap">
    <div class="nav-shell">
      <!-- Solo logo UT3C Heritage (clic para ir al home) -->
      <router-link to="/" class="nav-logo" aria-label="Ir al inicio">
        <img
          src="/public/img/app/ut3c-heritage.png"
          alt="UT3C Heritage"
          class="nav-logo-img"
        />
      </router-link>
    </div>
  </header>
</template>

<script setup>
/* Sin estado, solo logo */
</script>

<style scoped>
/* ====== Barra fija blanca ====== */
.nav-wrap {
  position: sticky;
  top: 0;
  z-index: 20;
  padding: 10px 0 8px;
  background: #ffffff; /* blanco sólido */
  border-bottom: 1px solid rgba(148, 163, 184, 0.25);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

/* Contenedor alineado con el Home */
.nav-shell {
  width: min(1200px, 92vw);
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: flex-start; /* desktop: logo alineado a la izquierda */
}

/* Solo el logo */
.nav-logo {
  display: inline-flex;
  align-items: center;
  text-decoration: none;
}

/* Logo visible y nítido sobre blanco */
.nav-logo-img {
  height: 42px;
  width: auto;
  display: block;
  object-fit: contain;
}

/* ====== Mobile: logo centrado ====== */
@media (max-width: 768px) {
  .nav-wrap {
    padding: 8px 0 6px;
  }

  .nav-shell {
    justify-content: center; /* centra el logo en mobile */
  }

  .nav-logo-img {
    height: 36px;
  }
}
</style>
