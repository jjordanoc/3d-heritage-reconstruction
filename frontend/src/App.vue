<style>
/* Solo se desactiva el scroll cuando el Viewer lo pide */
body.no-scroll {
  overflow: hidden;
  height: 100%;
}
</style>

<template>
  <RouterView />
  <UserOnboardingModal v-if="showOnboarding" @close="showOnboarding = false" />
</template>

<script setup>
import { ref, onMounted, defineAsyncComponent } from 'vue'

const UserOnboardingModal = defineAsyncComponent(() => import('@/components/UserOnboardingModal.vue'))
const showOnboarding = ref(false)

onMounted(() => {
  const user = localStorage.getItem('heritage_user')
  if (!user) {
    showOnboarding.value = true
  }
})
</script>
