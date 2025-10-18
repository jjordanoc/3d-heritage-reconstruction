<template>
  <Suspense>
    <AsyncPLYViewer />
  </Suspense>
</template>

<script setup>
import { defineAsyncComponent } from 'vue'
const AsyncPLYViewer = defineAsyncComponent(() => import('@/components/PLYViewer.vue'))
</script>
