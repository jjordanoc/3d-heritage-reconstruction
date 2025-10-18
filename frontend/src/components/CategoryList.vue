<template>
  <div class="category-list">
    <h2>Categorías</h2>
    <ul>
      <li>Reconstrucciones de Patrimonio</li>
      <li>Reconstrucciones de Edificaciones</li>
      <li>Reconstrucciones de Prueba</li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'CategoryList'
}
</script>

<style scoped>
.category-list {
  margin: 20px 0;
}

.category-list h2 {
  font-size: 24px;
  margin-bottom: 10px;
}

.category-list ul {
  list-style-type: none;
  padding: 0;
}

.category-list li {
  font-size: 18px;
  margin: 5px 0;
}
</style>