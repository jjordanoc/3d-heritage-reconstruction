<template>
  <div ref="container" class="ply-root"></div>
</template>

<script>
import * as THREE from 'three'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

// Lee la base desde .env (VITE_API_BASE_URL). Si está vacío, usa same-origin (ideal si proxéas /pointcloud en Nginx/Vite).
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

export default {
  name: 'PLYViewer',
  data() {
    return {
      _raf: null,
      _cleanup: null,

      _scene: null,
      _camera: null,
      _renderer: null,
      _orbit: null,

      _pickables: [],
      _currentObject: null,
    }
  },

  mounted() {
    const container = this.$refs.container

    // --- Escena / Cámara / Render ---
    const scene = this._scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0b0d12)

    const camera = this._camera = new THREE.PerspectiveCamera(
      60,
      Math.max(container.clientWidth, 1) / Math.max(container.clientHeight, 1),
      0.001,
      1e6
    )
    camera.position.set(0, 0, 3)

    const renderer = this._renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1, false)
    container.appendChild(renderer.domElement)

    // --- Luces ---
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6))
    const dir = new THREE.DirectionalLight(0xffffff, 0.9)
    dir.position.set(5, 10, 7.5)
    scene.add(dir)

    // --- Controles ---
    const orbit = this._orbit = new ArcballControls(camera, renderer.domElement, scene)
    orbit.setGizmosVisible(false)
    orbit.enableAnimations = true
    orbit.dampingFactor = 0.1

    // --- Interacción opcional: doble click para cambiar pivot ---
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()
    const onDblClick = (e) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(this._pickables, true)
      if (hits.length) {
        this._setPivot(hits[0].point)
      }
    }
    renderer.domElement.addEventListener('dblclick', onDblClick)

    // --- Animación ---
    const animate = () => {
      this._raf = requestAnimationFrame(animate)
      orbit.update()
      renderer.render(scene, camera)
    }
    animate()

    // --- Resize ---
    const onResize = () => {
      const w = Math.max(container.clientWidth, 1)
      const h = Math.max(container.clientHeight, 1)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
      orbit.update()
    }
    window.addEventListener('resize', onResize)
    // Inicial
    onResize()

    // --- Cargar PLY por ID desde la ruta ---
    const id = this.$route.query.id ?? this.$route.params.id
    if (id) this.loadPLYById(String(id)).catch(console.error)
    else console.warn('[PLYViewer] No id in route (query/params).')

    // Watch: si cambia el id en la URL, recarga el modelo
    this.$watch(() => this.$route.fullPath, () => {
      const newId = this.$route.query.id ?? this.$route.params.id
      if (newId) this.loadPLYById(String(newId)).catch(console.error)
    })

    // --- Limpieza ---
    this._cleanup = () => {
      cancelAnimationFrame(this._raf)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('dblclick', onDblClick)
      this._removeCurrentObject()
      orbit.dispose?.()
      renderer.dispose?.()
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement)
    }
  },

  beforeUnmount() {
    this._cleanup && this._cleanup()
  },

  methods: {
    async loadPLYById(id) {
      const base = API_BASE // '' = same-origin (si tienes proxy /pointcloud)
      const url = `${base}/pointcloud/${encodeURIComponent(id)}/latest`

      const res = await fetch(url)
      if (!res.ok) throw new Error(`Error ${res.status} al obtener PLY para id=${id}`)
      const buf = await res.arrayBuffer()

      // Parsear
      const loader = new PLYLoader()
      let geometry = loader.parse(buf)

      // Crear objeto
      geometry.computeVertexNormals?.()
      let object
      if (geometry.getAttribute && geometry.getAttribute('color')) {
        const mat = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, sizeAttenuation: true })
        object = new THREE.Points(geometry, mat)
        object.raycast = THREE.Points.prototype.raycast
      } else {
        const mat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false })
        object = new THREE.Mesh(geometry, mat)
      }

      // Reemplazar si había otro
      this._removeCurrentObject()
      this._scene.add(object)
      this._pickables = [object]
      this._currentObject = object

      // Encajar cámara/pivot robusto
      geometry.computeBoundingBox?.()
      const bbox = geometry.boundingBox
      if (bbox) {
        const center = new THREE.Vector3()
        bbox.getCenter(center)
        const size = bbox.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z) || 1

        const fov = this._camera.fov * (Math.PI / 180)
        let camDist = Math.abs(maxDim / (2 * Math.tan(fov / 2)))
        camDist *= 1.5

        this._camera.position.set(center.x, center.y, center.z + camDist)
        this._camera.near = Math.max(maxDim / 1000, 0.001)
        this._camera.far  = Math.max(maxDim * 100, 10)
        this._camera.lookAt(center)
        this._camera.updateProjectionMatrix()

        this._orbit.target.copy(center)
        this._orbit.update()
      }
    },

    _removeCurrentObject() {
      if (!this._currentObject) return
      this._scene.remove(this._currentObject)
      this._currentObject.geometry?.dispose?.()
      const m = this._currentObject.material
      if (Array.isArray(m)) m.forEach(mm => mm?.dispose?.())
      else m?.dispose?.()
      this._currentObject = null
      this._pickables = []
    },

    _setPivot(newPivot) {
      const camToPivot = new THREE.Vector3().subVectors(this._camera.position, this._orbit.target)
      this._orbit.target.copy(newPivot)
      this._camera.position.copy(newPivot).add(camToPivot)
      this._camera.updateProjectionMatrix()
      this._orbit.update()
    },
  },
}
</script>

<style scoped>
.ply-root {
  width: 100%;
  height: 100%;
  display: block;
  background: #0b0d12; /* da contraste si el modelo es claro */
  overflow: hidden;
}
</style>
