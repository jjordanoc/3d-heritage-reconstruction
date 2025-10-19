<template>
  <div ref="container" class="ply-root"></div>
</template>

<script>
import * as THREE from 'three'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

// Lee la base desde .env (VITE_API_BASE_URL). Si está vacío, usa same-origin (ideal si proxéas /pointcloud en Nginx/Vite).
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

// Utilidad: base64 → ArrayBuffer
function base64ToArrayBuffer(b64) {
  const comma = b64.indexOf(',')
  const raw = comma >= 0 ? b64.slice(comma + 1) : b64
  const bin = atob(raw)
  const len = bin.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i)
  return bytes.buffer
}

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
    onResize()

    // --- Cargar dato por ID desde la ruta ---
    const id = this.$route.query.id ?? this.$route.params.id
    if (id) this.loadById(String(id)).catch(console.error)
    else console.warn('[PLYViewer] No id in route (query/params).')

    // Watch cambios de id
    this.$watch(() => this.$route.fullPath, () => {
      const newId = this.$route.query.id ?? this.$route.params.id
      if (newId) this.loadById(String(newId)).catch(console.error)
    })

    // --- Limpieza ---
    this._cleanup = () => {
      cancelAnimationFrame(this._raf)
      window.removeEventListener('resize', onResize)
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
    /**
     * Pide el JSON { point_cloud: ..., camera: 4x4 }.
     * Admite point_cloud como:
     *  - base64
     */
    async loadById(id) {
      const base = API_BASE // '' = same-origin si proxéas
      const url = `${base}/pointcloud/${encodeURIComponent(id)}/latest`

      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status} al pedir ${url}`)

      const ctype = res.headers.get('content-type') || ''
      let cameraCv = null
      let plyBuffer = null
      let plyUrl = null

      if (ctype.includes('application/json')) {
        const json = await res.json()

        // 1) obtener el PLY
        if (typeof json.point_cloud === 'string') {
          if (json.point_cloud.startsWith('http') || json.point_cloud.startsWith('/')) {
            // URL directa
            plyUrl = json.point_cloud
          } else if (json.point_cloud.startsWith('data:') || /^[A-Za-z0-9+/]+=*$/.test(json.point_cloud)) {
            // dataURL o base64 "puro"
            plyBuffer = base64ToArrayBuffer(json.point_cloud)
          }
        } else {
          console.warn('[PLYViewer] point_cloud no es string; espera URL o base64.')
        }

        // 2) cámara (OpenCV Camera-to-World 4x4)
        if (Array.isArray(json.camera) && json.camera.length === 4 && json.camera.every(r => Array.isArray(r) && r.length === 4)) {
          cameraCv = json.camera
        }
      } else {
        // Fallback a binario PLY directo (compatibilidad con API antigua)
        plyBuffer = await res.arrayBuffer()
      }

      // Cargar PLY -> Geometry
      const loader = new PLYLoader()
      let geometry
      if (plyUrl) {
        // usa "await load" con promesa
        geometry = await new Promise((resolve, reject) => {
          loader.load(
            plyUrl,
            g => resolve(g),
            undefined,
            err => reject(err)
          )
        })
      } else if (plyBuffer) {
        geometry = loader.parse(plyBuffer)
      } else {
        throw new Error('[PLYViewer] No se pudo obtener point_cloud (ni URL ni buffer)')
      }

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

      // Reemplazar y montar
      this._removeCurrentObject()
      this._scene.add(object)
      this._pickables = [object]
      this._currentObject = object

      // Ajustar near/far en función del tamaño
      geometry.computeBoundingBox?.()
      const bbox = geometry.boundingBox
      let center = new THREE.Vector3(0, 0, 0)
      let maxDim = 1
      if (bbox) {
        center = bbox.getCenter(new THREE.Vector3())
        const size = bbox.getSize(new THREE.Vector3())
        maxDim = Math.max(size.x, size.y, size.z) || 1
        this._camera.near = Math.max(maxDim / 1000, 0.001)
        this._camera.far  = Math.max(maxDim * 100, 10)
        this._camera.updateProjectionMatrix()
      }

      // Si viene cámara → aplicarla. Si no, encuadrar por bbox.
      if (cameraCv) {
        this._applyOpenCVCameraToWorld(cameraCv)
      } else {
        // Fallback: encuadrar por BBox
        const fov = this._camera.fov * (Math.PI / 180)
        let camDist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5
        this._camera.position.set(center.x, center.y, center.z + camDist)
        this._camera.lookAt(center)
        this._orbit.target.copy(center)
        this._camera.updateProjectionMatrix()
        this._orbit.update()
      }
    },

    /**
     * Aplica una matriz 4x4 Camera-to-World en convención OpenCV a Three.js.
     * OpenCV: x→derecha, y→abajo, z→adelante
     * Three:  x→derecha, y→arriba,  z→hacia el observador (cámara mira -Z)
     *
     * Conversión aproximada: M_three = C * M_cv * C, con C = diag(1, -1, -1, 1)
     * Luego se asigna a camera.matrixWorld.
     */
    _applyOpenCVCameraToWorld(cv4x4) {
      // 1) construir Matrix4 desde filas (row-major)
      const a = cv4x4.flat()
      const Mcv = new THREE.Matrix4().set(
        a[0],  a[1],  a[2],  a[3],
        a[4],  a[5],  a[6],  a[7],
        a[8],  a[9],  a[10], a[11],
        a[12], a[13], a[14], a[15]
      )

      // 2) cambio de base OpenCV->Three
      const C = new THREE.Matrix4().set(
        1,  0,  0, 0,
        0, -1,  0, 0,
        0,  0, -1, 0,
        0,  0,  0, 1
      )
      const Mthree = new THREE.Matrix4().copy(C).multiply(Mcv).multiply(C)

      // 3) aplicar a la cámara
      this._camera.matrixAutoUpdate = false
      this._camera.matrixWorld.copy(Mthree)
      this._camera.matrixWorldNeedsUpdate = true

      // 4) actualizar posición / target
      this._camera.position.setFromMatrixPosition(Mthree)

      // dirección hacia adelante de la cámara en Three es -Z local
      const rot = new THREE.Matrix4().extractRotation(Mthree)
      const forward = new THREE.Vector3(0, 0, -1).applyMatrix4(rot).normalize()

      // coloca el target un poco delante de la cámara
      const target = new THREE.Vector3().copy(this._camera.position).add(forward)
      this._orbit.target.copy(target)
      this._camera.lookAt(target)
      this._camera.updateProjectionMatrix()
      this._orbit.update()
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
  background: #0b0d12;
  overflow: hidden;
}
</style>
