<template>
  <div ref="container" class="gsplat-root">
    <!-- HUD / Controles -->
    <div class="hud">
      <div class="row">
        <label>Velocidad</label>
        <input type="range" min="0.2" max="20" step="0.1" v-model.number="moveSpeed" />
        <span class="pill">{{ moveSpeed.toFixed(1) }} u/s</span>
      </div>
      <div class="row toggles">
        <label><input type="checkbox" v-model="showPivot" @change="togglePivot"/> Pivot</label>
        <label><input type="checkbox" v-model="showAxes" @change="toggleAxes"/> Ejes</label>
        <label><input type="checkbox" v-model="showGrid" @change="toggleGrid"/> Grilla</label>
      </div>
      <div class="coords">
        <div><strong>Cam:</strong> x {{ camPos.x.toFixed(3) }} | y {{ camPos.y.toFixed(3) }} | z {{ camPos.z.toFixed(3) }}</div>
        <div><strong>Target:</strong> x {{ camTarget.x.toFixed(3) }} | y {{ camTarget.y.toFixed(3) }} | z {{ camTarget.z.toFixed(3) }}</div>
        <div style="opacity:.85;margin-top:4px">
          <strong>Tips:</strong> Doble clic = fijar pivote · F = encuadrar · Shift=x3 · Ctrl=x0.25
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as THREE from 'three'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
import { SplatMesh } from '@sparkjsdev/spark'

const API_BASE = (import.meta.env.VITE_SPLAT_API_BASE_URL || '').replace(/\/+$/, '')

export default {
  name: 'GSplatViewer',
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

      // Helpers
      _axes: null,
      _grid: null,

      // BBoxes
      _worldBBox: new THREE.Box3(),
      _clampBBox: new THREE.Box3(),

      // UI state
      moveSpeed: 3.5,
      showPivot: true,
      showAxes: true,
      showGrid: true,

      // HUD coords
      camPos: new THREE.Vector3(),
      camTarget: new THREE.Vector3(),

      // Pivot viz
      _pivotMarker: null,
      _pivotAxes: null,
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
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1, true)
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
    orbit.dampingFactor = 0.12
    if ('minDistance' in orbit) orbit.minDistance = 0.01
    if ('maxDistance' in orbit) orbit.maxDistance = 1e6
    if ('zoomSpeed' in orbit) orbit.zoomSpeed = 0.8

    // --- Helpers ---
    this._createHelpers(1)

    // --- Pivot visual ---
    this._createPivotViz()
    this._updatePivotViz()

    // --- WASD + modificadores ---
    const keys = {
      KeyW: false, KeyA: false, KeyS: false, KeyD: false,
      Space: false, ShiftLeft: false, ShiftRight: false,
      ControlLeft: false, ControlRight: false,
      KeyQ: false, KeyE: false
    }
    const onKeyDown = (e) => {
      if (e.code in keys) keys[e.code] = true
      if (e.code === 'Equal') this.moveSpeed = Math.min(this.moveSpeed + 0.5, 50)
      if (e.code === 'Minus') this.moveSpeed = Math.max(this.moveSpeed - 0.5, 0.1)
      if (e.code === 'KeyF') this._frameToBBox()
    }
    const onKeyUp = (e) => { if (e.code in keys) keys[e.code] = false }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)

    // --- Doble clic: fijar pivote con raycast ---
    const onDblClick = (ev) => {
      if (!this._pickables.length) return
      const rect = renderer.domElement.getBoundingClientRect()
      const ndc = new THREE.Vector2(
        ((ev.clientX - rect.left) / rect.width) * 2 - 1,
        -(((ev.clientY - rect.top) / rect.height) * 2 - 1)
      )
      const raycaster = new THREE.Raycaster()
      raycaster.setFromCamera(ndc, camera)
      const intersects = raycaster.intersectObjects(this._pickables, true)
      if (intersects.length) {
        const p = intersects[0].point
        this._setPivot(p)
        this._updatePivotViz()
      }
    }
    renderer.domElement.addEventListener('dblclick', onDblClick)

    // --- Animación ---
    const clock = new THREE.Clock()
    const upVec = new THREE.Vector3(0, 1, 0)
    const tmpFwd = new THREE.Vector3()
    const tmpRight = new THREE.Vector3()

    const animate = () => {
      this._raf = requestAnimationFrame(animate)
      const dt = clock.getDelta()

      // velocidad
      let speed = this.moveSpeed
      const fast = keys.ShiftLeft || keys.ShiftRight
      const precise = keys.ControlLeft || keys.ControlRight
      if (fast) speed *= 3.0
      if (precise) speed *= 0.25

      if (keys.KeyW || keys.KeyS || keys.KeyA || keys.KeyD || keys.Space || keys.ShiftLeft || keys.ShiftRight || keys.KeyQ || keys.KeyE) {
        camera.matrixAutoUpdate = true
        camera.getWorldDirection(tmpFwd).normalize()
        tmpRight.copy(tmpFwd).cross(upVec).normalize()

        const vel = speed * dt
        const delta = new THREE.Vector3()

        if (keys.KeyW) delta.addScaledVector(tmpFwd,  vel)
        if (keys.KeyS) delta.addScaledVector(tmpFwd, -vel)
        if (keys.KeyA) delta.addScaledVector(tmpRight, -vel)
        if (keys.KeyD) delta.addScaledVector(tmpRight,  vel)
        if (keys.Space || keys.KeyE) delta.addScaledVector(upVec,    vel)
        if (keys.ShiftLeft || keys.ShiftRight || keys.KeyQ) delta.addScaledVector(upVec, -vel)

        camera.position.add(delta)
        orbit.target.add(delta)
        camera.updateProjectionMatrix()
      }

      // Clamp del target
      if (!this._clampBBox.isEmpty()) {
        const clamped = this._clampBBox.clampPoint(orbit.target, new THREE.Vector3())
        orbit.target.copy(clamped)
      }

      // HUD
      this.camPos.copy(camera.position)
      this.camTarget.copy(orbit.target)

      // Pivot viz
      this._updatePivotViz()

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
      renderer.setSize(w, h, true)
      orbit.update()
    }
    window.addEventListener('resize', onResize)
    onResize()

    // --- Cargar dato por ID desde la ruta ---
    const id = this.$route?.query?.id ?? this.$route?.params?.id
    if (id) this.loadById(String(id)).catch(console.error)
    else console.warn('[GSplatViewer] No id in route (query/params).')

    // Watch cambios de id
    this.$watch(() => this.$route?.fullPath, () => {
      const newId = this.$route?.query?.id ?? this.$route?.params?.id
      if (newId) this.loadById(String(newId)).catch(console.error)
    })

    // --- Limpieza ---
    this._cleanup = () => {
      cancelAnimationFrame(this._raf)
      window.removeEventListener('resize', onResize)
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      renderer.domElement.removeEventListener('dblclick', onDblClick)
      this._removeCurrentObject()
      this._disposeHelpers()
      this._disposePivotViz()
      orbit.dispose?.()
      renderer.dispose?.()
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement)
    }
  },

  beforeUnmount() { this._cleanup && this._cleanup() },

  methods: {
    async loadById(id, preserveCamera = false) {
      // Save current camera state if preserving
      let savedCameraPos = null
      let savedCameraTarget = null
      if (preserveCamera && this._camera && this._orbit) {
        savedCameraPos = this._camera.position.clone()
        savedCameraTarget = this._orbit.target.clone()
      }

      const base = API_BASE
      const url = `${base}/scene/${encodeURIComponent(id)}/gsplat/model`
      
      console.log('[GSplatViewer] Loading gsplat from:', url)

      // Fetch the .ply file ourselves to avoid 431 error with Spark's internal fetch
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`Failed to fetch gsplat model: ${response.status} ${response.statusText}`)
      }
      
      const blob = await response.blob()
      console.log('[GSplatViewer] Fetched blob:', blob.size, 'bytes')
      
      // Create a blob URL
      const blobUrl = URL.createObjectURL(blob)
      console.log('[GSplatViewer] Created blob URL:', blobUrl)
      
      // Create SplatMesh with blob URL
      const splatMesh = new SplatMesh({ url: blobUrl })
      
      // Wait for splat to load
      await new Promise((resolve, reject) => {
        const checkLoaded = setInterval(() => {
          // Check if the splat has loaded by checking if it has geometry/data
          if (splatMesh.geometry && splatMesh.geometry.attributes) {
            clearInterval(checkLoaded)
            resolve()
          }
        }, 100)
        
        // Timeout after 30 seconds
        setTimeout(() => {
          clearInterval(checkLoaded)
          reject(new Error('Splat loading timeout'))
        }, 30000)
      })

      console.log('[GSplatViewer] Splat loaded successfully')
      
      // Clean up blob URL after loading
      URL.revokeObjectURL(blobUrl)

      // Calculate bounding box from splat
      splatMesh.geometry.computeBoundingBox()
      const bbox = splatMesh.geometry.boundingBox

      let center = new THREE.Vector3(0, 0, 0)
      let maxDim = 1
      if (bbox) {
        center = bbox.getCenter(new THREE.Vector3())
        const size = bbox.getSize(new THREE.Vector3())
        maxDim = Math.max(size.x, size.y, size.z) || 1
      }

      this._removeCurrentObject()
      this._scene.add(splatMesh)
      this._pickables = [splatMesh]
      this._currentObject = splatMesh

      // BBoxes globales
      this._worldBBox.makeEmpty()
      if (splatMesh.geometry?.boundingBox) this._worldBBox.union(splatMesh.geometry.boundingBox)
      this._updateClampBBox()

      // Helpers al tamaño
      this._recalibrateHelpers(maxDim)

      // near/far y límites de distancia
      this._camera.near = Math.max(maxDim / 1000, 0.001)
      this._camera.far  = Math.max(maxDim * 100, 10)
      this._camera.updateProjectionMatrix()
      if ('minDistance' in this._orbit) this._orbit.minDistance = Math.max(maxDim * 0.02, 0.001)
      if ('maxDistance' in this._orbit) this._orbit.maxDistance = Math.max(maxDim * 20, 10)

      // Handle camera positioning
      if (preserveCamera && savedCameraPos && savedCameraTarget) {
        // Restore saved camera position
        this._camera.matrixAutoUpdate = true
        this._camera.position.copy(savedCameraPos)
        this._orbit.target.copy(savedCameraTarget)
        this._camera.updateProjectionMatrix()
        this._orbit.update()
      } else {
        this._frameToBBox()
      }
      this._updatePivotViz()
    },

    async reloadPointCloud(id) {
      await this.loadById(id, true) // preserveCamera = true
      this.$emit('loadComplete')
    },

    // === Utilidades de cámara ===
    _frameToBBox() {
      if (this._worldBBox.isEmpty()) return
      const center = this._worldBBox.getCenter(new THREE.Vector3())
      const size = this._worldBBox.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z) || 1
      const fov = this._camera.fov * (Math.PI / 180)
      const camDist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.35
      this._camera.matrixAutoUpdate = true
      this._camera.position.set(center.x, center.y, center.z + camDist)
      this._orbit.target.copy(center)
      this._camera.updateProjectionMatrix()
      if ('minDistance' in this._orbit) this._orbit.minDistance = Math.max(maxDim * 0.02, 0.001)
      if ('maxDistance' in this._orbit) this._orbit.maxDistance = Math.max(maxDim * 20, 10)
      this._updatePivotViz()
    },

    _removeCurrentObject() {
      for (const obj of this._pickables) {
        this._scene.remove(obj)
        obj.geometry?.dispose?.()
        const m = obj.material
        if (Array.isArray(m)) m.forEach(mm => mm?.dispose?.())
        else m?.dispose?.()
      }
      this._currentObject = null
      this._pickables = []
      this._worldBBox.makeEmpty()
      this._updateClampBBox()
    },

    _setPivot(newPivot) {
      const camToPivot = new THREE.Vector3().subVectors(this._camera.position, this._orbit.target)
      this._orbit.target.copy(newPivot)
      this._camera.position.copy(newPivot).add(camToPivot)
      this._camera.updateProjectionMatrix()
      this._orbit.update()
      this._updatePivotViz()
    },

    // ===== Helpers (Axes + Grid) =====
    _createHelpers(size = 1) {
      this._axes = new THREE.AxesHelper(size)
      this._axes.visible = this.showAxes
      this._scene.add(this._axes)

      this._grid = new THREE.GridHelper(size * 10, 10)
      this._grid.material.opacity = 0.25
      this._grid.material.transparent = true
      this._grid.visible = this.showGrid
      this._scene.add(this._grid)
    },

    _disposeHelpers() {
      if (this._axes) { this._scene.remove(this._axes); this._axes.geometry?.dispose?.(); this._axes = null }
      if (this._grid) {
        this._scene.remove(this._grid)
        this._grid.geometry?.dispose?.()
        Array.isArray(this._grid.material) ? this._grid.material.forEach(m => m.dispose?.()) : this._grid.material?.dispose?.()
        this._grid = null
      }
    },

    _recalibrateHelpers(maxDim) {
      const size = Math.max(maxDim, 1)
      const axesVis = this.showAxes
      const gridVis = this.showGrid
      this._disposeHelpers()
      this._createHelpers(size * 0.5)
      if (this._grid) {
        const gridSize = Math.max(size * 5, 1)
        const divisions = 10
        this._scene.remove(this._grid)
        this._grid.geometry?.dispose?.()
        Array.isArray(this._grid.material) ? this._grid.material.forEach(m => m.dispose?.()) : this._grid.material?.dispose?.()
        this._grid = new THREE.GridHelper(gridSize, divisions)
        this._grid.material.opacity = 0.25
        this._grid.material.transparent = true
        this._scene.add(this._grid)
      }
      if (this._axes) this._axes.visible = axesVis
      if (this._grid) this._grid.visible = gridVis
    },

    _updateClampBBox() {
      if (this._worldBBox.isEmpty()) {
        this._clampBBox.makeEmpty()
        return
      }
      const size = this._worldBBox.getSize(new THREE.Vector3())
      const pad = Math.max(size.x, size.y, size.z) * 0.2
      this._clampBBox.copy(this._worldBBox).expandByScalar(pad)
    },

    // ===== Pivot viz =====
    _createPivotViz() {
      if (!this._scene) return
      const geo = new THREE.SphereGeometry(0.05, 24, 16)
      const mat = new THREE.MeshBasicMaterial({ color: 0xffff00 })
      this._pivotMarker = new THREE.Mesh(geo, mat)
      this._pivotMarker.visible = this.showPivot
      this._scene.add(this._pivotMarker)

      this._pivotAxes = new THREE.AxesHelper(0.15)
      this._pivotAxes.visible = this.showPivot
      this._scene.add(this._pivotAxes)
    },

    _updatePivotViz() {
      if (!this._pivotMarker || !this._pivotAxes || !this._orbit || !this._camera) return
      const t = this._orbit.target
      this._pivotMarker.position.copy(t)
      this._pivotAxes.position.copy(t)
      const dist = this._camera.position.distanceTo(t) || 1
      const k = THREE.MathUtils.clamp(dist * 0.02, 0.5, 4.0)
      this._pivotMarker.scale.setScalar(k)
      this._pivotAxes.scale.setScalar(k * 1.2)
      this._pivotMarker.visible = this.showPivot
      this._pivotAxes.visible = this.showPivot
    },

    _disposePivotViz() {
      if (this._pivotMarker) {
        this._scene.remove(this._pivotMarker)
        this._pivotMarker.geometry?.dispose?.()
        this._pivotMarker.material?.dispose?.()
        this._pivotMarker = null
      }
      if (this._pivotAxes) {
        this._scene.remove(this._pivotAxes)
        this._pivotAxes.geometry?.dispose?.()
        this._pivotAxes = null
      }
    },

    togglePivot() {
      if (!this._pivotMarker || !this._pivotAxes) return
      this._pivotMarker.visible = this.showPivot
      this._pivotAxes.visible = this.showPivot
    },

    toggleAxes() { if (this._axes) this._axes.visible = this.showAxes },
    toggleGrid() { if (this._grid) this._grid.visible = this.showGrid },
  },
}
</script>

<style scoped>
.gsplat-root {
  position: absolute;
  width: 100% !important;
  height: 100% !important;
  display: block;
  background: #0b0d12;
  overflow: hidden;
}

.gsplat-root > canvas {
  position: absolute;
  inset: 0;
  width: 100% !important;
  height: 100% !important;
  display: block;
}

/* HUD y controles */
.hud {
  position: absolute;
  top: 10px;
  left: 10px;
  min-width: 260px;
  max-width: 42vw;
  background: rgba(10, 12, 18, 0.7);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 10px 12px;
  color: #e9eefc;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;
  font-size: 12px;
  line-height: 1.3;
  user-select: none;
}

.row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.row label { opacity: 0.9; min-width: 64px; }
.row input[type="range"] { flex: 1; }

.pill {
  padding: 2px 6px;
  background: rgba(255,255,255,0.1);
  border-radius: 999px;
  font-variant-numeric: tabular-nums;
}

.toggles { display: flex; gap: 16px; }
.coords { margin-top: 6px; opacity: 0.95; font-variant-numeric: tabular-nums; }
</style>

