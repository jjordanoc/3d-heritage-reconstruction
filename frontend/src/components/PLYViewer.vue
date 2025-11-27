<template>
  <div ref="container" class="ply-root">
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

    <!-- Update Bubble REMOVED -->
  </div>
</template>

<script>
import * as THREE from 'three'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { useWebSocket } from '@vueuse/core'
import { watch } from 'vue'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')
const WS_API_BASE = (import.meta.env.VITE_WS_API_URL || '').replace(/\/+$/, '')


function base64ToArrayBuffer(b64) {
  const comma = b64.indexOf(',')
  const raw = comma >= 0 ? b64.slice(comma + 1) : b64
  const bin = atob(raw)
  const len = bin.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i)
  return bytes.buffer
}

function parseMultipartResponse(arrayBuffer, contentType) {
  const boundaryMatch = contentType.match(/boundary=([^;]+)/)
  if (!boundaryMatch) throw new Error('No boundary found in multipart content-type')
  const boundary = '--' + boundaryMatch[1]
  const decoder = new TextDecoder('utf-8')
  const text = decoder.decode(arrayBuffer)
  const parts = text.split(boundary).filter(p => p.trim() && !p.trim().startsWith('--'))
  const result = {}
  for (const part of parts) {
    const headerEndIndex = part.indexOf('\r\n\r\n')
    if (headerEndIndex === -1) continue
    const headers = part.substring(0, headerEndIndex)
    const body = part.substring(headerEndIndex + 4).replace(/\r\n$/, '')
    const nameMatch = headers.match(/name="([^"]+)"/)
    if (!nameMatch) continue
    const fieldName = nameMatch[1]
    if (fieldName === 'pointcloud') {
      const encoder = new TextEncoder()
      const partStart = text.indexOf(part)
      const bodyStartInText = partStart + headerEndIndex + 4
      const bodyStartBytes = encoder.encode(text.substring(0, bodyStartInText)).length
      let bodyEndBytes = arrayBuffer.byteLength
      const nextBoundaryInText = text.indexOf(boundary, bodyStartInText)
      if (nextBoundaryInText !== -1) {
        bodyEndBytes = encoder.encode(text.substring(0, nextBoundaryInText)).length - 2
      }
      result[fieldName] = arrayBuffer.slice(bodyStartBytes, bodyEndBytes)
    } else {
      result[fieldName] = body
    }
  }
  return result
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
      _fadingObjects: [],

      // Helpers
      _axes: null,
      _grid: null,

      // BBoxes
      _worldBBox: new THREE.Box3(),      // BBox combinado
      _clampBBox: new THREE.Box3(),      // BBox para clamping

      // UI state
      moveSpeed: 3.5,
      showPivot: true,                   // <--- por defecto ON
      showAxes: true,
      showGrid: true,

      // HUD coords
      camPos: new THREE.Vector3(),
      camTarget: new THREE.Vector3(),

      // Pivot viz
      _pivotMarker: null,
      _pivotAxes: null,
      
      // Metadata storage for sync
      pendingMetadata: null,

      // WebSocket
      // websocket: null, // Removed manual websocket
      wsConnected: false,
      // showUpdateBubble: false, // Removed internal bubble
      // updateMessage: '', // Removed internal bubble
      wsHandle: null, // Store the return from useWebSocket
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

    // --- Pivot visual (amarillo y gordito) ---
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
        this._setPivot(p)      // mueve el target ahí
        this._updatePivotViz() // <-- actualiza para que lo veas ya
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
    if (id) {
      this.loadById(String(id)).catch(console.error)
      this.initWebSocket(String(id))
    }
    else console.warn('[PLYViewer] No id in route (query/params).')

    // Watch cambios de id
    this.$watch(() => this.$route?.fullPath, () => {
      const newId = this.$route?.query?.id ?? this.$route?.params?.id
      if (newId) {
        this.loadById(String(newId)).catch(console.error)
        this.initWebSocket(String(newId))
      }
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
      
      // Clean up useWebSocket
      if (this.wsHandle) {
        this.wsHandle.close()
        this.wsHandle = null
      }
      
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
      const url = `${base}/pointcloud/${encodeURIComponent(id)}/latest`
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status} al pedir ${url}`)

      const ctype = res.headers.get('content-type') || ''
      let cameraCv = null
      let plyBuffer = null
      let plyUrl = null

      if (ctype.includes('multipart/form-data')) {
        const buffer = await res.arrayBuffer()
        const parsed = parseMultipartResponse(buffer, ctype)
        if (parsed.pointcloud) plyBuffer = parsed.pointcloud
        if (parsed.camera_pose) {
          try {
            const cameraPose = JSON.parse(parsed.camera_pose)
            if (Array.isArray(cameraPose) && cameraPose.length === 4 && cameraPose.every(r => Array.isArray(r) && r.length === 4)) {
              cameraCv = cameraPose
            }
          } catch (e) { console.warn('[PLYViewer] Failed to parse camera_pose from multipart:', e) }
        }
      } else if (ctype.includes('application/json')) {
        const json = await res.json()
        if (typeof json.pointcloud === 'string') {
          if (json.pointcloud.startsWith('http') || json.pointcloud.startsWith('/')) plyUrl = json.pointcloud
          else if (json.pointcloud.startsWith('data:') || /^[A-Za-z0-9+/]+=*$/.test(json.pointcloud)) plyBuffer = base64ToArrayBuffer(json.pointcloud)
        } else {
          console.warn('[PLYViewer] pointcloud no es string; espera URL o base64.')
        }
        if (Array.isArray(json.camera_pose) && json.camera_pose.length === 4 && json.camera_pose.every(r => Array.isArray(r) && r.length === 4)) {
          cameraCv = json.camera_pose
        }
      } else {
        plyBuffer = await res.arrayBuffer()
      }

      const loader = new PLYLoader()
      let geometry
      if (plyUrl) {
        geometry = await new Promise((resolve, reject) => {
          loader.load(plyUrl, g => resolve(g), undefined, err => reject(err))
        })
      } else if (plyBuffer) {
        geometry = loader.parse(plyBuffer)
      } else {
        throw new Error('[PLYViewer] No se pudo obtener pointcloud (ni URL ni buffer)')
      }

      geometry.computeVertexNormals?.()
      geometry.computeBoundingBox?.()

      const bbox = geometry.boundingBox
      let center = new THREE.Vector3(0, 0, 0)
      let maxDim = 1
      let pointSize = 0.02
      if (bbox) {
        center = bbox.getCenter(new THREE.Vector3())
        const size = bbox.getSize(new THREE.Vector3())
        maxDim = Math.max(size.x, size.y, size.z) || 1
        pointSize = maxDim * 0.005
      }

      let object
      if (geometry.getAttribute && geometry.getAttribute('color')) {
        const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true })
        object = new THREE.Points(geometry, mat)
        object.raycast = THREE.Points.prototype.raycast
      } else {
        const mat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false })
        object = new THREE.Mesh(geometry, mat)
      }

      this._removeCurrentObject()
      this._scene.add(object)
      this._pickables = [object]
      this._currentObject = object

      // BBoxes globales
      this._worldBBox.makeEmpty()
      if (object.geometry?.boundingBox) this._worldBBox.union(object.geometry.boundingBox)
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
      } else if (cameraCv) {
        this._applyOpenCVCameraToWorld(cameraCv, true)
      } else {
        this._frameToBBox()
      }
      this._updatePivotViz()
    },

    async addIncrementalPoints(arrayBuffer, contentType) {
      let plyBuffer = null
      let cameraCv = null

      if (contentType.includes('multipart/form-data')) {
        const parsed = parseMultipartResponse(arrayBuffer, contentType)
        if (parsed.pointcloud) plyBuffer = parsed.pointcloud
        if (parsed.camera_pose) {
          try {
            const cameraPose = JSON.parse(parsed.camera_pose)
            if (Array.isArray(cameraPose) && cameraPose.length === 4 && cameraPose.every(r => Array.isArray(r) && r.length === 4)) {
              cameraCv = cameraPose
            }
          } catch (e) { console.warn('[PLYViewer] Failed to parse camera_pose:', e) }
        }
      } else {
        plyBuffer = arrayBuffer
      }
      if (!plyBuffer) { console.error('[PLYViewer] No pointcloud data in response'); return }

      const loader = new PLYLoader()
      const geometry = loader.parse(plyBuffer)
      geometry.computeVertexNormals?.()
      geometry.computeBoundingBox?.()

      const bbox = geometry.boundingBox
      let pointSize = 0.02
      let maxDimNew = 1
      if (bbox) {
        const size = bbox.getSize(new THREE.Vector3())
        maxDimNew = Math.max(size.x, size.y, size.z) || 1
        pointSize = maxDimNew * 0.005
      }

      let newObject
      if (geometry.getAttribute && geometry.getAttribute('color')) {
        const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true })
        newObject = new THREE.Points(geometry, mat)
        newObject.raycast = THREE.Points.prototype.raycast
      } else {
        const mat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false })
        newObject = new THREE.Mesh(geometry, mat)
      }

      this._scene.add(newObject)
      this._pickables.push(newObject)
      this._currentObject = newObject

      // BBox combinado
      for (const obj of this._pickables) {
        obj.geometry?.computeBoundingBox?.()
        if (obj.geometry?.boundingBox) this._worldBBox.union(obj.geometry.boundingBox)
      }
      this._updateClampBBox()

      // near/far y límites de distancia
      const sizeAll = this._worldBBox.getSize(new THREE.Vector3())
      const maxDimAll = Math.max(sizeAll.x, sizeAll.y, sizeAll.z) || 1
      this._camera.near = Math.max(maxDimAll / 1000, 0.001)
      this._camera.far  = Math.max(maxDimAll * 100, 10)
      this._camera.updateProjectionMatrix()
      if ('minDistance' in this._orbit) this._orbit.minDistance = Math.max(maxDimAll * 0.02, 0.001)
      if ('maxDistance' in this._orbit) this._orbit.maxDistance = Math.max(maxDimAll * 20, 10)

      if (cameraCv) this._applyOpenCVCameraToWorld(cameraCv, true)
      this._updatePivotViz()
    },

    async reloadPointCloud(id) {
      // Fade out current objects, then reload
      await this._fadeOutCurrentObjects()
      await this.loadById(id, true) // preserveCamera = true
      this.$emit('loadComplete')
    },

    async _fadeOutCurrentObjects() {
      if (this._pickables.length === 0) return

      // Move current objects to fading array
      const objectsToFade = [...this._pickables]
      this._fadingObjects.push(...objectsToFade)
      
      // Clear current pickables
      this._pickables = []
      this._currentObject = null

      // Enable transparency on materials
      for (const obj of objectsToFade) {
        const mat = obj.material
        if (mat) {
          if (Array.isArray(mat)) {
            mat.forEach(m => {
              m.transparent = true
              m.opacity = 1.0
            })
          } else {
            mat.transparent = true
            mat.opacity = 1.0
          }
        }
      }

      // Animate fade out
      const fadeDuration = 1200 // 1.2 seconds
      const startTime = Date.now()

      return new Promise((resolve) => {
        const animateFade = () => {
          const elapsed = Date.now() - startTime
          const progress = Math.min(elapsed / fadeDuration, 1.0)
          const opacity = 1.0 - progress

          for (const obj of objectsToFade) {
            const mat = obj.material
            if (mat) {
              if (Array.isArray(mat)) {
                mat.forEach(m => { m.opacity = opacity })
              } else {
                mat.opacity = opacity
              }
            }
          }

          if (progress < 1.0) {
            requestAnimationFrame(animateFade)
          } else {
            // Fade complete, dispose objects
            for (const obj of objectsToFade) {
              this._scene.remove(obj)
              obj.geometry?.dispose?.()
              const m = obj.material
              if (Array.isArray(m)) m.forEach(mm => mm?.dispose?.())
              else m?.dispose?.()
            }
            // Remove from fading array
            this._fadingObjects = this._fadingObjects.filter(o => !objectsToFade.includes(o))
            resolve()
          }
        }
        animateFade()
      })
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

    _applyOpenCVCameraToWorld(cv4x4, animate = false) {
      const a = cv4x4.flat()
      const Mcv = new THREE.Matrix4().set(
        a[0],  a[1],  a[2],  a[3],
        a[4],  a[5],  a[6],  a[7],
        a[8],  a[9],  a[10], a[11],
        a[12], a[13], a[14], a[15]
      )
      const C = new THREE.Matrix4().set(
        1,  0,  0, 0,
        0, -1,  0, 0,
        0,  0, -1, 0,
        0,  0,  0, 1
      )
      const Mthree = new THREE.Matrix4().copy(C).multiply(Mcv).multiply(C)

      const newPosition = new THREE.Vector3().setFromMatrixPosition(Mthree)
      const rot = new THREE.Matrix4().extractRotation(Mthree)
      const forward = new THREE.Vector3(0, 0, -1).applyMatrix4(rot).normalize()
      const newTarget = new THREE.Vector3().copy(newPosition).add(forward)

      if (animate) {
        this._camera.matrixAutoUpdate = true
        const startPosition = this._camera.position.clone()
        const startTarget = this._orbit.target.clone()
        const duration = 1000
        const startTime = Date.now()
        const animateCamera = () => {
          const elapsed = Date.now() - startTime
          const t = Math.min(elapsed / duration, 1)
          const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
          this._camera.position.lerpVectors(startPosition, newPosition, eased)
          this._orbit.target.lerpVectors(startTarget, newTarget, eased)
          this._camera.lookAt(this._orbit.target)
          this._camera.updateProjectionMatrix()
          this._orbit.update()
          if (t < 1) requestAnimationFrame(animateCamera)
          else {
            this._camera.matrixAutoUpdate = false
            this._camera.matrixWorld.copy(Mthree)
            this._camera.matrixWorldNeedsUpdate = true
            this._orbit.update()
            this._updatePivotViz()
          }
        }
        animateCamera()
      } else {
        this._camera.matrixAutoUpdate = false
        this._camera.matrixWorld.copy(Mthree)
        this._camera.matrixWorldNeedsUpdate = true
        this._camera.position.copy(newPosition)
        this._orbit.target.copy(newTarget)
        this._camera.lookAt(newTarget)
        this._camera.updateProjectionMatrix()
        this._orbit.update()
        this._updatePivotViz()
      }
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

    async smartReload(id) {
      console.log('[PLYViewer] Starting smart reload...')
      const base = API_BASE
      const url = `${base}/pointcloud/${encodeURIComponent(id)}/latest`
      
      try {
        // 1. Fetch in background (no visual change yet)
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        
        const ctype = res.headers.get('content-type') || ''
        let plyBuffer = null
        let plyUrl = null
        
        // Parse response similar to loadById
        if (ctype.includes('multipart/form-data')) {
          const buffer = await res.arrayBuffer()
          const parsed = parseMultipartResponse(buffer, ctype)
          if (parsed.pointcloud) plyBuffer = parsed.pointcloud
        } else if (ctype.includes('application/json')) {
          const json = await res.json()
          if (typeof json.pointcloud === 'string') {
             if (json.pointcloud.startsWith('http') || json.pointcloud.startsWith('/')) plyUrl = json.pointcloud
             else if (json.pointcloud.startsWith('data:') || /^[A-Za-z0-9+/]+=*$/.test(json.pointcloud)) plyBuffer = base64ToArrayBuffer(json.pointcloud)
          }
        } else {
          plyBuffer = await res.arrayBuffer()
        }

        // 2. Parse geometry in memory
        const loader = new PLYLoader()
        let geometry
        if (plyUrl) {
          geometry = await new Promise((resolve, reject) => {
            loader.load(plyUrl, g => resolve(g), undefined, err => reject(err))
          })
        } else if (plyBuffer) {
          geometry = loader.parse(plyBuffer)
        } else {
          throw new Error('No data found')
        }

        geometry.computeVertexNormals?.()
        geometry.computeBoundingBox?.()

        // Calculate size for material
        const bbox = geometry.boundingBox
        let pointSize = 0.02
        if (bbox) {
          const size = bbox.getSize(new THREE.Vector3())
          const maxDim = Math.max(size.x, size.y, size.z) || 1
          pointSize = maxDim * 0.005
        }

        // Create new Object3D
        let newObject
        if (geometry.getAttribute && geometry.getAttribute('color')) {
          const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true })
          newObject = new THREE.Points(geometry, mat)
          newObject.raycast = THREE.Points.prototype.raycast
        } else {
          const mat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false })
          newObject = new THREE.Mesh(geometry, mat)
        }

        // 3. Hot Swap (Instant visual update)
        // Remove old objects
        this._removeCurrentObject()
        
        // Add new object
        this._scene.add(newObject)
        this._pickables = [newObject]
        this._currentObject = newObject
        
        // Update bounding boxes
        this._worldBBox.makeEmpty()
        if (newObject.geometry?.boundingBox) this._worldBBox.union(newObject.geometry.boundingBox)
        this._updateClampBBox()

        // Notify UI via event
        // this.showUpdateNotification("Modelo actualizado") // Internal bubble removed
        if (this.pendingMetadata) {
           this.$emit('model-updated', this.pendingMetadata)
           this.pendingMetadata = null
        } else {
           // Fallback if no metadata was stored (e.g. initial load or manual reload)
           this.$emit('model-updated', { user_id: 'System', image_id: 'Update' })
        }
        
        console.log('[PLYViewer] Smart reload complete')

      } catch (e) {
        console.error('[PLYViewer] Smart reload failed:', e)
      }
    },

    showUpdateNotification(msg) {
      this.updateMessage = msg
      this.showUpdateBubble = true
      setTimeout(() => {
        this.showUpdateBubble = false
      }, 3000)
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
      // esfera amarilla bien visible (gordita)
      const geo = new THREE.SphereGeometry(0.05, 24, 16) // radio grande
      const mat = new THREE.MeshBasicMaterial({ color: 0xffff00 }) // amarillo brillante
      this._pivotMarker = new THREE.Mesh(geo, mat)
      this._pivotMarker.visible = this.showPivot
      this._scene.add(this._pivotMarker)

      // mini ejes en el pivote
      this._pivotAxes = new THREE.AxesHelper(0.15)
      this._pivotAxes.visible = this.showPivot
      this._scene.add(this._pivotAxes)
    },

    _updatePivotViz() {
      if (!this._pivotMarker || !this._pivotAxes || !this._orbit || !this._camera) return
      const t = this._orbit.target
      // posicionar
      this._pivotMarker.position.copy(t)
      this._pivotAxes.position.copy(t)
      // mantener tamaño suficiente en pantalla (ligera adaptación con distancia)
      const dist = this._camera.position.distanceTo(t) || 1
      const k = THREE.MathUtils.clamp(dist * 0.02, 0.5, 4.0) // factor de escala
      this._pivotMarker.scale.setScalar(k)                   // esfera gordita
      this._pivotAxes.scale.setScalar(k * 1.2)               // ejes un poco mayores
      // toggle
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

    // ===== WebSocket Integration (using @vueuse/core) =====
    initWebSocket(projectId) {
      if (this.wsHandle) {
        this.wsHandle.close()
        this.wsHandle = null
      }

      let wsUrl = WS_API_BASE.replace(/^http/, 'ws')
      if (!wsUrl || wsUrl.startsWith('/')) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        wsUrl = `${protocol}//${host}${API_BASE}`
      }
      
      wsUrl = `${wsUrl}/ws/${projectId}`
      
      console.log('[PLYViewer] Connecting WebSocket:', wsUrl)
      
      // Initialize useWebSocket
      const { status, data, send, close, open } = useWebSocket(wsUrl, {
        autoReconnect: {
          retries: -1, // Infinite retries
          delay: 1000,
          onFailed() {
            console.log('[PLYViewer] Failed to connect WebSocket after retries')
          },
        },
        heartbeat: {
          message: JSON.stringify({ type: 'ping' }),
          interval: 30000, // 30 seconds
          pongTimeout: 10000,
        },
        onConnected: (ws) => {
          console.log('[PLYViewer] WebSocket connected')
          this.wsConnected = true
        },
        onDisconnected: (ws, event) => {
          console.log('[PLYViewer] WebSocket disconnected')
          this.wsConnected = false
        },
        onError: (ws, event) => {
          console.error('[PLYViewer] WebSocket error:', event)
        },
      })

      // Watch for data changes
      watch(data, (newData) => {
        if (!newData) return
        try {
          const msg = JSON.parse(newData)
          // Handle pong if needed, but heartbeat handles it mostly
          if (msg.type === 'pong') return

          if (msg.type === 'update' && msg.status === 'updated') {
            const receivedTime = Date.now() / 1000
            console.log(`[PLYViewer] Received update notification at ${receivedTime.toFixed(3)}:`, msg)
            
            // Store metadata to emit later upon successful reload
            if (msg.metadata) {
              this.pendingMetadata = msg.metadata
            }

            if (msg.timestamp) {
               const latency = receivedTime - msg.timestamp
               console.log(`[PLYViewer] Notification Latency: ${latency.toFixed(3)}s`)
            }
            // Trigger smart reload
            this.smartReload(projectId)
          }
        } catch (e) {
          console.error('[PLYViewer] Error parsing WebSocket message:', e)
        }
      })

      // Store handle to close later
      this.wsHandle = { close, open, send, status, data }
    },
  },
}
</script>

<style scoped>
.ply-root {
  position: absolute;
  width: 100% !important;
  height: 100% !important;
  display: block;
  background: #0b0d12;
  overflow: hidden;
}

/* Asegura que el canvas calce exacto en el contenedor */
.ply-root > canvas {
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

/* Update Bubble REMOVED */

/* Vue Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translate(-50%, 10px);
}
</style>
