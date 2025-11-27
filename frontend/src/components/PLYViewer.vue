<template>
  <div ref="container" class="ply-root">
    <!-- HUD / Controles -->
    <div class="hud">
      <!-- Solo desktop: velocidad para movimiento con teclado -->
      <div class="row" v-if="!isMobile">
        <label>Velocidad (teclado)</label>
        <input
          type="range"
          min="0.2"
          max="20"
          step="0.1"
          v-model.number="moveSpeed"
        />
        <span class="pill">{{ moveSpeed.toFixed(1) }} u/s</span>
      </div>

      <div class="row toggles">
        <!-- Pivot solo en desktop -->
        <label v-if="!isMobile">
          <input type="checkbox" v-model="showPivot" @change="togglePivot" />
          Pivot
        </label>

        <label>
          <input type="checkbox" v-model="showAxes" @change="toggleAxes" />
          Ejes
        </label>
        <label>
          <input type="checkbox" v-model="showGrid" @change="toggleGrid" />
          Grilla
        </label>

        <!-- Encuadrar siempre disponible -->
        <button type="button" class="hud-btn" @click="frameView">
          Encuadrar
        </button>
      </div>

      <div class="coords">
        <div>
          <strong>Cam:</strong>
          x {{ camPos.x.toFixed(3) }} |
          y {{ camPos.y.toFixed(3) }} |
          z {{ camPos.z.toFixed(3) }}
        </div>
        <div>
          <strong>Target:</strong>
          x {{ camTarget.x.toFixed(3) }} |
          y {{ camTarget.y.toFixed(3) }} |
          z {{ camTarget.z.toFixed(3) }}
        </div>

        <!-- Tips según dispositivo -->
        <div class="tips" v-if="isMobile">
          <strong>Gestos:</strong>
          Arrastra = orbitar · Pellizca = zoom · Dos dedos = desplazar ·
          “Encuadrar” = centrar vista
        </div>
        <div class="tips" v-else>
          <strong>Tips:</strong>
          Doble clic = fijar pivote · F = encuadrar · Shift = x3 · Ctrl = x0.25
        </div>
      </div>
    </div>
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
  const parts = text
    .split(boundary)
    .filter(p => p.trim() && !p.trim().startsWith('--'))

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
      _worldBBox: new THREE.Box3(), // Nube de puntos real
      _clampBBox: new THREE.Box3(), // BBox extendido para limitar el movimiento

      // HUD / UI
      moveSpeed: 3.5, // solo teclado
      showPivot: true,
      showAxes: true,
      showGrid: true,

      camPos: new THREE.Vector3(),
      camTarget: new THREE.Vector3(),

      // Pivot viz
      _pivotMarker: null,
      _pivotAxes: null,

      // Metadata para feed
      pendingMetadata: null,

      // WebSocket
      wsConnected: false,
      wsHandle: null,

      // Modo mobile
      isMobile: false,
    }
  },

  mounted() {
    const container = this.$refs.container

    // Detectar mobile / touch
    this.isMobile =
      (window.matchMedia &&
        window.matchMedia('(max-width: 768px)').matches) ||
      'ontouchstart' in window

    if (this.isMobile) {
      this.showPivot = false
    }

    // --- Escena / Cámara / Render ---
    const scene = (this._scene = new THREE.Scene())
    scene.background = new THREE.Color(0x0b0d12)

    const camera = (this._camera = new THREE.PerspectiveCamera(
      60,
      Math.max(container.clientWidth, 1) /
        Math.max(container.clientHeight, 1),
      0.001,
      1e6
    ))
    camera.position.set(0, 0, 3)

    const renderer = (this._renderer = new THREE.WebGLRenderer({
      antialias: true,
    }))
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(
      container.clientWidth || 1,
      container.clientHeight || 1,
      true
    )
    container.appendChild(renderer.domElement)

    // --- Luces ---
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6))
    const dir = new THREE.DirectionalLight(0xffffff, 0.9)
    dir.position.set(5, 10, 7.5)
    scene.add(dir)

    // --- Controles Arcball ---
    const orbit = (this._orbit = new ArcballControls(
      camera,
      renderer.domElement,
      scene
    ))
    orbit.setGizmosVisible(false)
    orbit.enableAnimations = false // sin inercia: no órbita infinita
    orbit.dampingFactor = 0
    if ('minDistance' in orbit) orbit.minDistance = 0.01
    if ('maxDistance' in orbit) orbit.maxDistance = 1e6

    // --- Helpers ---
    this._createHelpers(1)

    // --- Pivot viz solo desktop ---
    if (!this.isMobile) {
      this._createPivotViz()
      this._updatePivotViz()
    }

    // --- WASD + modificadores (desktop) ---
    const keys = {
      KeyW: false,
      KeyA: false,
      KeyS: false,
      KeyD: false,
      Space: false,
      ShiftLeft: false,
      ShiftRight: false,
      ControlLeft: false,
      ControlRight: false,
      KeyQ: false,
      KeyE: false,
    }

    const onKeyDown = e => {
      if (e.code in keys) keys[e.code] = true
      if (e.code === 'Equal') this.moveSpeed = Math.min(this.moveSpeed + 0.5, 50)
      if (e.code === 'Minus') this.moveSpeed = Math.max(this.moveSpeed - 0.5, 0.1)
      if (e.code === 'KeyF') this._frameToBBox()
    }
    const onKeyUp = e => {
      if (e.code in keys) keys[e.code] = false
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)

    // --- Doble clic (desktop) para fijar pivote ---
    const onDblClick = ev => {
      if (this.isMobile) return
      this._setPivotFromClientCoords(ev.clientX, ev.clientY)
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

      // Movimiento por teclado (desktop)
      let speed = this.moveSpeed
      const fast = keys.ShiftLeft || keys.ShiftRight
      const precise = keys.ControlLeft || keys.ControlRight
      if (fast) speed *= 3.0
      if (precise) speed *= 0.25

      if (
        keys.KeyW ||
        keys.KeyS ||
        keys.KeyA ||
        keys.KeyD ||
        keys.Space ||
        keys.ShiftLeft ||
        keys.ShiftRight ||
        keys.KeyQ ||
        keys.KeyE
      ) {
        camera.matrixAutoUpdate = true
        camera.getWorldDirection(tmpFwd).normalize()
        tmpRight.copy(tmpFwd).cross(upVec).normalize()

        const vel = speed * dt
        const delta = new THREE.Vector3()

        if (keys.KeyW) delta.addScaledVector(tmpFwd, vel)
        if (keys.KeyS) delta.addScaledVector(tmpFwd, -vel)
        if (keys.KeyA) delta.addScaledVector(tmpRight, -vel)
        if (keys.KeyD) delta.addScaledVector(tmpRight, vel)
        if (keys.Space || keys.KeyE) delta.addScaledVector(upVec, vel)
        if (keys.ShiftLeft || keys.ShiftRight || keys.KeyQ)
          delta.addScaledVector(upVec, -vel)

        camera.position.add(delta)
        orbit.target.add(delta)
        camera.updateProjectionMatrix()
      }

      // Actualizar controles
      orbit.update()

      // Limitar target y cámara al bbox extendido
      if (!this._clampBBox.isEmpty()) {
        const clampedTarget = this._clampBBox.clampPoint(
          orbit.target,
          new THREE.Vector3()
        )
        orbit.target.copy(clampedTarget)

        const clampedCam = this._clampBBox.clampPoint(
          camera.position,
          new THREE.Vector3()
        )
        camera.position.copy(clampedCam)
      }

      // HUD
      this.camPos.copy(camera.position)
      this.camTarget.copy(orbit.target)

      // Pivot viz (solo si existe)
      this._updatePivotViz()

      this._camera.updateProjectionMatrix()
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

      // chequear cambio mobile/desktop
      const wasMobile = this.isMobile
      this.isMobile =
        (window.matchMedia &&
          window.matchMedia('(max-width: 768px)').matches) ||
        'ontouchstart' in window

      if (!this.isMobile && wasMobile && !this._pivotMarker) {
        // pasamos a desktop -> crear pivot viz
        this.showPivot = true
        this._createPivotViz()
        this._updatePivotViz()
      } else if (this.isMobile && !wasMobile && this._pivotMarker) {
        // pasamos a mobile -> ocultar pivot viz
        this.showPivot = false
        this._disposePivotViz()
      }
    }
    window.addEventListener('resize', onResize)
    onResize()

    // --- Cargar dato por ID desde la ruta ---
    const id = this.$route?.query?.id ?? this.$route?.params?.id
    if (id) {
      this.loadById(String(id)).catch(console.error)
      this.initWebSocket(String(id))
    } else {
      console.warn('[PLYViewer] No id in route (query/params).')
    }

    // Watch cambios de id
    this.$watch(
      () => this.$route?.fullPath,
      () => {
        const newId = this.$route?.query?.id ?? this.$route?.params?.id
        if (newId) {
          this.loadById(String(newId)).catch(console.error)
          this.initWebSocket(String(newId))
        }
      }
    )

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

      if (this.wsHandle) {
        this.wsHandle.close()
        this.wsHandle = null
      }

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  },

  beforeUnmount() {
    this._cleanup && this._cleanup()
  },

  methods: {
    frameView() {
      this._frameToBBox()
    },

    _setPivotFromClientCoords(clientX, clientY) {
      if (!this._renderer || !this._camera || !this._pickables.length) return
      const rect = this._renderer.domElement.getBoundingClientRect()
      const ndc = new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -(((clientY - rect.top) / rect.height) * 2 - 1)
      )
      const raycaster = new THREE.Raycaster()
      raycaster.setFromCamera(ndc, this._camera)
      const intersects = raycaster.intersectObjects(this._pickables, true)
      if (intersects.length) {
        const p = intersects[0].point
        this._setPivot(p)
        this._updatePivotViz()
      }
    },

    async loadById(id, preserveCamera = false) {
      // Guardar cámara si queremos preservar
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
            if (
              Array.isArray(cameraPose) &&
              cameraPose.length === 4 &&
              cameraPose.every(r => Array.isArray(r) && r.length === 4)
            ) {
              cameraCv = cameraPose
            }
          } catch (e) {
            console.warn('[PLYViewer] Failed to parse camera_pose from multipart:', e)
          }
        }
      } else if (ctype.includes('application/json')) {
        const json = await res.json()
        if (typeof json.pointcloud === 'string') {
          if (json.pointcloud.startsWith('http') || json.pointcloud.startsWith('/')) {
            plyUrl = json.pointcloud
          } else if (
            json.pointcloud.startsWith('data:') ||
            /^[A-Za-z0-9+/]+=*$/.test(json.pointcloud)
          ) {
            plyBuffer = base64ToArrayBuffer(json.pointcloud)
          }
        } else {
          console.warn('[PLYViewer] pointcloud no es string; espera URL o base64.')
        }
        if (
          Array.isArray(json.camera_pose) &&
          json.camera_pose.length === 4 &&
          json.camera_pose.every(r => Array.isArray(r) && r.length === 4)
        ) {
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
        const mat = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        })
        object = new THREE.Points(geometry, mat)
        object.raycast = THREE.Points.prototype.raycast
      } else {
        const mat = new THREE.MeshStandardMaterial({
          color: 0xaaaaaa,
          flatShading: false,
        })
        object = new THREE.Mesh(geometry, mat)
      }

      this._removeCurrentObject()
      this._scene.add(object)
      this._pickables = [object]
      this._currentObject = object

      // BBoxes
      this._worldBBox.makeEmpty()
      if (object.geometry?.boundingBox) {
        const bb = object.geometry.boundingBox.clone()
        this._worldBBox.union(bb)
      }
      this._updateClampBBox() // recalcular bbox extendido

      // Helpers al tamaño
      this._recalibrateHelpers(maxDim)

      // near/far y distancias
      this._camera.near = Math.max(maxDim / 1000, 0.001)
      this._camera.far = Math.max(maxDim * 100, 10)
      this._camera.updateProjectionMatrix()
      if ('minDistance' in this._orbit) {
        this._orbit.minDistance = Math.max(maxDim * 0.02, 0.001)
      }
      if ('maxDistance' in this._orbit) {
        this._orbit.maxDistance = Math.max(maxDim * 20, 10)
      }

      // Cámara
      if (preserveCamera && savedCameraPos && savedCameraTarget) {
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
            if (
              Array.isArray(cameraPose) &&
              cameraPose.length === 4 &&
              cameraPose.every(r => Array.isArray(r) && r.length === 4)
            ) {
              cameraCv = cameraPose
            }
          } catch (e) {
            console.warn('[PLYViewer] Failed to parse camera_pose:', e)
          }
        }
      } else {
        plyBuffer = arrayBuffer
      }

      if (!plyBuffer) {
        console.error('[PLYViewer] No pointcloud data in response')
        return
      }

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
        const mat = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        })
        newObject = new THREE.Points(geometry, mat)
        newObject.raycast = THREE.Points.prototype.raycast
      } else {
        const mat = new THREE.MeshStandardMaterial({
          color: 0xaaaaaa,
          flatShading: false,
        })
        newObject = new THREE.Mesh(geometry, mat)
      }

      this._scene.add(newObject)
      this._pickables.push(newObject)
      this._currentObject = newObject

      // BBox combinado
      this._worldBBox.makeEmpty()
      for (const obj of this._pickables) {
        obj.geometry?.computeBoundingBox?.()
        if (obj.geometry?.boundingBox) {
          const bb = obj.geometry.boundingBox.clone()
          this._worldBBox.union(bb)
        }
      }
      this._updateClampBBox()

      const sizeAll = this._worldBBox.getSize(new THREE.Vector3())
      const maxDimAll = Math.max(sizeAll.x, sizeAll.y, sizeAll.z) || 1
      this._camera.near = Math.max(maxDimAll / 1000, 0.001)
      this._camera.far = Math.max(maxDimAll * 100, 10)
      this._camera.updateProjectionMatrix()
      if ('minDistance' in this._orbit) {
        this._orbit.minDistance = Math.max(maxDimAll * 0.02, 0.001)
      }
      if ('maxDistance' in this._orbit) {
        this._orbit.maxDistance = Math.max(maxDimAll * 20, 10)
      }

      if (cameraCv) this._applyOpenCVCameraToWorld(cameraCv, true)
      this._updatePivotViz()
    },

    async reloadPointCloud(id) {
      await this._fadeOutCurrentObjects()
      await this.loadById(id, true)
      this.$emit('loadComplete')
    },

    async _fadeOutCurrentObjects() {
      if (this._pickables.length === 0) return

      const objectsToFade = [...this._pickables]
      this._fadingObjects.push(...objectsToFade)

      this._pickables = []
      this._currentObject = null

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

      const fadeDuration = 1200
      const startTime = Date.now()

      return new Promise(resolve => {
        const animateFade = () => {
          const elapsed = Date.now() - startTime
          const progress = Math.min(elapsed / fadeDuration, 1.0)
          const opacity = 1.0 - progress

          for (const obj of objectsToFade) {
            const mat = obj.material
            if (mat) {
              if (Array.isArray(mat)) {
                mat.forEach(m => {
                  m.opacity = opacity
                })
              } else {
                mat.opacity = opacity
              }
            }
          }

          if (progress < 1.0) {
            requestAnimationFrame(animateFade)
          } else {
            for (const obj of objectsToFade) {
              this._scene.remove(obj)
              obj.geometry?.dispose?.()
              const m = obj.material
              if (Array.isArray(m)) m.forEach(mm => mm?.dispose?.())
              else m?.dispose?.()
            }
            this._fadingObjects = this._fadingObjects.filter(
              o => !objectsToFade.includes(o)
            )
            resolve()
          }
        }
        animateFade()
      })
    },

    _frameToBBox() {
      if (this._worldBBox.isEmpty()) return
      const center = this._worldBBox.getCenter(new THREE.Vector3())
      const size = this._worldBBox.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z) || 1

      const fov = (this._camera.fov * Math.PI) / 180
      const camDist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.35

      this._camera.matrixAutoUpdate = true
      this._camera.position.set(center.x, center.y, center.z + camDist)
      this._camera.lookAt(center)

      this._orbit.target.copy(center)
      this._camera.updateProjectionMatrix()

      if ('minDistance' in this._orbit) {
        this._orbit.minDistance = Math.max(maxDim * 0.02, 0.001)
      }
      if ('maxDistance' in this._orbit) {
        this._orbit.maxDistance = Math.max(maxDim * 20, 10)
      }

      this._updatePivotViz()
      this._orbit.update()
    },

    _applyOpenCVCameraToWorld(cv4x4, animate = false) {
      const a = cv4x4.flat()
      const Mcv = new THREE.Matrix4().set(
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7],
        a[8],
        a[9],
        a[10],
        a[11],
        a[12],
        a[13],
        a[14],
        a[15]
      )
      const C = new THREE.Matrix4().set(
        1,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        1
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
      const camToPivot = new THREE.Vector3().subVectors(
        this._camera.position,
        this._orbit.target
      )
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
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        const ctype = res.headers.get('content-type') || ''
        let plyBuffer = null
        let plyUrl = null

        if (ctype.includes('multipart/form-data')) {
          const buffer = await res.arrayBuffer()
          const parsed = parseMultipartResponse(buffer, ctype)
          if (parsed.pointcloud) plyBuffer = parsed.pointcloud
        } else if (ctype.includes('application/json')) {
          const json = await res.json()
          if (typeof json.pointcloud === 'string') {
            if (json.pointcloud.startsWith('http') || json.pointcloud.startsWith('/')) {
              plyUrl = json.pointcloud
            } else if (
              json.pointcloud.startsWith('data:') ||
              /^[A-Za-z0-9+/]+=*$/.test(json.pointcloud)
            ) {
              plyBuffer = base64ToArrayBuffer(json.pointcloud)
            }
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
          throw new Error('No data found')
        }

        geometry.computeVertexNormals?.()
        geometry.computeBoundingBox?.()

        const bbox = geometry.boundingBox
        let pointSize = 0.02
        if (bbox) {
          const size = bbox.getSize(new THREE.Vector3())
          const maxDim = Math.max(size.x, size.y, size.z) || 1
          pointSize = maxDim * 0.005
        }

        let newObject
        if (geometry.getAttribute && geometry.getAttribute('color')) {
          const mat = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true,
          })
          newObject = new THREE.Points(geometry, mat)
          newObject.raycast = THREE.Points.prototype.raycast
        } else {
          const mat = new THREE.MeshStandardMaterial({
            color: 0xaaaaaa,
            flatShading: false,
          })
          newObject = new THREE.Mesh(geometry, mat)
        }

        this._removeCurrentObject()
        this._scene.add(newObject)
        this._pickables = [newObject]
        this._currentObject = newObject

        this._worldBBox.makeEmpty()
        if (newObject.geometry?.boundingBox) {
          const bb = newObject.geometry.boundingBox.clone()
          this._worldBBox.union(bb)
        }
        this._updateClampBBox()

        if (this.pendingMetadata) {
          this.$emit('model-updated', this.pendingMetadata)
          this.pendingMetadata = null
        } else {
          this.$emit('model-updated', { user_id: 'System', image_id: 'Update' })
        }

        console.log('[PLYViewer] Smart reload complete')
      } catch (e) {
        console.error('[PLYViewer] Smart reload failed:', e)
      }
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
      if (this._axes) {
        this._scene.remove(this._axes)
        this._axes.geometry?.dispose?.()
        this._axes = null
      }
      if (this._grid) {
        this._scene.remove(this._grid)
        this._grid.geometry?.dispose?.()
        Array.isArray(this._grid.material)
          ? this._grid.material.forEach(m => m.dispose?.())
          : this._grid.material?.dispose?.()
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
        Array.isArray(this._grid.material)
          ? this._grid.material.forEach(m => m.dispose?.())
          : this._grid.material?.dispose?.()
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
      const maxDim = Math.max(size.x, size.y, size.z) || 1

      // FACTOR CONSTANTE: el usuario puede moverse dentro de un bbox
      // que extiende la nube original maxDim * 2 en todas las direcciones.
      const pad = maxDim * 2.0

      this._clampBBox.copy(this._worldBBox).expandByScalar(pad)
    },

    // ===== Pivot viz (solo desktop) =====
    _createPivotViz() {
      if (!this._scene || this.isMobile) return
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
      if (
        this.isMobile ||
        !this._pivotMarker ||
        !this._pivotAxes ||
        !this._orbit ||
        !this._camera
      )
        return
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

    toggleAxes() {
      if (this._axes) this._axes.visible = this.showAxes
    },

    toggleGrid() {
      if (this._grid) this._grid.visible = this.showGrid
    },

    // ===== WebSocket =====
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

      const { status, data, send, close, open } = useWebSocket(wsUrl, {
        autoReconnect: {
          retries: -1,
          delay: 1000,
          onFailed() {
            console.log(
              '[PLYViewer] Failed to connect WebSocket after retries'
            )
          },
        },
        heartbeat: {
          message: JSON.stringify({ type: 'ping' }),
          interval: 30000,
          pongTimeout: 10000,
        },
        onConnected: () => {
          console.log('[PLYViewer] WebSocket connected')
          this.wsConnected = true
        },
        onDisconnected: () => {
          console.log('[PLYViewer] WebSocket disconnected')
          this.wsConnected = false
        },
        onError: (ws, event) => {
          console.error('[PLYViewer] WebSocket error:', event)
        },
      })

      watch(data, newData => {
        if (!newData) return
        try {
          const msg = JSON.parse(newData)
          if (msg.type === 'pong') return

          if (msg.type === 'update' && msg.status === 'updated') {
            const receivedTime = Date.now() / 1000
            console.log(
              `[PLYViewer] Received update notification at ${receivedTime.toFixed(
                3
              )}:`,
              msg
            )

            if (msg.metadata) {
              this.pendingMetadata = msg.metadata
            }

            if (msg.timestamp) {
              const latency = receivedTime - msg.timestamp
              console.log(
                `[PLYViewer] Notification Latency: ${latency.toFixed(3)}s`
              )
            }
            this.smartReload(projectId)
          }
        } catch (e) {
          console.error('[PLYViewer] Error parsing WebSocket message:', e)
        }
      })

      this.wsHandle = { close, open, send, status, data }
    },
  },
}
</script>

<style scoped>
/* Bloquear selección de texto y callouts dentro del viewer (iOS / mobile) */
.ply-root,
.ply-root * {
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;

  -webkit-touch-callout: none;           /* sin menú de copiar/buscar */
  -webkit-tap-highlight-color: transparent;
}

/* Contenedor principal del canvas */
.ply-root {
  position: absolute;
  width: 100% !important;
  height: 100% !important;
  display: block;
  background: #0b0d12;
  overflow: hidden;
}

/* Canvas ocupa todo */
.ply-root > canvas {
  position: absolute;
  inset: 0;
  width: 100% !important;
  height: 100% !important;
  display: block;
}

/* HUD */
.hud {
  position: absolute;
  top: 10px;
  left: 10px;
  min-width: 260px;
  max-width: 42vw;
  background: rgba(10, 12, 18, 0.7);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 10px 12px;
  color: #e9eefc;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto,
    'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;
  font-size: 12px;
  line-height: 1.3;
}

/* Filas HUD */
.row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.row label {
  opacity: 0.9;
  min-width: 64px;
}

.row input[type='range'] {
  flex: 1;
}

.pill {
  padding: 2px 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 999px;
  font-variant-numeric: tabular-nums;
}

.toggles {
  display: flex;
  gap: 12px;
  align-items: center;
}

.coords {
  margin-top: 6px;
  opacity: 0.95;
  font-variant-numeric: tabular-nums;
}

.tips {
  opacity: 0.9;
  margin-top: 4px;
}

/* Botón Encuadrar */
.hud-btn {
  margin-left: auto;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.8);
  background: rgba(15, 23, 42, 0.9);
  color: #e5e7eb;
  font-size: 11px;
  cursor: pointer;
  transition:
    background 0.15s ease,
    border-color 0.15s ease,
    transform 0.15s ease;
}

.hud-btn:hover,
.hud-btn:focus-visible {
  border-color: rgba(129, 140, 248, 0.95);
  background: rgba(30, 64, 175, 0.95);
  transform: translateY(-0.5px);
}

/* Mobile tweaks */
@media (max-width: 768px) {
  .hud {
    top: 8px;
    left: 8px;
    right: 8px;
    min-width: auto;
    max-width: none;
    padding: 8px 10px;
    font-size: 11px;
  }

  .row {
    flex-wrap: wrap;
  }

  .toggles {
    flex-wrap: wrap;
    row-gap: 4px;
  }

  .hud-btn {
    font-size: 11px;
    padding: 4px 8px;
  }
}
</style>
