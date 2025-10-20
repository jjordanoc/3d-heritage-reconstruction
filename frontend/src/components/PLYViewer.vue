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

// Utilidad: parsear respuesta multipart/form-data
function parseMultipartResponse(arrayBuffer, contentType) {
  // Extraer boundary del content-type
  const boundaryMatch = contentType.match(/boundary=([^;]+)/)
  if (!boundaryMatch) {
    throw new Error('No boundary found in multipart content-type')
  }
  const boundary = '--' + boundaryMatch[1]
  
  // Convertir arrayBuffer a string para procesarlo
  const decoder = new TextDecoder('utf-8')
  const text = decoder.decode(arrayBuffer)
  
  // Dividir por boundary
  const parts = text.split(boundary).filter(p => p.trim() && !p.trim().startsWith('--'))
  
  const result = {}
  
  for (const part of parts) {
    // Separar headers del body
    const headerEndIndex = part.indexOf('\r\n\r\n')
    if (headerEndIndex === -1) continue
    
    const headers = part.substring(0, headerEndIndex)
    const body = part.substring(headerEndIndex + 4).replace(/\r\n$/, '')
    
    // Extraer nombre del campo
    const nameMatch = headers.match(/name="([^"]+)"/)
    if (!nameMatch) continue
    
    const fieldName = nameMatch[1]
    
    // Si es pointcloud (binario), necesitamos el buffer original
    if (fieldName === 'pointcloud') {
      // Encontrar el inicio del body en el buffer original
      const encoder = new TextEncoder()
      const headerBytes = encoder.encode(part.substring(0, headerEndIndex + 4))
      
      // Buscar donde empieza este part en el buffer original
      const partStart = text.indexOf(part)
      const bodyStartInText = partStart + headerEndIndex + 4
      
      // Calcular offset en bytes (aproximado, asumiendo UTF-8)
      const bodyStartBytes = encoder.encode(text.substring(0, bodyStartInText)).length
      
      // Encontrar el final del body (antes del siguiente boundary o final)
      let bodyEndBytes = arrayBuffer.byteLength
      const nextBoundaryInText = text.indexOf(boundary, bodyStartInText)
      if (nextBoundaryInText !== -1) {
        bodyEndBytes = encoder.encode(text.substring(0, nextBoundaryInText)).length - 2 // -2 para \r\n
      }
      
      result[fieldName] = arrayBuffer.slice(bodyStartBytes, bodyEndBytes)
    } else {
      // Para campos de texto (camera_pose), parsearlo
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

    // --- WASD Controls ---
    const keys = {
      KeyW: false, KeyA: false, KeyS: false, KeyD: false,
      Space: false, ShiftLeft: false, ShiftRight: false
    }
    
    const onKeyDown = (e) => {
      if (e.code in keys) {
        keys[e.code] = true
      }
    }
    
    const onKeyUp = (e) => {
      if (e.code in keys) {
        keys[e.code] = false
      }
    }
    
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)

    // --- Animación con delta time ---
    const clock = new THREE.Clock()
    const moveSpeed = 3.5
    const upVec = new THREE.Vector3(0, 1, 0)
    const tmpFwd = new THREE.Vector3()
    const tmpRight = new THREE.Vector3()
    
    const animate = () => {
      this._raf = requestAnimationFrame(animate)
      const dt = clock.getDelta()
      
      // Actualizar controles WASD con delta time
      if (keys.KeyW || keys.KeyS || keys.KeyA || keys.KeyD || keys.Space || keys.ShiftLeft || keys.ShiftRight) {
        // Asegurar que la cámara puede actualizarse
        camera.matrixAutoUpdate = true
        
        camera.getWorldDirection(tmpFwd).normalize()
        tmpRight.copy(tmpFwd).cross(upVec).normalize()
        
        const vel = moveSpeed * dt
        const delta = new THREE.Vector3()
        
        if (keys.KeyW) delta.addScaledVector(tmpFwd, vel)
        if (keys.KeyS) delta.addScaledVector(tmpFwd, -vel)
        if (keys.KeyA) delta.addScaledVector(tmpRight, -vel)
        if (keys.KeyD) delta.addScaledVector(tmpRight, vel)
        if (keys.Space) delta.addScaledVector(upVec, vel)
        if (keys.ShiftLeft || keys.ShiftRight) delta.addScaledVector(upVec, -vel)
        
        camera.position.add(delta)
        orbit.target.add(delta)
        camera.updateProjectionMatrix()
        orbit.update()
      }
      
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
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
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
     * Pide el JSON { pointcloud: ..., camera: 4x4 } o multipart.
     * Admite pointcloud como:
     *  - base64
     *  - multipart/form-data
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

      if (ctype.includes('multipart/form-data')) {
        // Nueva API: multipart con pointcloud y camera_pose
        const buffer = await res.arrayBuffer()
        const parsed = parseMultipartResponse(buffer, ctype)
        
        if (parsed.pointcloud) {
          plyBuffer = parsed.pointcloud
        }
        
        if (parsed.camera_pose) {
          try {
            const cameraPose = JSON.parse(parsed.camera_pose)
            if (Array.isArray(cameraPose) && cameraPose.length === 4 && cameraPose.every(r => Array.isArray(r) && r.length === 4)) {
              cameraCv = cameraPose
            }
          } catch (e) {
            console.warn('[PLYViewer] Failed to parse camera_pose from multipart:', e)
          }
        }
      } else if (ctype.includes('application/json')) {
        const json = await res.json()

        // 1) obtener el PLY
        if (typeof json.pointcloud === 'string') {
          if (json.pointcloud.startsWith('http') || json.pointcloud.startsWith('/')) {
            // URL directa
            plyUrl = json.pointcloud
          } else if (json.pointcloud.startsWith('data:') || /^[A-Za-z0-9+/]+=*$/.test(json.pointcloud)) {
            // dataURL o base64 "puro"
            plyBuffer = base64ToArrayBuffer(json.pointcloud)
          }
        } else {
          console.warn('[PLYViewer] pointcloud no es string; espera URL o base64.')
        }

        // 2) cámara (OpenCV Camera-to-World 4x4)
        if (Array.isArray(json.camera_pose) && json.camera_pose.length === 4 && json.camera_pose.every(r => Array.isArray(r) && r.length === 4)) {
          cameraCv = json.camera_pose
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
        throw new Error('[PLYViewer] No se pudo obtener pointcloud (ni URL ni buffer)')
      }

      // Crear objeto
      geometry.computeVertexNormals?.()
      
      // Calcular bounding box una sola vez
      geometry.computeBoundingBox?.()
      const bbox = geometry.boundingBox
      let center = new THREE.Vector3(0, 0, 0)
      let maxDim = 1
      let pointSize = 0.02 // tamaño base más grande
      
      if (bbox) {
        center = bbox.getCenter(new THREE.Vector3())
        const size = bbox.getSize(new THREE.Vector3())
        maxDim = Math.max(size.x, size.y, size.z) || 1
        // Ajustar el tamaño del punto según el tamaño de la escena (0.5% del tamaño máximo)
        pointSize = maxDim * 0.005
      }
      
      let object
      if (geometry.getAttribute && geometry.getAttribute('color')) {
        const mat = new THREE.PointsMaterial({ 
          size: pointSize, 
          vertexColors: true, 
          sizeAttenuation: true 
        })
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

      // Ajustar near/far en función del tamaño (ya calculado)
      this._camera.near = Math.max(maxDim / 1000, 0.001)
      this._camera.far  = Math.max(maxDim * 100, 10)
      this._camera.updateProjectionMatrix()

      // Si viene cámara → aplicarla. Si no, encuadrar por bbox.
      if (cameraCv) {
        this._applyOpenCVCameraToWorld(cameraCv, true)
        console.log('[PLYViewer] Cámara aplicada')
      } else {
        // Fallback: encuadrar por BBox
        const fov = this._camera.fov * (Math.PI / 180)
        let camDist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5
        this._camera.position.set(center.x, center.y, center.z + camDist)
        this._camera.lookAt(center)
        this._orbit.target.copy(center)
        this._camera.updateProjectionMatrix()
        this._orbit.update()
        console.log('[PLYViewer] Encuadrado por BBox')
      }
    },

    /**
     * Método para agregar puntos incrementales desde POST
     * FUSIONA los nuevos puntos con los existentes (no reemplaza)
     */
    async addIncrementalPoints(arrayBuffer, contentType) {
      console.log('[PLYViewer] Adding incremental points...')
      
      // Parsear respuesta multipart
      let plyBuffer = null
      let cameraCv = null
      
      if (contentType.includes('multipart/form-data')) {
        const parsed = parseMultipartResponse(arrayBuffer, contentType)
        
        if (parsed.pointcloud) {
          plyBuffer = parsed.pointcloud
        }
        
        if (parsed.camera_pose) {
          try {
            const cameraPose = JSON.parse(parsed.camera_pose)
            if (Array.isArray(cameraPose) && cameraPose.length === 4 && cameraPose.every(r => Array.isArray(r) && r.length === 4)) {
              cameraCv = cameraPose
              console.log('[PLYViewer] Parsed camera pose:', cameraPose)
            }
          } catch (e) {
            console.warn('[PLYViewer] Failed to parse camera_pose:', e)
          }
        }
      } else {
        // Fallback: asumir que es el PLY directo
        plyBuffer = arrayBuffer
      }
      
      if (!plyBuffer) {
        console.error('[PLYViewer] No pointcloud data in response')
        return
      }
      
      // Cargar geometría desde buffer
      const loader = new PLYLoader()
      const geometry = loader.parse(plyBuffer)
      geometry.computeVertexNormals?.()
      
      // Calcular tamaño de punto adaptativo basado en la escena
      geometry.computeBoundingBox?.()
      const bbox = geometry.boundingBox
      let pointSize = 0.02 // tamaño base más grande
      if (bbox) {
        const size = bbox.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z) || 1
        // Ajustar el tamaño del punto según el tamaño de la escena
        pointSize = maxDim * 0.005 // 0.5% del tamaño máximo
      }
      
      // Crear objeto nuevo
      let newObject
      if (geometry.getAttribute && geometry.getAttribute('color')) {
        const mat = new THREE.PointsMaterial({ 
          size: pointSize, 
          vertexColors: true, 
          sizeAttenuation: true 
        })
        newObject = new THREE.Points(geometry, mat)
        newObject.raycast = THREE.Points.prototype.raycast
      } else {
        const mat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false })
        newObject = new THREE.Mesh(geometry, mat)
      }
      
      // AGREGAR a la escena (no reemplazar)
      this._scene.add(newObject)
      this._pickables.push(newObject)
      // Mantener referencia al último objeto añadido
      this._currentObject = newObject
      
      console.log(`[PLYViewer] Now have ${this._pickables.length} point cloud objects in scene`)
      
      // Calcular bounding box combinado de todos los objetos
      const combinedBox = new THREE.Box3()
      for (const obj of this._pickables) {
        if (obj.geometry) {
          obj.geometry.computeBoundingBox?.()
          if (obj.geometry.boundingBox) {
            combinedBox.union(obj.geometry.boundingBox)
          }
        }
      }
      
      // Ajustar near/far basado en el tamaño total
      let center = new THREE.Vector3(0, 0, 0)
      let maxDim = 1
      if (!combinedBox.isEmpty()) {
        center = combinedBox.getCenter(new THREE.Vector3())
        const size = combinedBox.getSize(new THREE.Vector3())
        maxDim = Math.max(size.x, size.y, size.z) || 1
        this._camera.near = Math.max(maxDim / 1000, 0.001)
        this._camera.far  = Math.max(maxDim * 100, 10)
        this._camera.updateProjectionMatrix()
      }
      
      // SIEMPRE aplicar cámara con animación si está disponible
      if (cameraCv) {
        console.log('[PLYViewer] Applying camera with animation')
        this._applyOpenCVCameraToWorld(cameraCv, true) // true = animar SIEMPRE
      } else {
        console.warn('[PLYViewer] No camera pose provided, keeping current view')
        // No cambiar la cámara si no hay pose
      }
      
      console.log('[PLYViewer] Incremental points added successfully')
    },

    /**
     * Aplica una matriz 4x4 Camera-to-World en convención OpenCV a Three.js.
     * OpenCV: x→derecha, y→abajo, z→adelante
     * Three:  x→derecha, y→arriba,  z→hacia el observador (cámara mira -Z)
     *
     * Conversión aproximada: M_three = C * M_cv * C, con C = diag(1, -1, -1, 1)
     * Luego se asigna a camera.matrixWorld.
     * 
     * @param {Array} cv4x4 - matriz 4x4 de cámara en convención OpenCV
     * @param {Boolean} animate - si true, anima la transición de cámara
     */
    _applyOpenCVCameraToWorld(cv4x4, animate = false) {
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

      // 3) calcular nueva posición y target
      const newPosition = new THREE.Vector3().setFromMatrixPosition(Mthree)
      const rot = new THREE.Matrix4().extractRotation(Mthree)
      const forward = new THREE.Vector3(0, 0, -1).applyMatrix4(rot).normalize()
      const newTarget = new THREE.Vector3().copy(newPosition).add(forward)

      if (animate) {
        console.log('[PLYViewer] Starting camera animation...')
        console.log('[PLYViewer] From:', this._camera.position.toArray())
        console.log('[PLYViewer] To:', newPosition.toArray())
        
        // Asegurar que la cámara puede actualizarse durante la animación
        this._camera.matrixAutoUpdate = true
        
        // Animación suave de la cámara
        const startPosition = this._camera.position.clone()
        const startTarget = this._orbit.target.clone()
        const duration = 1500 // 1.5 segundos (más visible)
        const startTime = Date.now()
        
        const animateCamera = () => {
          const elapsed = Date.now() - startTime
          const t = Math.min(elapsed / duration, 1)
          // Easing suave (ease-in-out)
          const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
          
          this._camera.position.lerpVectors(startPosition, newPosition, eased)
          this._orbit.target.lerpVectors(startTarget, newTarget, eased)
          this._camera.lookAt(this._orbit.target)
          this._camera.updateProjectionMatrix()
          this._orbit.update()
          
          if (t < 1) {
            requestAnimationFrame(animateCamera)
          } else {
            // Al finalizar, establecer matriz final
            console.log('[PLYViewer] Camera animation complete!')
            this._camera.matrixAutoUpdate = false
            this._camera.matrixWorld.copy(Mthree)
            this._camera.matrixWorldNeedsUpdate = true
            this._orbit.update()
          }
        }
        animateCamera()
      } else {
        // Aplicación inmediata
        this._camera.matrixAutoUpdate = false
        this._camera.matrixWorld.copy(Mthree)
        this._camera.matrixWorldNeedsUpdate = true
        this._camera.position.copy(newPosition)
        this._orbit.target.copy(newTarget)
        this._camera.lookAt(newTarget)
        this._camera.updateProjectionMatrix()
        this._orbit.update()
      }
    },

    _removeCurrentObject() {
      // Remover TODOS los objetos acumulados, no solo el actual
      for (const obj of this._pickables) {
        this._scene.remove(obj)
        obj.geometry?.dispose?.()
        const m = obj.material
        if (Array.isArray(m)) m.forEach(mm => mm?.dispose?.())
        else m?.dispose?.()
      }
      this._currentObject = null
      this._pickables = []
      console.log('[PLYViewer] Removed all point cloud objects from scene')
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
