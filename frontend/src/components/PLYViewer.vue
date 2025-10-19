<template>
  <div ref="container" style="width:100%;height:100%;position:relative;"></div>
</template>

<script>
import * as THREE from 'three'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

export default {
  name: 'PLYViewer',
  data() {
    return {
      _raf: null,
      _cleanup: null,
    }
  },
  mounted() {
    const container = this.$refs.container

    // --- Escena / Cámara / Render ---
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0f0f11)

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.001,
      1e6
    )
    camera.position.set(0, 0, 3)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    container.appendChild(renderer.domElement)

    // --- Luces ---
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6))
    const dir = new THREE.DirectionalLight(0xffffff, 0.9)
    dir.position.set(5, 10, 7.5)
    scene.add(dir)

    // --- OrbitControls (un solo modo) ---
    const orbit = new ArcballControls(camera, renderer.domElement, scene)
    orbit.setGizmosVisible(false)       // sin widgets
    orbit.enableAnimations = true
    orbit.dampingFactor = 0.1           // sensación similar a damping

    // --- Cargar PLY ---
    const pickables = []
    const loader = new PLYLoader()
    loader.load(
      '/ply/auditorio.ply',
      (geometry) => {
        geometry.computeVertexNormals?.()

        let object
        if (geometry.getAttribute && geometry.getAttribute('color')) {
          const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, sizeAttenuation: true })
          object = new THREE.Points(geometry, material)
          object.raycast = THREE.Points.prototype.raycast
        } else {
          const material = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false })
          object = new THREE.Mesh(geometry, material)
        }
        scene.add(object)
        pickables.push(object)

        // Ajustar cámara y pivot inicial
        geometry.computeBoundingBox?.()
        const bbox = geometry.boundingBox
        if (bbox) {
          const center = new THREE.Vector3()
          bbox.getCenter(center)
          const radius = bbox.getSize(new THREE.Vector3()).length() * 0.6

          camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, radius)))
          camera.near = Math.max(radius / 1000, 0.001)
          camera.far  = radius * 100
          camera.updateProjectionMatrix()

          setPivot(center)
          orbit.update()
        }
      },
      undefined,
      (err) => console.error('PLY load error', err)
    )

    // --- Doble click: fijar pivot en el punto clicado ---
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()
    const onDblClick = (e) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(pickables, true)
      if (hits.length) {
        setPivot(hits[0].point)
      }
    }
    renderer.domElement.addEventListener('dblclick', onDblClick)

    // --- Movimiento con teclado (WASD + Space/Shift) ---
    const keys = {
      KeyW: false, KeyA: false, KeyS: false, KeyD: false,
      Space: false, ShiftLeft: false, ShiftRight: false
    }
    const onKeyDown = (e) => { if (e.code in keys) keys[e.code] = true }
    const onKeyUp   = (e) => { if (e.code in keys) keys[e.code] = false }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)

    // Mover cámara **y** pivot juntos (strafe/dolly)
    function moveCameraAndTarget(deltaVec) {
      camera.position.add(deltaVec)
      orbit.target.add(deltaVec)
      orbit.update()
    }

    // Fijar pivot manteniendo la misma distancia relativa de la cámara
    function setPivot(newPivot) {
      const camToPivot = new THREE.Vector3().subVectors(camera.position, orbit.target)
      orbit.target.copy(newPivot)
      camera.position.copy(newPivot).add(camToPivot)
      camera.updateProjectionMatrix()
      orbit.update()
    }
    // expón si lo quieres usar desde fuera
    this.setPivot = (x, y, z) => setPivot(new THREE.Vector3(x, y, z))

    // --- Animación ---
    const clock = new THREE.Clock()
    const moveSpeed = 3.5
    const upVec = new THREE.Vector3(0, 1, 0)
    const tmpFwd = new THREE.Vector3()
    const tmpRight = new THREE.Vector3()

    const animate = () => {
      this._raf = requestAnimationFrame(animate)
      const dt = clock.getDelta()

      // WASD: mover en 3D siguiendo la orientación de cámara
      if (keys.KeyW || keys.KeyS || keys.KeyA || keys.KeyD || keys.Space || keys.ShiftLeft || keys.ShiftRight) {
        camera.getWorldDirection(tmpFwd).normalize()
        tmpRight.copy(tmpFwd).cross(upVec).normalize()

        const vel = moveSpeed * dt
        const delta = new THREE.Vector3()
        if (keys.KeyW) delta.addScaledVector(tmpFwd,  vel)
        if (keys.KeyS) delta.addScaledVector(tmpFwd, -vel)
        if (keys.KeyA) delta.addScaledVector(tmpRight, -vel)
        if (keys.KeyD) delta.addScaledVector(tmpRight,  vel)
        if (keys.Space)        delta.addScaledVector(upVec,  vel)
        if (keys.ShiftLeft || keys.ShiftRight) delta.addScaledVector(upVec, -vel)

        moveCameraAndTarget(delta)
      }

      orbit.update()
      renderer.render(scene, camera)
    }

    // Pausar cuando la pestaña no está visible
    const onVisibility = () => {
      if (document.hidden) cancelAnimationFrame(this._raf)
      else animate()
    }
    document.addEventListener('visibilitychange', onVisibility)

    // Iniciar
    animate()

    // --- Resize ---
    const onResize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      orbit.update()
    }
    window.addEventListener('resize', onResize)

    // Limpieza
    this._cleanup = () => {
      cancelAnimationFrame(this._raf)
      window.removeEventListener('resize', onResize)
      document.removeEventListener('visibilitychange', onVisibility)
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      renderer.domElement.removeEventListener('dblclick', onDblClick)
      orbit.dispose?.()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  },
  beforeUnmount() {
    this._cleanup && this._cleanup()
  },
  methods: {
    setPivot() {} // se asigna en mounted
  }
}
</script>

<style scoped>
:host, div { width: 100%; height: 100%; display: block; }
</style>
