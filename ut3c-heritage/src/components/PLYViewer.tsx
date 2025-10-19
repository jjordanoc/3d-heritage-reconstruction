"use client";

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

interface PLYViewerProps {
  plyUrl: string;
}

const PLYViewer = ({ plyUrl }: PLYViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (typeof window === 'undefined' || !containerRef.current) {
      return;
    }

    const container = containerRef.current;

    // --- Scene / Camera / Renderer ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0f11);

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.001,
      1e6
    );
    camera.position.set(0, 0, 3);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // --- Lights ---
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(5, 10, 7.5);
    scene.add(dir);

    // --- ArcballControls ---
    const orbit = new ArcballControls(camera, renderer.domElement, scene);
    orbit.setGizmosVisible(false);
    orbit.enableAnimations = true;
    orbit.dampingFactor = 0.1;

    // --- Load PLY ---
    const pickables: THREE.Object3D[] = [];
    const loader = new PLYLoader();

    // Fetch PLY from remote URL, convert to blob, then load
    let blobUrl: string | null = null;

    fetch(plyUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch PLY: ${response.statusText}`);
        }
        return response.blob();
      })
      .then(blob => {
        blobUrl = URL.createObjectURL(blob);
        return new Promise<THREE.BufferGeometry>((resolve, reject) => {
          loader.load(
            blobUrl!,
            (geometry) => resolve(geometry),
            undefined,
            (err) => reject(err)
          );
        });
      })
      .then((geometry) => {
        geometry.computeVertexNormals?.();

        let object;
        if (geometry.hasAttribute('color')) {
          const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, sizeAttenuation: true });
          object = new THREE.Points(geometry, material);
        } else {
          const material = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, flatShading: false });
          object = new THREE.Mesh(geometry, material);
        }
        scene.add(object);
        pickables.push(object);

        // Adjust camera and pivot initial
        geometry.computeBoundingBox?.();
        const bbox = geometry.boundingBox;
        if (bbox) {
          const center = new THREE.Vector3();
          bbox.getCenter(center);
          const radius = bbox.getSize(new THREE.Vector3()).length() * 0.6;

          camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, radius)));
          camera.near = Math.max(radius / 1000, 0.001);
          camera.far = radius * 100;
          camera.updateProjectionMatrix();

          setPivot(center);
          orbit.update();
        }
      })
      .catch((err) => {
        console.error('PLY load error', err);
      });

    // --- Double click: set pivot ---
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onDblClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(pickables, true);
      if (hits.length) {
        setPivot(hits[0].point);
      }
    };
    renderer.domElement.addEventListener('dblclick', onDblClick);

    // --- Keyboard movement ---
    const keys: { [key: string]: boolean } = {
      KeyW: false, KeyA: false, KeyS: false, KeyD: false,
      Space: false, ShiftLeft: false, ShiftRight: false
    };
    const onKeyDown = (e: KeyboardEvent) => { if (e.code in keys) keys[e.code] = true; };
    const onKeyUp = (e: KeyboardEvent) => { if (e.code in keys) keys[e.code] = false; };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    function moveCameraAndTarget(deltaVec: THREE.Vector3) {
      camera.position.add(deltaVec);
      (orbit as any).target.add(deltaVec);
      orbit.update();
    }

    function setPivot(newPivot: THREE.Vector3) {
      const camToPivot = new THREE.Vector3().subVectors(camera.position, (orbit as any).target);
      (orbit as any).target.copy(newPivot);
      camera.position.copy(newPivot).add(camToPivot);
      camera.updateProjectionMatrix();
      orbit.update();
    }

    // --- Animation ---
    const clock = new THREE.Clock();
    const moveSpeed = 3.5;
    const upVec = new THREE.Vector3(0, 1, 0);
    const tmpFwd = new THREE.Vector3();
    const tmpRight = new THREE.Vector3();

    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      const dt = clock.getDelta();

      if (keys.KeyW || keys.KeyS || keys.KeyA || keys.KeyD || keys.Space || keys.ShiftLeft || keys.ShiftRight) {
        camera.getWorldDirection(tmpFwd).normalize();
        tmpRight.copy(tmpFwd).cross(upVec).normalize();

        const vel = moveSpeed * dt;
        const delta = new THREE.Vector3();
        if (keys.KeyW) delta.addScaledVector(tmpFwd, vel);
        if (keys.KeyS) delta.addScaledVector(tmpFwd, -vel);
        if (keys.KeyA) delta.addScaledVector(tmpRight, -vel);
        if (keys.KeyD) delta.addScaledVector(tmpRight, vel);
        if (keys.Space) delta.addScaledVector(upVec, vel);
        if (keys.ShiftLeft || keys.ShiftRight) delta.addScaledVector(upVec, -vel);

        moveCameraAndTarget(delta);
      }

      orbit.update();
      renderer.render(scene, camera);
    };

    const onVisibility = () => {
      if (document.hidden && rafRef.current) cancelAnimationFrame(rafRef.current);
      else animate();
    };
    document.addEventListener('visibilitychange', onVisibility);

    animate();

    // --- Resize ---
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      orbit.update();
    };
    window.addEventListener('resize', onResize);

    // --- Cleanup ---
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', onResize);
      document.removeEventListener('visibilitychange', onVisibility);
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      renderer.domElement.removeEventListener('dblclick', onDblClick);
      orbit.dispose?.();
      renderer.dispose();
      if (container && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      // Clean up blob URL
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [plyUrl]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}></div>;
};

export default PLYViewer;
