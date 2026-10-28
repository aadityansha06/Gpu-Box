// src/lib/hero3d.js
// 3D Metallic Gold Silicon Die & Torus Bus Visualization using Three.js
import * as THREE from 'three';

let scene, camera, renderer, group, clock, rings = [];
let rafId;

export function startHero3D(canvasEl) {
  if (!canvasEl) return () => {};
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, canvasEl.clientWidth / canvasEl.clientHeight, 0.1, 100);
  camera.position.set(0, 0, 7.2);

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
  renderer.setSize(canvasEl.clientWidth, canvasEl.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x040407, 1);

  clock = new THREE.Clock();
  group = new THREE.Group();

  // Luxury Gold & Titanium Silicon Die geometry
  const goldMaterial = new THREE.MeshStandardMaterial({
    color: 0xd4af37,
    emissive: 0x6e5210,
    emissiveIntensity: 0.45,
    metalness: 0.85,
    roughness: 0.25,
    transparent: true,
    opacity: 0.65
  });

  // Concentric Bus Rings
  for (let i = 0; i < 4; i++) {
    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(1.15 + i * 0.58, 0.032, 16, 64),
      goldMaterial
    );
    torus.rotation.x = Math.PI / 2;
    group.add(torus);
    rings.push(torus);
  }

  // Central Silicon Micro-Architectural Die (Titanium / Obsidian Core)
  const coreMaterial = new THREE.MeshStandardMaterial({
    color: 0x14141b,
    emissive: 0x42320b,
    emissiveIntensity: 0.7,
    metalness: 0.9,
    roughness: 0.2,
    transparent: true,
    opacity: 0.92
  });
  group.add(new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.92, 0.45), coreMaterial));

  // Champagne Gold Wireframe Shroud
  const wireframeMaterial = new THREE.MeshBasicMaterial({
    color: 0xf5d77f,
    wireframe: true,
    transparent: true,
    opacity: 0.28
  });
  group.add(new THREE.Mesh(new THREE.BoxGeometry(0.98, 0.98, 0.5), wireframeMaterial));

  // Golden Point Lights
  const l1 = new THREE.PointLight(0xffe299, 2.2, 16);
  l1.position.set(5, 3.5, 5);

  const l2 = new THREE.PointLight(0xd4af37, 1.8, 16);
  l2.position.set(-5, -3, 5);

  const ambient = new THREE.AmbientLight(0x221d12, 0.6);
  scene.add(ambient, l1, l2, group);

  function animate() {
    const t = clock.getElapsedTime();
    if (group) {
      group.rotation.y = t * 0.18;
      group.rotation.x = Math.sin(t * 0.4) * 0.15;
      group.rotation.z = Math.cos(t * 0.3) * 0.1;

      for (let i = 0; i < rings.length; i++) {
        rings[i].rotation.z = t * (0.1 + i * 0.05);
        rings[i].rotation.y = t * (0.06 + i * 0.04);
      }
    }
    renderer.render(scene, camera);
    rafId = requestAnimationFrame(animate);
  }
  animate();

  const onResize = () => {
    if (!canvasEl) return;
    camera.aspect = canvasEl.clientWidth / canvasEl.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvasEl.clientWidth, canvasEl.clientHeight);
  };
  window.addEventListener('resize', onResize);

  return () => {
    cancelAnimationFrame(rafId);
    window.removeEventListener('resize', onResize);
    if (renderer) renderer.setAnimationLoop(null);
  };
}

export function stopHero3D() {
  if (rafId) cancelAnimationFrame(rafId);
}
