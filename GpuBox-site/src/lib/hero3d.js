import * as THREE from 'three';

let scene, camera, renderer, group, clock, rings = [];
let rafId;

export function startHero3D(canvasEl) {
  if (!canvasEl) return () => {};
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, canvasEl.clientWidth / canvasEl.clientHeight, 0.1, 100);
  camera.position.set(0, 0, 7);
  renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
  renderer.setSize(canvasEl.clientWidth, canvasEl.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x07080c, 1);
  clock = new THREE.Clock();
  group = new THREE.Group();
  for (let i = 0; i < 4; i++) {
    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(1.1 + i * 0.55, 0.035, 16, 64),
      new THREE.MeshStandardMaterial({ color: 0x4f46e5, emissive: 0x312e81, emissiveIntensity: 0.5, transparent: true, opacity: 0.5 })
    );
    torus.rotation.x = Math.PI / 2;
    group.add(torus);
    rings.push(torus);
  }
  group.add(new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.9, 0.45), new THREE.MeshStandardMaterial({ color: 0x6366f1, emissive: 0x1e1b4b, emissiveIntensity: 0.6, metalness: 0.4, roughness: 0.35, transparent: true, opacity: 0.85 })));
  group.add(new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.95, 0.5), new THREE.MeshBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.35 })));
  const l1 = new THREE.PointLight(0x818cf8, 2, 15); l1.position.set(5, 3, 5);
  const l2 = new THREE.PointLight(0x6366f1, 1.5, 15); l2.position.set(-5, -3, 5);
  scene.add(new THREE.AmbientLight(0x404040, 0.5), l1, l2, group);
  function animate() {
    const t = clock.getElapsedTime();
    if (group) {
      group.rotation.y = t * 0.18; group.rotation.x = Math.sin(t * 0.4) * 0.15; group.rotation.z = Math.cos(t * 0.3) * 0.1;
      for (let i = 0; i < rings.length; i++) { rings[i].rotation.z = t * (0.1 + i * 0.05); rings[i].rotation.y = t * (0.06 + i * 0.04); }
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

