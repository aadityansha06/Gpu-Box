// src/lib/hero3d.js
// 3D GPU Silicon Die with Streaming Multiprocessor (SM) Cluster & NVLink Bus Visualization using Three.js
import * as THREE from 'three';

let scene, camera, renderer, rootGroup, clock;
let smMeshes = [];
let rings = [];
let busParticles;
let rafId;

export function startHero3D(canvasEl) {
  if (!canvasEl) return () => {};

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, canvasEl.clientWidth / canvasEl.clientHeight, 0.1, 100);
  camera.position.set(0, 3.8, 6.2);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
  renderer.setSize(canvasEl.clientWidth, canvasEl.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0); // Transparent over pure black background

  clock = new THREE.Clock();
  rootGroup = new THREE.Group();
  smMeshes = [];
  rings = [];

  // ==========================================
  // 1. SILICON SUBSTRATE & PACKAGE (BGA Carrier)
  // ==========================================
  const packageMaterial = new THREE.MeshStandardMaterial({
    color: 0x090a0f,
    metalness: 0.85,
    roughness: 0.35,
    emissive: 0x18140a,
    emissiveIntensity: 0.3
  });

  const goldTrimMaterial = new THREE.MeshStandardMaterial({
    color: 0xd4af37,
    metalness: 0.95,
    roughness: 0.2,
    emissive: 0x544010,
    emissiveIntensity: 0.4
  });

  // Base Package (Carrier Board)
  const baseMesh = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.16, 3.4), packageMaterial);
  baseMesh.position.y = -0.12;
  rootGroup.add(baseMesh);

  // Gold Heatspreader / Interposer Rim
  const rimMesh = new THREE.Mesh(
    new THREE.BoxGeometry(4.45, 0.04, 3.45),
    goldTrimMaterial
  );
  rimMesh.position.y = -0.04;
  rootGroup.add(rimMesh);

  // Central Silicon Die Substrate
  const dieMesh = new THREE.Mesh(
    new THREE.BoxGeometry(3.6, 0.12, 2.6),
    new THREE.MeshStandardMaterial({
      color: 0x07080d,
      metalness: 0.9,
      roughness: 0.15,
      emissive: 0x221a08,
      emissiveIntensity: 0.5
    })
  );
  dieMesh.position.y = 0.04;
  rootGroup.add(dieMesh);

  // ==========================================
  // 2. STREAMING MULTIPROCESSOR (SM) CLUSTERS
  // ==========================================
  // 2 GPCs (Graphics Processing Clusters): Left & Right
  // Each has 4 rows x 2 cols = 8 SMs => 16 SMs total
  const smMaterial = new THREE.MeshStandardMaterial({
    color: 0x0f1118,
    metalness: 0.88,
    roughness: 0.22,
    emissive: 0xd4af37,
    emissiveIntensity: 0.15
  });

  const smEdgeMaterial = new THREE.LineBasicMaterial({
    color: 0xf5d77f,
    transparent: true,
    opacity: 0.85
  });

  const smWidth = 0.52;
  const smLength = 0.42;
  const smHeight = 0.14;
  const smGeom = new THREE.BoxGeometry(smWidth, smHeight, smLength);
  const smEdgesGeom = new THREE.EdgesGeometry(smGeom);

  const clusterOffsets = [-1.05, 1.05]; // Left and Right GPC
  const rows = 4;
  const cols = 2;

  clusterOffsets.forEach((centerX, gpcIdx) => {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const smGroup = new THREE.Group();

        const x = centerX + (c - 0.5) * (smWidth + 0.12);
        const z = (r - 1.5) * (smLength + 0.14);

        const mesh = new THREE.Mesh(smGeom, smMaterial.clone());
        const wire = new THREE.LineSegments(smEdgesGeom, smEdgeMaterial);

        smGroup.add(mesh);
        smGroup.add(wire);
        smGroup.position.set(x, 0.14, z);

        // Store SM reference with index for thread wave animation
        smMeshes.push({
          group: smGroup,
          mesh,
          baseY: 0.14,
          speed: 1.5 + (r * cols + c) * 0.25,
          phase: (gpcIdx * 8 + r * cols + c) * 0.4
        });

        rootGroup.add(smGroup);
      }
    }
  });

  // ==========================================
  // 3. CENTRAL L2 CACHE & CROSSBAR BUS
  // ==========================================
  const l2BusMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.32, 0.08, 2.3),
    new THREE.MeshStandardMaterial({
      color: 0xd4af37,
      emissive: 0xf5d77f,
      emissiveIntensity: 0.75,
      metalness: 0.9,
      roughness: 0.1
    })
  );
  l2BusMesh.position.set(0, 0.1, 0);
  rootGroup.add(l2BusMesh);

  // Micro Interconnect Bus Lines (Crossbar traces between SMs and L2)
  const lineMat = new THREE.LineBasicMaterial({
    color: 0xd4af37,
    transparent: true,
    opacity: 0.45
  });

  for (let i = -1.2; i <= 1.2; i += 0.4) {
    const points = [
      new THREE.Vector3(-1.6, 0.1, i),
      new THREE.Vector3(1.6, 0.1, i)
    ];
    const lineGeom = new THREE.BufferGeometry().setFromPoints(points);
    const busLine = new THREE.Line(lineGeom, lineMat);
    rootGroup.add(busLine);
  }

  // ==========================================
  // 4. NVLINK RING INTERCONNECT TORI
  // ==========================================
  const ringMat = new THREE.MeshStandardMaterial({
    color: 0xd4af37,
    emissive: 0x8a6a16,
    emissiveIntensity: 0.4,
    metalness: 0.9,
    roughness: 0.2,
    transparent: true,
    opacity: 0.55
  });

  for (let i = 0; i < 3; i++) {
    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(2.6 + i * 0.65, 0.024, 16, 80),
      ringMat
    );
    torus.rotation.x = Math.PI / 2.3 + (i * 0.2);
    torus.rotation.y = i * 0.35;
    rootGroup.add(torus);
    rings.push({ mesh: torus, speed: 0.15 + i * 0.08 });
  }

  // ==========================================
  // 5. PACKET FLOW PARTICLES
  // ==========================================
  const particleCount = 48;
  const particleGeom = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);

  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const radius = 2.6 + (i % 3) * 0.65;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = (Math.sin(angle * 2) * 0.25);
    positions[i * 3 + 2] = Math.sin(angle) * radius;
  }

  particleGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const particleMat = new THREE.PointsMaterial({
    color: 0xfff0aa,
    size: 0.07,
    transparent: true,
    opacity: 0.85
  });
  busParticles = new THREE.Points(particleGeom, particleMat);
  rootGroup.add(busParticles);

  // ==========================================
  // 6. LIGHTING
  // ==========================================
  const ambient = new THREE.AmbientLight(0xfffaed, 0.5);
  scene.add(ambient);

  const topGoldLight = new THREE.DirectionalLight(0xffe899, 2.4);
  topGoldLight.position.set(4, 7, 5);
  scene.add(topGoldLight);

  const fillLight = new THREE.PointLight(0xd4af37, 2.0, 14);
  fillLight.position.set(-5, 3, -4);
  scene.add(fillLight);

  const underGlow = new THREE.PointLight(0x7c5a14, 1.5, 8);
  underGlow.position.set(0, -2, 0);
  scene.add(underGlow);

  scene.add(rootGroup);

  // Initial tilt for beautiful perspective isometric look
  rootGroup.rotation.x = 0.55;
  rootGroup.rotation.y = -0.45;

  // ==========================================
  // ANIMATION LOOP
  // ==========================================
  function animate() {
    const elapsed = clock.getElapsedTime();

    if (rootGroup) {
      // Gentle ambient floating orientation
      rootGroup.rotation.y = -0.45 + Math.sin(elapsed * 0.25) * 0.18;
      rootGroup.rotation.x = 0.55 + Math.cos(elapsed * 0.3) * 0.08;

      // Animate SM thread execution wave
      for (let i = 0; i < smMeshes.length; i++) {
        const item = smMeshes[i];
        const pulse = Math.sin(elapsed * item.speed + item.phase);
        item.group.position.y = item.baseY + Math.max(0, pulse * 0.04);
        item.mesh.material.emissiveIntensity = 0.15 + (pulse > 0 ? pulse * 0.45 : 0);
      }

      // Rotate NVLink Rings
      for (let i = 0; i < rings.length; i++) {
        rings[i].mesh.rotation.z = elapsed * rings[i].speed;
      }

      // Rotate Packet Particles
      if (busParticles) {
        busParticles.rotation.y = elapsed * 0.22;
      }
    }

    renderer.render(scene, camera);
    rafId = visible ? requestAnimationFrame(animate) : 0;
  }
  // Stop rendering while the hero is scrolled out of view (saves GPU/battery).
  let visible = true;
  let io = null;
  if (typeof IntersectionObserver !== 'undefined') {
    io = new IntersectionObserver((entries) => {
      const now = entries[0].isIntersecting;
      if (now && !visible) { visible = true; rafId = requestAnimationFrame(animate); }
      else visible = now;
    }, { threshold: 0 });
    io.observe(canvasEl);
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
    if (io) io.disconnect();
    cancelAnimationFrame(rafId);
    window.removeEventListener('resize', onResize);
    if (renderer) renderer.setAnimationLoop(null);
  };
}

export function stopHero3D() {
  if (rafId) cancelAnimationFrame(rafId);
}
