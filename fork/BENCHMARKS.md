# Benchmark Results

Performance comparison between the reference [ivanov84/ninfer-windows-tp2](https://github.com/ivanov84/ninfer-windows-tp2) build (TP2, MTP, CUDA graphs enabled) and this fork [parallelno/ninfer-dflash2-tp2](https://github.com/parallelno/ninfer-dflash2-tp2) (TP2, DFlash2, CUDA graphs enabled).

## Workload

Prompt: `make a single html 3D snake game`

The benchmark records one agent workflow with tool calls enabled. Sampling seeds, output lengths, and cache reuse differ between the two runs; treat the figures as an observed end-to-end comparison, not a controlled model-quality evaluation.

## Configuration

- Model: Qwen3 8-27B NVFP4
- GPUs: 2 x NVIDIA GeForce RTX 5060 Ti 16 GB
- Topology: GPU 0 on PCIe x16; GPU 1 on PCIe x4; no P2P
- Context capacity: 32,768 tokens
- NVIDIA driver: 616.92; CUDA UMD: 13.4

## Results

| Metric | Reference (MTP) | Fork (DFlash2) | Difference |
| --- | ---: | ---: | ---: |
| Prefill average | 836.44 tok/s | 406.09 tok/s | -51.4% |
| Decode average | 56.78 tok/s | 73.66 tok/s | +29.7% |
| End-to-end time | 1m 34s | 1m 25s | -9s |
| Output tokens | 12,745 | 13,909 | +1,164 |
| Generated HTML | 172 lines, working | 300 lines, working | Both completed |

### Reference

- idle

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:03:00.0  On |                  N/A |
|  0%   48C    P8             12W /  180W |   12251MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:81:00.0  On |                  N/A |
|  0%   50C    P8             10W /  180W |   13528MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

- decode

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:03:00.0  On |                  N/A |
| 46%   70C    P1            143W /  180W |   12304MiB /  16311MiB |     93%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:81:00.0  On |                  N/A |
| 43%   68C    P1            133W /  180W |   13601MiB /  16311MiB |     90%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

- prefill avg: 836.4375 tok/s
- decode avg: 56.78 tok/s
- time: 1m 34s
- context: 32768
- output: 12,745 tokens
- HTML: 172 lines, one-shot, working

### Fork

- idle

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:03:00.0  On |                  N/A |
|  0%   47C    P5             11W /  180W |   14324MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:81:00.0  On |                  N/A |
|  0%   50C    P5             12W /  180W |   13523MiB /  16311MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

- decode

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:03:00.0  On |                  N/A |
|  0%   65C    P1            135W /  180W |   14425MiB /  16311MiB |     96%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:81:00.0  On |                  N/A |
| 31%   66C    P1            108W /  180W |   13518MiB /  16311MiB |     83%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

- prefill avg: 406.09 tok/s
- decode avg: 73.66 tok/s
- time: 1m 25s
- context: 32768
- output: 13,909 tokens
- HTML: 300 lines, one-shot, working

# Full Logs

## Reference

### Server Log

```
[2026-09-13 10:45:27.537] [info] ninfer-serve: loading model...
[2026-09-13 10:45:28.877] [info] ninfer-serve: load        weights                      0.00%           0 B /      20.92 GiB     0.000 s
[2026-09-13 10:45:38.878] [info] ninfer-serve: load        weights                     22.10%      4.62 GiB /      20.92 GiB    10.001 s
[2026-09-13 10:45:48.882] [info] ninfer-serve: load        weights                     34.65%      7.25 GiB /      20.92 GiB    20.005 s
[2026-09-13 10:45:58.921] [info] ninfer-serve: load        weights                     46.01%      9.62 GiB /      20.92 GiB    30.044 s
[2026-09-13 10:46:08.988] [info] ninfer-serve: load        weights                     58.26%     12.19 GiB /      20.92 GiB    40.111 s
[2026-09-13 10:46:19.099] [info] ninfer-serve: load        weights                     70.21%     14.69 GiB /      20.92 GiB    50.222 s
[2026-09-13 10:46:29.107] [info] ninfer-serve: load        weights                     83.06%     17.37 GiB /      20.92 GiB    60.230 s
[2026-09-13 10:46:37.355] [info] ninfer-serve: load        weights                    100.00%     20.92 GiB /      20.92 GiB    68.478 s
[2026-09-13 10:46:43.356] [info] ninfer-serve: model loaded in 75.8169 s
[2026-09-13 10:46:43.356] [info] ninfer-serve: KV capacity auto resolved=32768 tokens pages=512/512 runtime=1001.50 MiB free-after-weights=3.63 GiB free-after-startup=3.40 GiB headroom=1.00 GiB slack=2.66 GiB graphs=6.00 MiB/82.00 MiB graph-peer=0.00 MiB graph-nodes=2092
[2026-09-13 10:46:43.356] [info] ninfer-serve: warming up...
[2026-09-13 10:46:43.672] [info] ninfer-serve: listening on http://127.0.0.1:30000 (model id: qwen-local, auth: disabled)
[2026-09-13 10:55:20.670] [info] ninfer-serve: [req 1] openai_chat_completions stream msgs=3 max_tokens=32000 (client) tools=0 tool_choice=auto tool_history=no thinking=on preserve_thinking=off preserve_change=no sampler=[temp=1.00 top_p=0.95 top_k=20 seed=8030188012238567250] → submitted
[2026-09-13 10:55:21.836] [info] ninfer-serve: [req 1] done finish=stop_token prompt=584 gen=39 cache=0 reuse=full_reset ttft=697ms prefill=844.3tok/s decode=80.3tok/s wall=1.17s speculative=mtp 3.73tok/round (68.2%)
[2026-09-13 10:55:24.146] [info] ninfer-serve: [req 2] openai_chat_completions stream msgs=2 max_tokens=32000 (client) tools=11 tool_choice=auto tool_history=no thinking=on preserve_thinking=off preserve_change=no sampler=[temp=1.00 top_p=0.95 top_k=20 seed=14793560960469179811] → submitted
[2026-09-13 10:55:24.491] [info] ninfer-serve: throughput interval=5.008s prefill=116.6tok/s decode=7.6tok/s running=1 prefilling=1 decode_ready=0 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:55:29.493] [info] ninfer-serve: throughput interval=5.002s prefill=1432.9tok/s decode=0.0tok/s running=1 prefilling=1 decode_ready=0 waiting=0 avg_decode_batch=n/a
[2026-09-13 10:55:34.500] [info] ninfer-serve: throughput interval=5.007s prefill=191.9tok/s decode=56.5tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:55:39.511] [info] ninfer-serve: throughput interval=5.011s prefill=0.0tok/s decode=51.5tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:55:44.512] [info] ninfer-serve: throughput interval=5.001s prefill=0.0tok/s decode=63.2tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:55:49.523] [info] ninfer-serve: throughput interval=5.011s prefill=0.0tok/s decode=51.9tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:55:54.534] [info] ninfer-serve: throughput interval=5.011s prefill=0.0tok/s decode=65.5tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:55:59.543] [info] ninfer-serve: throughput interval=5.009s prefill=0.0tok/s decode=74.5tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:04.548] [info] ninfer-serve: throughput interval=5.005s prefill=0.0tok/s decode=77.3tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:09.563] [info] ninfer-serve: throughput interval=5.015s prefill=0.0tok/s decode=91.1tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:14.577] [info] ninfer-serve: throughput interval=5.014s prefill=0.0tok/s decode=90.7tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:19.589] [info] ninfer-serve: throughput interval=5.012s prefill=0.0tok/s decode=99.4tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:24.564] [info] ninfer-serve: [req 2] done finish=tool_calls tool_calls=1 prompt=8129 gen=4121 cache=0 reuse=full_reset ttft=5857ms prefill=1399.5tok/s decode=75.5tok/s wall=60.47s speculative=mtp 3.25tok/round (56.4%)
[2026-09-13 10:56:24.598] [info] ninfer-serve: throughput interval=5.008s prefill=0.0tok/s decode=100.8tok/s running=0 prefilling=0 decode_ready=0 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:24.842] [info] ninfer-serve: [req 3] openai_chat_completions stream msgs=4 max_tokens=32000 (client) tools=11 tool_choice=auto tool_history=yes thinking=on preserve_thinking=off preserve_change=no sampler=[temp=1.00 top_p=0.95 top_k=20 seed=14129814941607462693] → submitted
[2026-09-13 10:56:29.607] [info] ninfer-serve: throughput interval=5.009s prefill=1226.5tok/s decode=0.0tok/s running=1 prefilling=1 decode_ready=0 waiting=0 avg_decode_batch=n/a
[2026-09-13 10:56:34.620] [info] ninfer-serve: throughput interval=5.013s prefill=1222.1tok/s decode=10.6tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:37.071] [info] ninfer-serve: [req 3] done finish=tool_calls tool_calls=1 prompt=12270 gen=237 cache=0 reuse=full_reset ttft=8910ms prefill=1385.9tok/s decode=70.0tok/s wall=12.29s speculative=mtp 3.06tok/round (51.6%)
[2026-09-13 10:56:37.322] [info] ninfer-serve: [req 4] openai_chat_completions stream msgs=6 max_tokens=32000 (client) tools=11 tool_choice=auto tool_history=yes thinking=on preserve_thinking=off preserve_change=no sampler=[temp=1.00 top_p=0.95 top_k=20 seed=17475051478519940927] → submitted
[2026-09-13 10:56:39.629] [info] ninfer-serve: throughput interval=5.009s prefill=613.3tok/s decode=36.5tok/s running=1 prefilling=1 decode_ready=0 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:44.633] [info] ninfer-serve: throughput interval=5.004s prefill=1419.4tok/s decode=0.0tok/s running=1 prefilling=1 decode_ready=0 waiting=0 avg_decode_batch=n/a
[2026-09-13 10:56:49.643] [info] ninfer-serve: throughput interval=5.010s prefill=468.8tok/s decode=39.5tok/s running=1 prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00
[2026-09-13 10:56:49.943] [info] ninfer-serve: [req 4] done finish=stop_token prompt=12524 gen=221 cache=0 reuse=full_reset ttft=8918ms prefill=1412.7tok/s decode=58.7tok/s wall=12.67s speculative=mtp 2.56tok/round (39.0%)
[2026-09-13 10:56:54.653] [info] ninfer-serve: throughput interval=5.010s prefill=0.0tok/s decode=4.4tok/s running=0 prefilling=0 decode_ready=0 waiting=0 avg_decode_batch=1.00
```

### HTML

``` html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D Snake</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#0b0f1a;font-family:'Segoe UI',system-ui,sans-serif;color:#e6edf7;user-select:none}
  #hud{position:fixed;top:16px;left:16px;z-index:10;pointer-events:none}
  #score{font-size:42px;font-weight:700;text-shadow:0 2px 12px rgba(0,0,0,.6)}
  #best{font-size:14px;opacity:.7}
  #help{position:fixed;bottom:16px;left:16px;z-index:10;font-size:13px;opacity:.75;pointer-events:none;line-height:1.6}
  kbd{background:#1c2540;border:1px solid #34406b;border-radius:4px;padding:1px 6px;font-family:inherit;font-size:12px}
  #overlay{position:fixed;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;background:rgba(5,8,18,.75);backdrop-filter:blur(4px);z-index:20;text-align:center}
  #overlay h1{font-size:48px;margin:0 0 8px}
  #overlay p{font-size:18px;margin:4px 0;opacity:.85}
  #overlay button{margin-top:22px;padding:12px 34px;font-size:17px;font-weight:600;border:none;border-radius:8px;background:#22c55e;color:#06210f;cursor:pointer}
  #overlay button:hover{background:#4ade80}
</style>
</head>
<body>
<div id="hud"><div id="score">0</div><div id="best"></div></div>
<div id="help">
  <kbd>&larr;</kbd><kbd>&rarr;</kbd><kbd>&uarr;</kbd><kbd>&darr;</kbd> move &nbsp;
  <kbd>W</kbd> up &nbsp;<kbd>S</kbd> down &nbsp;
  <kbd>mouse drag</kbd> orbit &nbsp;<kbd>wheel</kbd> zoom<br>
  eat the orb &middot; don't hit walls or yourself
</div>
<div id="overlay">
  <h1 id="overTitle">Game Over</h1>
  <p id="overScore"></p>
  <button onclick="restart()">Play Again</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const N = 8, SEG = 0.9;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f1a);
scene.fog = new THREE.Fog(0x0b0f1a, 22, 55);

const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 100);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(8, 14, 6);
scene.add(dir);

const half = (N - 1) / 2;
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(N, N, N)),
  new THREE.LineBasicMaterial({color: 0x3b4a7a, transparent: true, opacity: 0.55})
);
scene.add(box);
const grid = new THREE.GridHelper(N, N, 0x31406e, 0x1e2a4d);
grid.position.y = -half - 0.5;
scene.add(grid);

let snake, meshes, curDir, queue, food, foodMesh, score, best, interval, alive, lastTick, acc;
best = +(localStorage.getItem('snake3d-best') || 0);
document.getElementById('best').textContent = best ? 'Best: ' + best : '';

function spawnFood(){
  do { food = {x:(Math.random()*N)|0, y:(Math.random()*N)|0, z:(Math.random()*N)|0}; }
  while (snake.some(s => s.x===food.x && s.y===food.y && s.z===food.z));
  foodMesh.position.set(food.x-half, food.y-half, food.z-half);
}
function segMesh(head){
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(SEG, SEG, SEG),
    new THREE.MeshStandardMaterial({color: head ? 0x4ade80 : 0x22c55e, roughness: 0.4, emissive: head ? 0x14532d : 0x052e16})
  );
  scene.add(m);
  return m;
}
function reset(){
  snake = [{x:3,y:3,z:3},{x:2,y:3,z:3},{x:1,y:3,z:3}];
  curDir = {x:1,y:0,z:0};
  queue = []; score = 0; alive = true; interval = 280; acc = 0; lastTick = performance.now();
  document.getElementById('score').textContent = 0;
  document.getElementById('overlay').style.display = 'none';
  meshes = [];
  snake.forEach(s => meshes.push(segMesh(meshes.length===0)));
  if (!foodMesh){
    foodMesh = new THREE.Mesh(new THREE.SphereGeometry(0.42, 20, 20),
      new THREE.MeshStandardMaterial({color: 0xff5533, emissive: 0xb91c1c, roughness: 0.3}));
    scene.add(foodMesh);
  }
  spawnFood();
  snake.forEach((s,i) => meshes[i].position.set(s.x-half, s.y-half, s.z-half));
}
function step(){
  while (queue.length){
    const d = queue.shift();
    if (!(d.x===-curDir.x && d.y===-curDir.y && d.z===-curDir.z)){ curDir = d; break; }
  }
  const head = {x: snake[0].x+curDir.x, y: snake[0].y+curDir.y, z: snake[0].z+curDir.z};
  if (head.x<0||head.x>=N||head.y<0||head.y>=N||head.z<0||head.z>=N) return die();
  const ate = head.x===food.x && head.y===food.y && head.z===food.z;
  const body = ate ? snake : snake.slice(0, -1);
  if (body.some(s => s.x===head.x && s.y===head.y && s.z===head.z)) return die();
  snake.unshift(head);
  if (ate){
    score++;
    document.getElementById('score').textContent = score;
    if (score > best){ best = score; localStorage.setItem('snake3d-best', best); document.getElementById('best').textContent = 'Best: ' + best; }
    interval = Math.max(100, 280 - score*7);
    const m = segMesh(false);
    m.position.copy(meshes[0].position);
    meshes.push(m);
    spawnFood();
  } else snake.pop();
  snake.forEach((s,i) => meshes[i].position.set(s.x-half, s.y-half, s.z-half));
  meshes[0].material.color.setHex(0x4ade80);
  meshes[0].material.emissive.setHex(0x14532d);
}
function die(){
  alive = false;
  document.getElementById('overScore').textContent = 'Score: ' + score + (score>=best && score>0 ? '  (new best!)' : '');
  document.getElementById('overlay').style.display = 'flex';
}
function restart(){ reset(); }

addEventListener('keydown', e => {
  const dirs = {ArrowUp:{x:0,y:0,z:-1}, ArrowDown:{x:0,y:0,z:1}, ArrowLeft:{x:-1,y:0,z:0}, ArrowRight:{x:1,y:0,z:0}, KeyW:{x:0,y:1,z:0}, KeyS:{x:0,y:-1,z:0}};
  const d = dirs[e.code];
  if (d){ e.preventDefault(); if (queue.length < 3) queue.push(d); }
  if (e.code === 'Enter' && !alive) restart();
});

let theta = 0.7, phi = 1.15, radius = 17, dragging = false, px = 0, py = 0;
renderer.domElement.addEventListener('mousedown', e => { dragging = true; px = e.clientX; py = e.clientY; });
addEventListener('mouseup', () => dragging = false);
addEventListener('mousemove', e => {
  if (!dragging) return;
  theta -= (e.clientX - px) * 0.005;
  phi = Math.min(1.45, Math.max(0.2, phi + (e.clientY - py) * 0.005));
  px = e.clientX; py = e.clientY;
});
addEventListener('wheel', e => { radius = Math.min(35, Math.max(8, radius + e.deltaY * 0.01)); }, {passive:true});

function loop(t){
  requestAnimationFrame(loop);
  if (alive){
    acc += t - lastTick;
    while (acc >= interval){ acc -= interval; step(); if (!alive) break; }
  }
  lastTick = t;
  foodMesh.position.y = food ? food.y - half : 0;
  if (foodMesh) foodMesh.scale.setScalar(1 + 0.15 * Math.sin(t * 0.006));
  camera.position.set(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
}

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

reset();
requestAnimationFrame(loop);
</script>
</body>
</html>
```

## This Fork

### Server Log

```
2026-09-13 16:47:17.058  INFO  starting engine
2026-09-13 16:47:17.635  INFO  loading weights | 22.9 GiB
2026-09-13 16:47:27.765  INFO    loading weights 22.4% | 5.12 GiB/22.9 GiB | 355.7 MiB/s | ETA 51.2s
2026-09-13 16:47:37.831  INFO    loading weights 36.3% | 8.31 GiB/22.9 GiB | 350.0 MiB/s | ETA 42.7s
2026-09-13 16:47:47.880  INFO    loading weights 50.2% | 11.5 GiB/22.9 GiB | 365.2 MiB/s | ETA 32.0s
2026-09-13 16:47:58.024  INFO    loading weights 64.1% | 14.7 GiB/22.9 GiB | 624.6 MiB/s | ETA 13.5s
2026-09-13 16:48:08.126  INFO    loading weights 80.2% | 18.4 GiB/22.9 GiB | 559.1 MiB/s | ETA 8.3s
2026-09-13 16:48:12.273  INFO    loading weights 100.0% | 22.9 GiB/22.9 GiB
2026-09-13 16:48:12.273  INFO  weights ready | 22.9 GiB | 54.6s | 429.2 MiB/s
2026-09-13 16:48:18.720  INFO  CUDA graphs ready | 4.7s
2026-09-13 16:48:18.728  INFO  engine ready | qwen3.8-27b/nvfp4 | total 1m 1.7s | weights 22.9 GiB
2026-09-13 16:48:18.729  INFO  rank 0 cuda:0 | weights 12.5 GiB (sharded 8.90 GiB, replicated 1.52 GiB, rank-only 2.07 GiB) | runtime 1.34 GiB (KV 528.0 MiB, graphs 0 B) | free 1.34 GiB of 15.9 GiB
2026-09-13 16:48:18.729  INFO  rank 1 cuda:1 | weights 10.4 GiB (sharded 8.90 GiB, replicated 1.52 GiB, rank-only 0 B) | runtime 1.34 GiB (KV 528.0 MiB, graphs 0 B) | free 2.88 GiB of 15.9 GiB
2026-09-13 16:48:18.730  INFO  rank 0 holds 2.07 GiB of rank-only weights the other rank does not; KV capacity is bounded by the rank with the least free memory
2026-09-13 16:48:18.730  INFO  capacity | KV 32,768 tokens, int8, explicit | pages 512/512 | runtime 1.34 GiB | free 1.34 GiB
2026-09-13 16:48:18.731  INFO  context cache | 1 active + 1 cached device states | host 0 states, 0 B KV | private 2 | shared 4 | anchors 2
2026-09-13 16:48:19.032  INFO  warmup complete | 300 ms
2026-09-13 16:48:19.032  INFO  listening on http://127.0.0.1:30000 | model qwen-local | auth disabled
2026-09-13 16:51:14.271  INFO  req#1 started | openai-chat stream | 3 messages | max output 32,000 | thinking low
2026-09-13 16:51:17.433  INFO  req#1 done | openai-chat | stop token | prompt 584 | output 188 | cache 0 (0.0%) | TTFT 902 ms | total 3.2s | prefill 653.2 tok/s | decode 82.8 tok/s | dflash2 accepted 137/204 (67.2%)
2026-09-13 16:51:17.916  INFO  req#2 started | openai-chat stream | 2 messages | max output 32,000 | thinking medium | tools 11
2026-09-13 16:51:19.048  INFO  throughput | 5.0s | prefill 321.0 tok/s (1,608 tok) | decode 37.3 tok/s (187 tok) | running 1 (prefill 1) | batch 1.00 | host 36.0% (1.8s)
2026-09-13 16:51:24.041  INFO  throughput | 5.0s | prefill 1.43k tok/s (7,125 tok) | running 1 (prefill 1) | host 106.4% (5.3s)
2026-09-13 16:51:29.041  INFO  throughput | 5.0s | prefill 1.0 tok/s (5 tok) | decode 58.0 tok/s (290 tok) | running 1 (decode-ready 1) | batch 1.00 | host 8.7% (434 ms)
2026-09-13 16:51:34.039  INFO  throughput | 5.0s | decode 62.4 tok/s (312 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.7% (233 ms)
2026-09-13 16:51:39.043  INFO  throughput | 5.0s | decode 81.7 tok/s (409 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.7% (236 ms)
2026-09-13 16:51:44.042  INFO  throughput | 5.0s | decode 82.2 tok/s (411 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.4% (222 ms)
2026-09-13 16:51:49.043  INFO  throughput | 5.0s | decode 91.6 tok/s (458 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.5% (224 ms)
2026-09-13 16:51:54.044  INFO  throughput | 5.0s | decode 91.8 tok/s (459 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.6% (230 ms)
2026-09-13 16:51:59.048  INFO  throughput | 5.0s | decode 94.3 tok/s (472 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.5% (226 ms)
2026-09-13 16:52:04.040  INFO  throughput | 5.0s | decode 93.9 tok/s (469 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.3% (214 ms)
2026-09-13 16:52:09.037  INFO  throughput | 5.0s | decode 85.1 tok/s (425 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.6% (231 ms)
2026-09-13 16:52:14.039  INFO  throughput | 5.0s | decode 90.6 tok/s (453 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.7% (236 ms)
2026-09-13 16:52:19.048  INFO  throughput | 5.0s | decode 89.4 tok/s (448 tok) | running 1 (decode-ready 1) | batch 1.00 | host 4.4% (221 ms)
2026-09-13 16:52:23.696  INFO  req#2 done | openai-chat | tool calls 1 | prompt 8,154 | output 5,028 | cache 0 (0.0%) | TTFT 6.3s | total 1m 5.8s | prefill 1.29k tok/s | decode 84.7 tok/s | dflash2 accepted 3,745/5,136 (72.9%)
2026-09-13 16:52:23.961  INFO  req#3 started | openai-chat stream | 4 messages | max output 32,000 | thinking medium | tools 11
2026-09-13 16:52:24.041  INFO  throughput | 5.0s | decode 84.3 tok/s (421 tok) | running 1 (prefill 1) | batch 1.00 | host 4.0% (198 ms)
2026-09-13 16:52:27.702  INFO  req#3 done | openai-chat | tool calls 1 | prompt 13,202 | output 308 | cache 13,181 (99.8%, private endpoint) | TTFT 386 ms | total 3.8s | prefill 57.4 tok/s | decode 91.2 tok/s | dflash2 accepted 235/292 (80.5%)
2026-09-13 16:52:27.909  INFO  req#4 started | openai-chat stream | 6 messages | max output 32,000 | thinking medium | tools 11
2026-09-13 16:52:29.049  INFO  throughput | 5.0s | prefill 8.2 tok/s (41 tok) | decode 72.5 tok/s (363 tok) | running 1 (decode-ready 1) | batch 1.00 | host 18.3% (915 ms)
2026-09-13 16:52:30.210  INFO  req#4 done | openai-chat | tool calls 1 | prompt 13,529 | output 139 | cache 13,509 (99.9%, private endpoint) | TTFT 373 ms | total 2.3s | prefill 56.6 tok/s | decode 71.1 tok/s | dflash2 accepted 97/168 (57.7%)
2026-09-13 16:52:33.023  INFO  req#5 started | openai-chat stream | 8 messages | max output 32,000 | thinking medium | tools 11
2026-09-13 16:52:34.047  INFO  throughput | 5.0s | prefill 19.8 tok/s (99 tok) | decode 25.4 tok/s (127 tok) | running 1 (decode-ready 1) | batch 1.00 | host 10.5% (526 ms)
2026-09-13 16:52:35.959  INFO  req#5 done | openai-chat | stop token | prompt 13,766 | output 143 | cache 13,667 (99.3%, private endpoint) | TTFT 467 ms | total 3.0s | prefill 223.7 tok/s | decode 57.1 tok/s | dflash2 accepted 89/212 (42.0%)
2026-09-13 16:52:39.039  INFO  throughput | 5.0s | decode 19.4 tok/s (97 tok) | running 0 | batch 1.00 | host 1.8% (90.4 ms)
```

### HTML

``` html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D Snake</title>
<style>
  html, body { margin: 0; height: 100%; background: #0b0e17; overflow: hidden; font-family: 'Segoe UI', system-ui, sans-serif; }
  #c { display: block; width: 100vw; height: 100vh; }
  #hud {
    position: fixed; top: 16px; left: 0; right: 0; text-align: center;
    color: #e8ecf8; pointer-events: none; user-select: none;
  }
  #score { font-size: 34px; font-weight: 700; letter-spacing: 2px; }
  #best { font-size: 14px; opacity: .7; margin-top: 4px; }
  #msg {
    position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
    flex-direction: column; color: #fff; background: rgba(8, 10, 20, .72);
  }
  #msg h1 { font-size: 46px; margin: 0 0 8px; }
  #msg p { font-size: 16px; opacity: .85; margin: 4px 0; }
  .key { display: inline-block; border: 1px solid #556; border-bottom-width: 3px; border-radius: 6px;
         padding: 2px 9px; margin: 0 2px; background: #1c2236; font-size: 14px; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <div id="score">0</div>
  <div id="best">BEST 0</div>
</div>
<div id="msg">
  <h1 id="title">3D SNAKE</h1>
  <p id="final"></p>
  <p><span class="key">&#8592;</span><span class="key">&#8593;</span><span class="key">&#8594;</span><span class="key">&#8595;</span> or WASD to move</p>
  <p>SPACE to (re)start</p>
</div>
<script>
const N = 12, CELL = 1, GROUND = 0.28;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');
const bestEl = document.getElementById('best');
const msgEl = document.getElementById('msg');
const titleEl = document.getElementById('title');
const finalEl = document.getElementById('final');

let W, H;
function resize() {
  W = canvas.width = innerWidth * devicePixelRatio;
  H = canvas.height = innerHeight * devicePixelRatio;
  canvas.style.width = innerWidth + 'px';
  canvas.style.height = innerHeight + 'px';
}
addEventListener('resize', resize);
resize();

const EYE = [N / 2, 6.5, -4.5];
const TARGET = [N / 2, 0.4, N / 2 + 0.5];
let F, R, U, FOCAL;
(function computeBasis() {
  const f = norm(sub(TARGET, EYE));
  const r = norm(cross(f, [0, 1, 0]));
  const u = cross(r, f);
  F = f; R = r; U = u;
  FOCAL = Math.min(W, H) * 1.15;
})();

function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm(a) { const l = Math.hypot(a[0], a[1], a[2]); return [a[0]/l, a[1]/l, a[2]/l]; }

function project(p) {
  const v = sub(p, EYE);
  const z = v[0]*F[0] + v[1]*F[1] + v[2]*F[2];
  const x = v[0]*R[0] + v[1]*R[1] + v[2]*R[2];
  const y = v[0]*U[0] + v[1]*U[1] + v[2]*U[2];
  return [W/2 + x * FOCAL / z, H/2 - y * FOCAL / z, z];
}

function cellCenter(x, z) { return [x + 0.5, GROUND, z + 0.5]; }

let snake, dir, nextDir, food, score, best = 0, dead, tick, timer = 0, foodPulse = 0;

function reset() {
  snake = [[N/2 - 1, N/2], [N/2 - 2, N/2], [N/2 - 3, N/2]];
  dir = [1, 0];
  nextDir = [1, 0];
  score = 0;
  dead = false;
  tick = 150;
  timer = 0;
  placeFood();
  updateHud();
  msgEl.style.display = 'none';
}

function placeFood() {
  while (true) {
    const f = [rand(N), rand(N)];
    if (!snake.some(s => s[0] === f[0] && s[1] === f[1])) { food = f; return; }
  }
}
function rand(n) { return Math.floor(Math.random() * n); }

function updateHud() {
  scoreEl.textContent = score;
  bestEl.textContent = 'BEST ' + best;
}

function step() {
  dir = nextDir;
  const head = [snake[0][0] + dir[0], snake[0][1] + dir[1]];
  if (head[0] < 0 || head[0] >= N || head[1] < 0 || head[1] >= N ||
      snake.some(s => s[0] === head[0] && s[1] === head[1])) {
    dead = true;
    best = Math.max(best, score);
    updateHud();
    titleEl.textContent = 'GAME OVER';
    finalEl.textContent = 'Score: ' + score + (score === best ? '  (new best!)' : '');
    msgEl.style.display = 'flex';
    return;
  }
  snake.unshift(head);
  if (head[0] === food[0] && head[1] === food[1]) {
    score++;
    best = Math.max(best, score);
    tick = Math.max(70, tick - 4);
    placeFood();
    updateHud();
  } else {
    snake.pop();
  }
}

const DIRS = {
  ArrowUp: [0, -1], KeyW: [0, -1],
  ArrowDown: [0, 1], KeyS: [0, 1],
  ArrowLeft: [-1, 0], KeyA: [-1, 0],
  ArrowRight: [1, 0], KeyD: [1, 0]
};
addEventListener('keydown', e => {
  if (e.code === 'Space') { reset(); return; }
  const d = DIRS[e.code];
  if (!d) return;
  e.preventDefault();
  if (dead) return;
  if (d[0] === -dir[0] && d[1] === -dir[1]) return;
  nextDir = d;
});

function drawGrid() {
  ctx.lineWidth = Math.max(1, W / 900);
  ctx.strokeStyle = 'rgba(120, 140, 200, 0.16)';
  for (let i = 0; i <= N; i++) {
    let a = project([i, GROUND, 0]), b = project([i, GROUND, N]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    a = project([0, GROUND, i]); b = project([N, GROUND, i]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(140, 165, 235, 0.55)';
  ctx.lineWidth = Math.max(2, W / 400);
  const c = [
    project([0, GROUND, 0]), project([N, GROUND, 0]),
    project([N, GROUND, N]), project([0, GROUND, N])
  ];
  ctx.beginPath();
  c.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.closePath();
  ctx.stroke();
}

function drawCube(cx, cz, h, top, sideA, sideB) {
  const pts = [
    [cx - 0.48, GROUND, cz - 0.48], [cx + 0.48, GROUND, cz - 0.48],
    [cx + 0.48, GROUND, cz + 0.48], [cx - 0.48, GROUND, cz + 0.48],
    [cx - 0.48, h, cz - 0.48], [cx + 0.48, h, cz - 0.48],
    [cx + 0.48, h, cz + 0.48], [cx - 0.48, h, cz + 0.48]
  ];
  const p = pts.map(project);
  const faces = [
    { v: [p[4], p[5], p[6], p[7]], c: top },
    { v: [p[0], p[1], p[5], p[4]], c: sideA },
    { v: [p[1], p[2], p[6], p[5]], c: sideB },
    { v: [p[2], p[3], p[7], p[6]], c: sideB },
    { v: [p[3], p[0], p[4], p[7]], c: sideA }
  ];
  for (const f of faces) {
    const z = (f.v[0][2] + f.v[1][2] + f.v[2][2] + f.v[3][2]) / 4;
    const back = (f.v[1][2] + f.v[2][2]) / 2 < (f.v[0][2] + f.v[3][2]) / 2;
    f.depth = z;
    f.skip = back;
  }
  faces.sort((a, b) => b.depth - a.depth);
  for (const f of faces) {
    if (f.skip) continue;
    ctx.fillStyle = f.c;
    ctx.beginPath();
    f.v.forEach((pt, i) => i ? ctx.lineTo(pt[0], pt[1]) : ctx.moveTo(pt[0], pt[1]));
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawFood(t) {
  const [fx, fz] = food;
  const bob = Math.sin(t / 300) * 0.12 + 0.75;
  const c = project([fx + 0.5, bob, fz + 0.5]);
  const s = FOCAL / c[2] * 0.30 * (1 + Math.sin(t / 250) * 0.12);
  ctx.fillStyle = 'rgba(255, 70, 90, 0.25)';
  ctx.beginPath();
  ctx.ellipse(c[0], c[1], s * 1.6, s * 1.6, 0, 0, Math.PI * 2);
  ctx.fill();
  const g = ctx.createRadialGradient(c[0] - s*0.35, c[1] - s*0.35, s*0.1, c[0], c[1], s);
  g.addColorStop(0, '#ff9aa8');
  g.addColorStop(0.5, '#ff4d66');
  g.addColorStop(1, '#c21f3a');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(c[0], c[1], s, 0, Math.PI * 2);
  ctx.fill();
  const sh = project([fx + 0.5, GROUND + 0.01, fz + 0.5]);
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.beginPath();
  ctx.ellipse(sh[0], sh[1], s * 0.9, s * 0.32, 0, 0, Math.PI * 2);
  ctx.fill();
}

function render(t) {
  ctx.fillStyle = '#0b0e17';
  ctx.fillRect(0, 0, W, H);
  const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.7);
  grad.addColorStop(0, 'rgba(40, 55, 110, 0.25)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  drawGrid();

  const cubes = [];
  snake.forEach((s, i) => {
    const h = i === 0 ? 0.95 : 0.8;
    cubes.push({ x: s[0], z: s[1], i, h });
  });
  cubes.sort((a, b) => {
    const za = sub([a.x + 0.5, 0, a.z + 0.5], EYE);
    const zb = sub([b.x + 0.5, 0, b.z + 0.5], EYE);
    return (zb[0]*F[0] + zb[1]*F[1] + zb[2]*F[2]) - (za[0]*F[0] + za[1]*F[1] + za[2]*F[2]);
  });

  for (const c of cubes) {
    const head = c.i === 0;
    if (head) {
      const glow = 0.5 + Math.sin(t / 200) * 0.2;
      drawCube(c.x, c.z, c.h, `rgba(${Math.round(120*glow+80)}, ${Math.round(255*glow+40)}, ${Math.round(160*glow+60)},1)`,
               `rgba(${Math.round(90*glow+50)}, ${Math.round(190*glow+30)}, ${Math.round(120*glow+40)},1)`,
               `rgba(${Math.round(60*glow+30)}, ${Math.round(150*glow+20)}, ${Math.round(90*glow+30)},1)`);
    } else {
      const k = 1 - c.i / Math.max(snake.length, 1) * 0.45;
      drawCube(c.x, c.z, c.h,
        `rgba(${Math.round(70*k+30)}, ${Math.round(160*k+30)}, ${Math.round(230*k+30)},1)`,
        `rgba(${Math.round(45*k+20)}, ${Math.round(110*k+20)}, ${Math.round(170*k+25)},1)`,
        `rgba(${Math.round(35*k+15)}, ${Math.round(85*k+15)}, ${Math.round(130*k+20)},1)`);
    }
  }

  drawFood(t);

  const hp = project(cellCenter(snake[0][0], snake[0][1]));
  if (hp[2] > 0.1) {
    const s = FOCAL / hp[2] * 0.07;
    ctx.fillStyle = '#111';
    const off = s * 1.2;
    ctx.beginPath();
    ctx.arc(hp[0] - off * 0.7, hp[1] - FOCAL/hp[2]*0.35, s, 0, Math.PI*2);
    ctx.arc(hp[0] + off * 0.7, hp[1] - FOCAL/hp[2]*0.35, s, 0, Math.PI*2);
    ctx.fill();
  }
}

let last = performance.now();
function loop(now) {
  const dt = now - last;
  last = now;
  if (!dead) {
    timer += dt;
    while (timer >= tick) { timer -= tick; step(); }
  }
  render(now);
  requestAnimationFrame(loop);
}

reset();
msgEl.style.display = 'flex';
requestAnimationFrame(loop);
</script>
</body>
</html>
```