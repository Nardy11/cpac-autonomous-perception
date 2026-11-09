// YOLOv8 ONNX inference in the browser: letterbox -> model -> decode [1, 4+C, 8400] -> NMS -> draw.
const MODELS = {
  custom: { url: "model/yolov8n_custom.onnx", names: ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "stop sign", "cat", "dog", "fence", "cone"] },
  coco: { url: "model/yolov8n_coco.onnx", names: ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"] },
};
let NAMES = MODELS.custom.names;
// colour follows the class name, so a class keeps its colour in both models
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const colorOf = (cls) => { let h = 0; for (const ch of NAMES[cls]) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return PALETTE[h % PALETTE.length]; };
const SIZE = 640, IOU = 0.45;
const SAMPLES = ["samples/auckland_traffic.jpg", "samples/highway_traffic.jpg", "samples/roadworks.jpg"];

const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let session = null, currentImg = null, lastRaw = null, lastMs = 0;

function setStatus(text, state) {
  $("status").textContent = text;
  $("dot").className = "dot " + (state || "");
}

// theme toggle (remembered per viewer; storage may be unavailable)
(function () {
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  $("theme").onclick = () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
  };
})();

function preprocess(img) {
  const s = Math.min(SIZE / img.width, SIZE / img.height);
  const w = Math.round(img.width * s), h = Math.round(img.height * s);
  const dx = Math.floor((SIZE - w) / 2), dy = Math.floor((SIZE - h) / 2);
  const c = document.createElement("canvas"); c.width = SIZE; c.height = SIZE;
  const g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, SIZE, SIZE);
  g.drawImage(img, dx, dy, w, h);
  const px = g.getImageData(0, 0, SIZE, SIZE).data;
  const area = SIZE * SIZE, data = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    data[i] = px[i * 4] / 255; data[i + area] = px[i * 4 + 1] / 255; data[i + 2 * area] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]), s, dx, dy };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
}

function decode(out, meta, conf) {
  const [, ch, n] = out.dims, d = out.data, nc = ch - 4, boxes = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) { const v = d[(4 + c) * n + i]; if (v > best) { best = v; cls = c; } }
    if (best < conf) continue;
    const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
    boxes.push({ cls, score: best,
      x1: (cx - w / 2 - meta.dx) / meta.s, y1: (cy - h / 2 - meta.dy) / meta.s,
      x2: (cx + w / 2 - meta.dx) / meta.s, y2: (cy + h / 2 - meta.dy) / meta.s });
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (!keep.some((k) => k.cls === b.cls && iou(k, b) > IOU)) keep.push(b);
  return keep;
}

function draw(img, dets) {
  cv.width = img.width; cv.height = img.height;
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(img.width / 320));
  const fs = Math.max(12, Math.round(img.width / 45));
  ctx.font = `600 ${fs}px ui-sans-serif, system-ui, sans-serif`;
  for (const b of dets) {
    const col = colorOf(b.cls);
    ctx.strokeStyle = col; ctx.lineWidth = lw;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const label = `${NAMES[b.cls]} ${(b.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 8, th = fs + 6;
    const ty = b.y1 - th < 0 ? b.y1 : b.y1 - th;
    ctx.fillStyle = col; ctx.fillRect(b.x1 - lw / 2, ty, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(label, b.x1 + 4 - lw / 2, ty + fs);
  }
  $("drop").style.display = "none";
  $("m-n").textContent = dets.length;
  $("m-ms").textContent = Math.round(lastMs);
  const ul = $("dets"); ul.innerHTML = "";
  if (!dets.length) { ul.innerHTML = '<li class="muted" style="background:none">No objects above the threshold.</li>'; return; }
  for (const b of dets) {
    const li = document.createElement("li");
    li.innerHTML = `<span><i class="sw" style="background:${colorOf(b.cls)}"></i>${NAMES[b.cls]}</span><span>${(b.score * 100).toFixed(1)}%</span>`;
    ul.appendChild(li);
  }
}

async function run(img) {
  currentImg = img;
  if (!session) { setStatus("Model still loading…", "busy"); return; }
  setStatus("Running…", "busy");
  const meta = preprocess(img);
  const t0 = performance.now();
  const res = await session.run({ [session.inputNames[0]]: meta.tensor });
  lastMs = performance.now() - t0;
  lastRaw = { out: res[session.outputNames[0]], meta };
  render();
}

function render() {
  if (!lastRaw || !currentImg) return;
  const dets = decode(lastRaw.out, lastRaw.meta, parseFloat($("conf").value));
  draw(currentImg, dets);
  setStatus(`Done · ${dets.length} object(s) · ${Math.round(lastMs)} ms on WASM CPU`, "ok");
}

function loadUrl(url) {
  return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = url; });
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  document.querySelectorAll(".samples button").forEach((b) => b.setAttribute("aria-pressed", "false"));
  const url = URL.createObjectURL(file);
  loadUrl(url).then(run).finally(() => URL.revokeObjectURL(url));
}

$("file").onchange = (e) => loadFile(e.target.files[0]);
$("conf").oninput = (e) => { $("confv").textContent = (+e.target.value).toFixed(2); render(); };
const box = $("box");
box.addEventListener("dragover", (e) => { e.preventDefault(); box.classList.add("dragging"); });
box.addEventListener("dragleave", () => box.classList.remove("dragging"));
box.addEventListener("drop", (e) => { e.preventDefault(); box.classList.remove("dragging"); loadFile(e.dataTransfer.files[0]); });

SAMPLES.forEach((src, i) => {
  const b = document.createElement("button");
  b.type = "button"; b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
  b.setAttribute("aria-label", `Sample frame ${i + 1}`);
  b.innerHTML = `<img src="${src}" alt="">`;
  b.onclick = () => {
    document.querySelectorAll(".samples button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", "true");
    loadUrl(src).then(run);
  };
  $("samples").appendChild(b);
});

const sessions = {};
async function useModel(key) {
  document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.model === key)));
  session = null;
  setStatus("Loading model (~12 MB)…", "busy");
  try {
    sessions[key] = sessions[key] || await ort.InferenceSession.create(MODELS[key].url, { executionProviders: ["wasm"] });
    session = sessions[key];
    NAMES = MODELS[key].names;
    setStatus("Model ready.", "ok");
    run(currentImg || await loadUrl(SAMPLES[0]));
  } catch (e) {
    console.error(e);
    setStatus("Could not load the model: " + e.message, "err");
  }
}
document.querySelectorAll(".seg button").forEach((b) => { b.onclick = () => useModel(b.dataset.model); });
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
useModel(new URLSearchParams(location.search).get("model") === "coco" ? "coco" : "custom");
