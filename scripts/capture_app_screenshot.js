// Capture a screenshot of a running EpiGrade app page with headless Edge or Chrome, through the
// DevTools protocol (so it can wait for Streamlit to finish rendering). Needs Node 22+ and no
// npm packages.
//
//   streamlit run src/epigrade/app/main.py --server.port 8765 --theme.base light
//   node scripts/capture_app_screenshot.js <url> <out.png> [width] [height] [wait_ms]
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const [url, out, width = "1500", height = "1900", waitMs = "25000"] = process.argv.slice(2);
if (!url || !out) {
  console.error("usage: node capture_app_screenshot.js <url> <out.png> [width] [height] [wait_ms]");
  process.exit(2);
}
const candidates = [
  process.env.BROWSER,
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium",
].filter((p) => p && fs.existsSync(p));
if (!candidates.length) {
  console.error("no Edge/Chrome found; set BROWSER to the executable");
  process.exit(2);
}

const port = 9333;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "epigrade-shot-"));
const browser = spawn(candidates[0], [
  "--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, `--window-size=${width},${height}`, "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  let target;
  for (let i = 0; i < 40 && !target; i += 1) {
    await sleep(500);
    try {
      const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      target = pages.find((p) => p.type === "page");
    } catch { /* browser still starting */ }
  }
  if (!target) throw new Error("browser did not start");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg.result); pending.delete(msg.id); }
  };
  const send = (method, params = {}) => new Promise((resolve) => {
    id += 1;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: Number(width), height: Number(height), deviceScaleFactor: 1.5, mobile: false,
  });
  await send("Page.navigate", { url });
  await sleep(Number(waitMs));
  const shot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(out, Buffer.from(shot.data, "base64"));
  console.log(`wrote ${out}`);
  ws.close();
}

main().catch((e) => { console.error(e.message); process.exitCode = 1; })
  .finally(() => { browser.kill(); });
