import { chromium } from "playwright";
import fs from "fs/promises";

const [,, inputPath, outputPath, heading] = process.argv;
if (!inputPath || !outputPath) {
  console.error("Usage: node capture_text_figure.mjs <inputPath> <outputPath> [heading]");
  process.exit(1);
}

const text = await fs.readFile(inputPath, "utf8");
const escapeHtml = (s) => s
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;");

const html = `<!doctype html>
<html><head><meta charset="utf-8" />
<style>
body{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; margin:0; background:#0f172a; color:#e2e8f0;}
.wrap{padding:24px;}
h1{font-size:20px; margin:0 0 16px; color:#93c5fd; font-family: Inter, system-ui, sans-serif;}
pre{white-space:pre-wrap; word-break:break-word; background:#111827; border:1px solid #334155; border-radius:12px; padding:16px; font-size:14px; line-height:1.45;}
</style></head>
<body><div class="wrap"><h1>${escapeHtml(heading || "Figure")}</h1><pre>${escapeHtml(text)}</pre></div></body></html>`;

const browser = await chromium.launch({
  headless: true,
  executablePath: "/home/adnan/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome",
  args: ["--no-sandbox", "--disable-setuid-sandbox"]
});
const page = await browser.newPage({ viewport: { width: 1400, height: 1800 } });
await page.setContent(html, { waitUntil: "domcontentloaded" });
await page.screenshot({ path: outputPath, fullPage: true });
await browser.close();
console.log(outputPath);
