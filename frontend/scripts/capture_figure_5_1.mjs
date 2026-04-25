import { chromium } from "playwright";

const executablePath = "/home/adnan/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome";
const outputPath = "/home/adnan/waasi/fyproject/projectpics/figure_5_1_pipeline_progress.png";

const browser = await chromium.launch({
  headless: true,
  executablePath,
  args: ["--no-sandbox", "--disable-setuid-sandbox"]
});

const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto("http://127.0.0.1:5173", { waitUntil: "networkidle" });
await page.fill("textarea", "generate fibonacci series for n terms");
await page.click('button:has-text("Generate")');
await page.waitForTimeout(2500);
await page.screenshot({ path: outputPath, fullPage: true });
await browser.close();
console.log(outputPath);
