import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const dist = join(root, "dist");
const itemsToCopy = ["index.html", "style.css", "app.js", "assets", "README.md"];

if (existsSync(dist)) {
  rmSync(dist, { recursive: true, force: true });
}

mkdirSync(dist, { recursive: true });

for (const item of itemsToCopy) {
  cpSync(join(root, item), join(dist, item), { recursive: true });
}

console.log("Build completed: dist/");
