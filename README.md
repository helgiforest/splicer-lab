# Splicer Lab

**A fast, fully local photo editor** — library, non-destructive develop tools, precision crop, clone stamp retouching, and size-targeted export. One app, no bloat, nothing leaves your machine.

![Splicer Lab](https://img.shields.io/badge/version-3.1.0-f3c34f) ![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-1c1f26)

---

## Features

- **Photo library** that mirrors your real folders — import once and it stays in sync with disk, even while the app is closed. Filter by rating, an "Edited" view, or folder; multi-select for batch export or reset. Removing a folder from the library (right-click it in the sidebar) never touches the files on disk. Originals are **never modified**; every edit is non-destructive.
- **Develop module**: Temperature, Tint, Saturation, Exposure, Contrast, Highlights, Shadows, Black/White point — all live, all reversible. A real-time histogram reflects the current crop and shows red clipping bars when the black or white point pushes pixels past 0% or 100%.
- **Clone Stamp**: paint from any source point to duplicate or remove detail — a raw, pixel-for-pixel copy, exactly like Photoshop's real Clone Stamp, with no automatic color correction. Adjustable brush size, feather and opacity. Fully non-destructive — edit it again anytime.
- **Before/After**: hold Compare (or `\`) to see the untouched original in the exact same crop and framing, then let go to snap back to your edit.
- **Precision crop**: aspect presets (Free, Original, 1:1, 3:2, 4:3, 16:9, 4:5, flip any of them), rule-of-thirds grid, quarter-turn rotation (works from the viewer too, not just inside Crop), a fine ±45° leveling slider, deep-zoom sharpening for pixel-precise framing, full undo/redo.
- **Size-targeted export**: type "9 MB" and Splicer Lab binary-searches the encoder to land just under it. MozJPEG and high-effort WebP, with a live estimate before you save. Save reusable **export presets** (size, format, destination folder) for one-click batch export.
- **Big files welcome**: a proxy pyramid keeps 200-megapixel files smooth to pan and zoom, with on-demand full-resolution tiles the moment you zoom in past the proxy.
- **Wide format support**: JPEG, PNG, WebP, GIF, AVIF, BMP, TIFF, HEIC/HEIF, and camera RAW (CR2, CR3, NEF, ARW, ORF, RW2, RAF, DNG, PEF and more).
- Star ratings, an EXIF panel, dark & light themes, drag-and-drop import, a resizable filmstrip and library sidebar that remember their size.

---

## Download & install

Grab the file for your system from the **[latest release](../../releases/latest)**.

| System | File | Notes |
|---|---|---|
| **Windows 10/11** | `Splicer Lab Setup X.X.X.exe` | Regular installer |
| **macOS — Apple Silicon** | `Splicer Lab-X.X.X-arm64.dmg` | |
| **Linux** | `Splicer Lab-X.X.X.AppImage` | No installation needed |

The app isn't code-signed with a paid certificate, so each OS shows a one-time warning on first launch — this is expected:

**Windows** — if SmartScreen appears: **More info → Run anyway**.

**macOS** — first launch: **right-click the app → Open → Open** (a plain double-click is blocked once). If it says the app *"is damaged"*, that's the quarantine flag on unsigned downloads, not real damage:
```bash
xattr -cr "/Applications/Splicer Lab.app"
```

**Linux**
```bash
chmod +x "Splicer Lab"*.AppImage
./"Splicer Lab"*.AppImage
```
FUSE error? `sudo apt install libfuse2`.

---

## Usage guide

### Quick start

1. **Import a folder** — it appears in the sidebar and stays in sync with disk.
2. Click a photo to open the viewer.
3. **Crop**: press `Enter` or click Crop, pick an aspect or drag the frame, `Enter` to apply.
4. **Develop**: adjust sliders in the right-hand panel — changes are live and non-destructive. Double-click any slider to reset it to 0.
5. **Retouch**: press `S` for the Clone Stamp, drag to paint over what you want gone.
6. **Export**: set a target size in MB and watch the live estimate, or use a saved preset for one click.

### Working the canvas

- **Mouse wheel** zooms in and out on the point under your cursor.
- **Double-click** empty space to fit the whole photo/crop to the window — the fast way back out after zooming in for detail work.
- **Drag** empty space to pan. Hold **Space** for a temporary pan tool while another tool (e.g. Clone Stamp) is active.
- The **bottom filmstrip** and **left sidebar** can be resized by dragging their edge — the size is remembered next time you open the app.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Apply current edits (crop + develop + clone stamp) |
| `Esc` | Apply & return to library — or clear the library's photo selection, if one is active |
| `←` `→` | Previous / next photo |
| `[` `]` | Rotate 90° — works in the viewer and in Crop |
| `\` (hold) | Compare with the original |
| `S` | Open the Clone Stamp tool |
| `[` `]`, `+` `-` | Shrink / grow the clone stamp brush (while the tool is open) |
| Double-click a stroke | Delete that clone stamp stroke |
| `Delete` / `Backspace` | Delete the selected clone stamp stroke |
| `Space` (hold) | Temporary pan tool |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| `1`–`5`, `0` | Rate photo / clear rating |
| `Ctrl+Click`, `Shift+Click` | Select photos / select a range (library) |

All of the above work by physical key position, so they land in the same place on a Cyrillic (or any other) keyboard layout, not just US English.

### Good to know

- Removing a folder from the library **never deletes files from disk** — there's an Undo in the toast either way.
- A deep-zoomed view sharpens to true 1:1 pixels a moment after you stop zooming — that's the full-resolution tile catching up with the proxy.
- If export says *"Maximum quality reached"*, your size limit is generous enough that quality 100 already fits under it.

---

## Building from source

Requires [Node.js](https://nodejs.org) 18+.

```bash
git clone https://github.com/helgiforest/splicer-lab.git
cd splicer-lab
npm install
npm start          # build & run
npm run dist       # create the installer for your OS in release/
```

Tech: Electron + React + Vite; sharp/libvips with MozJPEG for encoding; dcraw (in a worker thread) for RAW development.
