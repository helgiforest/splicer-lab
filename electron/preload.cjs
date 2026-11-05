const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("meridian", {
  pickFilesOrFolder: () => ipcRenderer.invoke("pick-files-or-folder"),
  importPaths: (paths) => ipcRenderer.invoke("import-paths", paths),
  watchRoots: (roots) => ipcRenderer.invoke("watch-roots", roots),
  unwatchRoot: (root) => ipcRenderer.invoke("unwatch-root", root),
  loadLibrary: () => ipcRenderer.invoke("load-library"),
  saveLibrary: (items, settings) => ipcRenderer.invoke("save-library", items, settings),
  cacheEditPreview: (filePath, dataUrl) => ipcRenderer.invoke("cache-edit-preview", filePath, dataUrl),
  exportFiles: (files) => ipcRenderer.invoke("export-files", files),
  estimatePhoto: (item, opts) => ipcRenderer.invoke("estimate-photo", item, opts),
  exportPhotos: (items, opts) => ipcRenderer.invoke("export-photos", items, opts),
  pickDir: () => ipcRenderer.invoke("pick-dir"),
  renderTile: (p, t) => ipcRenderer.invoke("render-tile", p, t),
  readExif: (p) => ipcRenderer.invoke("read-exif", p),
  logError: (message) => ipcRenderer.invoke("log-error", message),
  // local neural inpainting for the stamp tool's Remove mode (MI-GAN, runs
  // in the main process — electron/neuralFill.mjs); rgb/hole are Uint8Arrays
  neuralInpaint: (job) => ipcRenderer.invoke("neural-inpaint", job),
  onExportProgress: (cb) => {
    const handler = (_e, payload) => cb(payload);
    ipcRenderer.on("export-progress", handler);
    return () => ipcRenderer.removeListener("export-progress", handler);
  },
  // resolves the absolute path of a dropped File object
  getPathForFile: (file) => webUtils.getPathForFile(file),
  // fires whenever a watched folder changes on disk
  onFolderScan: (cb) => {
    const handler = (_e, payload) => cb(payload);
    ipcRenderer.on("folder-scan", handler);
    return () => ipcRenderer.removeListener("folder-scan", handler);
  },
});
