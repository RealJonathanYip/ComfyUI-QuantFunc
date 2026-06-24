import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// QuantFunc Layer Viewer — interactive viewer for QwenImageLayered's N RGBA layers.
// Backend node (QuantFuncLayerViewer) saves each layer as an RGBA PNG to ComfyUI's
// temp dir and returns {"ui": {qf_layers: [{filename,subfolder,type}], qf_size:[W,H]}}.
// This extension renders a thumbnail list + a main canvas: click a layer to isolate it;
// default (nothing selected) shows the alpha-over composite of all layers.

function viewURL(layer) {
  const p = new URLSearchParams({
    filename: layer.filename,
    subfolder: layer.subfolder || "",
    type: layer.type || "temp",
  });
  return api.apiURL ? api.apiURL("/view?" + p.toString()) : "/view?" + p.toString();
}

function el(tag, style, props) {
  const e = document.createElement(tag);
  if (style) Object.assign(e.style, style);
  if (props) Object.assign(e, props);
  return e;
}

function drawChecker(ctx, w, h, s = 14) {
  for (let y = 0; y < h; y += s) {
    for (let x = 0; x < w; x += s) {
      ctx.fillStyle = (((x / s) | 0) + ((y / s) | 0)) % 2 === 0 ? "#6a6a6a" : "#9a9a9a";
      ctx.fillRect(x, y, s, s);
    }
  }
}

app.registerExtension({
  name: "QuantFunc.LayerViewer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "QuantFuncLayerViewer") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;

      const root = el("div", {
        display: "flex", flexDirection: "row", gap: "6px",
        width: "100%", height: "100%", minHeight: "240px",
        boxSizing: "border-box", overflow: "hidden",
        background: "#1e1e1e", borderRadius: "6px", padding: "4px",
      });
      const list = el("div", {
        display: "flex", flexDirection: "column", gap: "4px",
        width: "116px", flex: "0 0 116px", overflowY: "auto", paddingRight: "2px",
      });
      const canvasWrap = el("div", {
        flex: "1 1 auto", display: "flex", alignItems: "center",
        justifyContent: "center", overflow: "hidden", minWidth: "0",
      });
      const canvas = el("canvas", {
        maxWidth: "100%", maxHeight: "100%", borderRadius: "4px", background: "#111",
      });
      canvasWrap.appendChild(canvas);
      root.appendChild(list);
      root.appendChild(canvasWrap);

      node._qf = { root, list, canvas, layers: [], selected: -1, size: [1, 1] };

      const widget = node.addDOMWidget("qf_layer_viewer", "qf_layer_viewer", root, {
        serialize: false,
        hideOnZoom: false,
      });
      widget.computeSize = () => [node.size?.[0] || 360, 300];

      node.size = [Math.max(node.size?.[0] || 0, 380), Math.max(node.size?.[1] || 0, 360)];
    };

    nodeType.prototype._qfRender = function () {
      const s = this._qf;
      if (!s) return;
      const [W, H] = s.size;
      const cv = s.canvas;
      if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
      const ctx = cv.getContext("2d");
      ctx.clearRect(0, 0, W, H);
      drawChecker(ctx, W, H);
      const draw = (i) => {
        const L = s.layers[i];
        if (L && L.img.complete && L.img.naturalWidth) ctx.drawImage(L.img, 0, 0, W, H);
      };
      if (s.selected < 0) {
        for (let i = 0; i < s.layers.length; i++) draw(i);   // composite: bottom→top
      } else {
        draw(s.selected);                                     // isolate one layer
      }
    };

    nodeType.prototype._qfHighlight = function () {
      const s = this._qf;
      [...s.list.children].forEach((row) => {
        const on = row._idx === s.selected;
        row.style.border = on ? "1px solid #4a9eff" : "1px solid transparent";
        row.style.background = on ? "#2a3a4a" : "transparent";
      });
    };

    nodeType.prototype._qfDownload = function (idx) {
      const s = this._qf;
      if (idx >= 0) {
        const a = el("a", null, { href: s.layers[idx].url, download: `layer_${idx + 1}.png` });
        document.body.appendChild(a); a.click(); a.remove();
        return;
      }
      const [W, H] = s.size;
      const off = el("canvas", null, { width: W, height: H });
      const ctx = off.getContext("2d");
      for (let i = 0; i < s.layers.length; i++) {
        const L = s.layers[i];
        if (L && L.img.complete && L.img.naturalWidth) ctx.drawImage(L.img, 0, 0, W, H);
      }
      off.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = el("a", null, { href: url, download: "composite.png" });
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
      }, "image/png");
    };

    nodeType.prototype._qfBuildList = function () {
      const s = this._qf;
      const self = this;
      s.list.innerHTML = "";
      const addRow = (idx, label, thumbURL) => {
        const row = el("div", {
          display: "flex", alignItems: "center", gap: "4px", padding: "3px",
          borderRadius: "4px", cursor: "pointer", border: "1px solid transparent",
          fontSize: "11px", color: "#ddd",
        });
        row._idx = idx;
        const thumb = thumbURL
          ? el("img", { width: "30px", height: "30px", objectFit: "cover", borderRadius: "3px",
              background: "#333", flex: "0 0 30px" }, { src: thumbURL })
          : el("div", { width: "30px", height: "30px", display: "flex", alignItems: "center",
              justifyContent: "center", background: "#333", borderRadius: "3px", flex: "0 0 30px" },
              { textContent: "▦" });
        row.appendChild(thumb);
        row.appendChild(el("span", { flex: "1 1 auto", whiteSpace: "nowrap", overflow: "hidden",
          textOverflow: "ellipsis" }, { textContent: label }));
        const dl = el("span", { cursor: "pointer", padding: "0 4px", opacity: "0.7", fontSize: "13px" },
          { textContent: "⤓", title: "下载 / download" });
        dl.onclick = (e) => { e.stopPropagation(); self._qfDownload(idx); };
        row.appendChild(dl);
        row.onclick = () => {
          s.selected = s.selected === idx ? -1 : idx;   // toggle isolate ↔ composite
          self._qfHighlight();
          self._qfRender();
        };
        s.list.appendChild(row);
      };
      addRow(-1, "合成 (全部)", null);
      s.layers.forEach((L, i) => addRow(i, `层 ${i + 1}`, L.url));
      this._qfHighlight();
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const layers = message?.qf_layers;
      if (!layers || !layers.length) return;
      const s = this._qf;
      if (!s) return;
      s.size = message.qf_size && message.qf_size.length === 2 ? message.qf_size : [512, 512];
      s.selected = -1;
      s.layers = layers.map((ly) => {
        const url = viewURL(ly);
        const img = new Image();
        const rec = { url, img };
        img.onload = () => this._qfRender();
        img.src = url;
        return rec;
      });
      this._qfBuildList();
      this._qfRender();
    };
  },
});
