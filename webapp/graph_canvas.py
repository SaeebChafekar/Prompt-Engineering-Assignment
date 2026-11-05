"""Interactive HTML5 Canvas visualization for the knowledge graph.

Zero external dependencies: runs completely offline in vanilla HTML/JavaScript.
"""

from __future__ import annotations

import json
from typing import Any


def render_interactive_graph_html(graph_data: dict[str, Any], height: int = 560) -> str:
    """Generate self-contained HTML/JS for an interactive force-directed graph canvas."""
    data_json = json.dumps(graph_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    overflow: hidden;
    height: {height}px;
    width: 100%;
    position: relative;
    user-select: none;
  }}
  #canvas-container {{
    width: 100%;
    height: 100%;
    position: relative;
  }}
  canvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }}
  canvas:active {{ cursor: grabbing; }}
  .toolbar {{
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 10;
    display: flex;
    gap: 8px;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 6px 10px;
  }}
  .btn {{
    background: rgba(30, 41, 59, 0.9);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn:hover {{
    background: #334155;
    color: #fff;
    border-color: #64748b;
  }}
  .legend {{
    position: absolute;
    bottom: 10px;
    left: 10px;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 11px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}
  .tooltip {{
    position: absolute;
    z-index: 20;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.4;
    color: #f1f5f9;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    pointer-events: none;
    display: none;
    max-width: 320px;
  }}
  .tooltip-title {{
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 2px;
    word-break: break-all;
  }}
  .tooltip-meta {{
    font-size: 11px;
    color: #94a3b8;
  }}
</style>
</head>
<body>
<div id="canvas-container">
  <div class="toolbar">
    <button class="btn" id="btn-reset">Reset View</button>
    <button class="btn" id="btn-physics">Pause Physics</button>
    <button class="btn" id="btn-zoom-in">+</button>
    <button class="btn" id="btn-zoom-out">-</button>
  </div>
  <div class="legend">
    <div class="legend-item"><span class="dot" style="background:#10b981;"></span> Function</div>
    <div class="legend-item"><span class="dot" style="background:#8b5cf6;"></span> Class</div>
    <div class="legend-item"><span class="dot" style="background:#0ea5e9;"></span> File</div>
    <div class="legend-item"><span class="dot" style="background:#f59e0b;"></span> DocSection</div>
    <div class="legend-item"><span class="dot" style="background:#64748b;"></span> External</div>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <canvas id="graph-canvas"></canvas>
</div>

<script>
(function() {{
  const rawData = {data_json};
  const canvas = document.getElementById("graph-canvas");
  const ctx = canvas.getContext("2d");
  const container = document.getElementById("canvas-container");
  const tooltip = document.getElementById("tooltip");

  const TYPE_COLORS = {{
    "Function": "#10b981",
    "Class": "#8b5cf6",
    "File": "#0ea5e9",
    "DocSection": "#f59e0b",
    "External": "#64748b"
  }};

  let width = container.clientWidth || 800;
  let height = container.clientHeight || {height};
  canvas.width = width * window.devicePixelRatio;
  canvas.height = height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  let transform = {{ x: width / 2, y: height / 2, k: 1.0 }};
  let physicsRunning = true;

  // Build node and edge models
  const nodeMap = new Map();
  const nodes = rawData.nodes.map((n, i) => {{
    const angle = (i / Math.max(1, rawData.nodes.length)) * Math.PI * 2;
    const radius = 100 + Math.random() * 140;
    const node = {{
      id: n.id,
      label: n.label,
      type: n.type,
      path: n.path || "",
      lines: n.lines || "",
      is_seed: n.is_seed || false,
      color: TYPE_COLORS[n.type] || "#94a3b8",
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      vx: 0,
      vy: 0,
      radius: n.is_seed ? 10 : (n.type === "File" ? 9 : 7)
    }};
    nodeMap.set(n.id, node);
    return node;
  }});

  const edges = rawData.edges.map(e => ({{
    source: nodeMap.get(e.source),
    target: nodeMap.get(e.target),
    type: e.type,
    confidence: e.confidence,
    details: e.details || ""
  }})).filter(e => e.source && e.target);

  let selectedNode = null;
  let hoveredNode = null;
  let draggingNode = null;
  let isPanning = false;
  let panStart = {{ x: 0, y: 0 }};

  // Force simulation loop
  function tick() {{
    if (physicsRunning) {{
      const springLength = 80;
      const springK = 0.04;
      const repulsion = 450;
      const centerGravity = 0.012;
      const damping = 0.88;

      // Node repulsion
      for (let i = 0; i < nodes.length; i++) {{
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {{
          const b = nodes[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < 300) {{
            let force = repulsion / (dist * dist);
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            a.vx -= fx;
            a.vy -= fy;
            b.vx += fx;
            b.vy += fy;
          }}
        }}
      }}

      // Edge spring attraction
      for (let i = 0; i < edges.length; i++) {{
        const e = edges[i];
        let dx = e.target.x - e.source.x;
        let dy = e.target.y - e.source.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = (dist - springLength) * springK;
        let fx = (dx / dist) * force;
        let fy = (dy / dist) * force;
        e.source.vx += fx;
        e.source.vy += fy;
        e.target.vx -= fx;
        e.target.vy -= fy;
      }}

      // Center gravity & update position
      for (let i = 0; i < nodes.length; i++) {{
        const n = nodes[i];
        if (n === draggingNode) continue;
        n.vx -= n.x * centerGravity;
        n.vy -= n.y * centerGravity;
        n.vx *= damping;
        n.vy *= damping;
        n.x += n.vx;
        n.y += n.vy;
      }}
    }}

    draw();
    requestAnimationFrame(tick);
  }}

  function toScreen(x, y) {{
    return {{
      x: x * transform.k + transform.x,
      y: y * transform.k + transform.y
    }};
  }}

  function toWorld(x, y) {{
    return {{
      x: (x - transform.x) / transform.k,
      y: (y - transform.y) / transform.k
    }};
  }}

  function draw() {{
    ctx.clearRect(0, 0, width, height);

    // Connected neighbors set for highlight
    const connectedNodeIds = new Set();
    if (selectedNode) {{
      connectedNodeIds.add(selectedNode.id);
      for (const e of edges) {{
        if (e.source === selectedNode) connectedNodeIds.add(e.target.id);
        if (e.target === selectedNode) connectedNodeIds.add(e.source.id);
      }}
    }}

    // Draw Edges
    for (let i = 0; i < edges.length; i++) {{
      const e = edges[i];
      const s = toScreen(e.source.x, e.source.y);
      const t = toScreen(e.target.x, e.target.y);

      let alpha = 0.45;
      if (selectedNode) {{
        alpha = (e.source === selectedNode || e.target === selectedNode) ? 0.9 : 0.08;
      }}

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);

      ctx.strokeStyle = `rgba(148, 163, 184, ${{alpha}})`;
      ctx.lineWidth = e.confidence === "certain" ? 1.6 : 1.0;
      if (e.confidence === "guess") {{
        ctx.setLineDash([3, 3]);
      }} else if (e.confidence === "likely") {{
        ctx.setLineDash([6, 3]);
      }}
      ctx.stroke();

      // Arrow head pointing to target
      const angle = Math.atan2(t.y - s.y, t.x - s.x);
      const targetRadius = (e.target.radius + 3) * transform.k;
      const arrowX = t.x - Math.cos(angle) * targetRadius;
      const arrowY = t.y - Math.sin(angle) * targetRadius;

      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(arrowX - 7 * Math.cos(angle - Math.PI / 6), arrowY - 7 * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(arrowX - 7 * Math.cos(angle + Math.PI / 6), arrowY - 7 * Math.sin(angle + Math.PI / 6));
      ctx.fillStyle = `rgba(148, 163, 184, ${{alpha}})`;
      ctx.fill();
      ctx.restore();
    }}

    // Draw Nodes
    for (let i = 0; i < nodes.length; i++) {{
      const n = nodes[i];
      const pt = toScreen(n.x, n.y);
      const isConnected = !selectedNode || connectedNodeIds.has(n.id);
      const alpha = isConnected ? 1.0 : 0.15;

      ctx.save();
      // Outer halo for seed or selected
      if (n.is_seed || n === selectedNode) {{
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, (n.radius + 5) * transform.k, 0, Math.PI * 2);
        ctx.fillStyle = n === selectedNode ? "rgba(56, 189, 248, 0.35)" : "rgba(16, 185, 129, 0.25)";
        ctx.fill();
      }}

      // Node circle
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, n.radius * transform.k, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.strokeStyle = n === selectedNode ? "#38bdf8" : "#0f172a";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label text
      if (transform.k > 0.6 || n.is_seed || n === selectedNode) {{
        ctx.font = `${{Math.max(10, Math.min(12, 11 * transform.k))}}px -apple-system, sans-serif`;
        ctx.fillStyle = isConnected ? "#f1f5f9" : "rgba(148, 163, 184, 0.2)";
        ctx.textAlign = "center";
        const shortName = n.label.split("::").pop();
        ctx.fillText(shortName, pt.x, pt.y + (n.radius + 12) * transform.k);
      }}
      ctx.restore();
    }}
  }}

  // Event handlers
  canvas.addEventListener("mousedown", (e) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const world = toWorld(mx, my);

    // Hit test node
    let hit = null;
    for (let i = nodes.length - 1; i >= 0; i--) {{
      const n = nodes[i];
      const dx = world.x - n.x;
      const dy = world.y - n.y;
      if (dx * dx + dy * dy <= n.radius * n.radius * 2) {{
        hit = n;
        break;
      }}
    }}

    if (hit) {{
      draggingNode = hit;
      selectedNode = (selectedNode === hit) ? null : hit;
    }} else {{
      isPanning = true;
      panStart = {{ x: mx - transform.x, y: my - transform.y }};
      selectedNode = null;
    }}
  }});

  window.addEventListener("mousemove", (e) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (draggingNode) {{
      const world = toWorld(mx, my);
      draggingNode.x = world.x;
      draggingNode.y = world.y;
      draggingNode.vx = 0;
      draggingNode.vy = 0;
    }} else if (isPanning) {{
      transform.x = mx - panStart.x;
      transform.y = my - panStart.y;
    }} else {{
      // Hover test
      const world = toWorld(mx, my);
      let hit = null;
      for (let i = nodes.length - 1; i >= 0; i--) {{
        const n = nodes[i];
        const dx = world.x - n.x;
        const dy = world.y - n.y;
        if (dx * dx + dy * dy <= n.radius * n.radius * 2) {{
          hit = n;
          break;
        }}
      }}
      hoveredNode = hit;
      if (hit) {{
        tooltip.style.display = "block";
        tooltip.style.left = (mx + 15) + "px";
        tooltip.style.top = (my + 15) + "px";
        tooltip.innerHTML = `
          <div class="tooltip-title">${{hit.label}}</div>
          <div class="tooltip-meta"><b>Type:</b> ${{hit.type}}</div>
          ${{hit.path ? `<div class="tooltip-meta"><b>Path:</b> ${{hit.path}}</div>` : ""}}
          ${{hit.lines ? `<div class="tooltip-meta"><b>Lines:</b> ${{hit.lines}}</div>` : ""}}
        `;
      }} else {{
        tooltip.style.display = "none";
      }}
    }}
  }});

  window.addEventListener("mouseup", () => {{
    draggingNode = null;
    isPanning = false;
  }});

  canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const newK = Math.max(0.2, Math.min(3.5, transform.k * zoomFactor));
    transform.x = mx - (mx - transform.x) * (newK / transform.k);
    transform.y = my - (my - transform.y) * (newK / transform.k);
    transform.k = newK;
  }}, {{ passive: false }});

  // Buttons
  document.getElementById("btn-reset").addEventListener("click", () => {{
    transform = {{ x: width / 2, y: height / 2, k: 1.0 }};
    selectedNode = null;
  }});

  const btnPhys = document.getElementById("btn-physics");
  btnPhys.addEventListener("click", () => {{
    physicsRunning = !physicsRunning;
    btnPhys.textContent = physicsRunning ? "Pause Physics" : "Resume Physics";
  }});

  document.getElementById("btn-zoom-in").addEventListener("click", () => {{
    transform.k = Math.min(3.5, transform.k * 1.25);
  }});

  document.getElementById("btn-zoom-out").addEventListener("click", () => {{
    transform.k = Math.max(0.2, transform.k * 0.8);
  }});

  // Start simulation
  requestAnimationFrame(tick);
}})();
</script>
</body>
</html>"""

