/**
 * gridpy.js — Browser-side rendering runtime for grid_py WebRenderer.
 *
 * Reads a scene graph JSON and renders it to layered SVG + Canvas.
 * Uses D3.js v7 for SVG manipulation and interactions (zoom, brush, tooltip).
 *
 * @version 0.1.0
 * @license MIT
 */
var gridpy = (function () {
    "use strict";

    // ---- Configuration ---------------------------------------------------

    var CANVAS_THRESHOLD = 2000; // Elements above this route to Canvas
    var TOOLTIP_SEARCH_RADIUS = 10; // px

    // ---- Gpar → CSS/Canvas style helpers ----------------------------------

    function parseColour(c) {
        if (!c || c === "transparent" || c === "NA" || c === "none") return null;
        return c; // CSS can handle hex, named colours, rgba()
    }

    function applyGparSvg(sel, gpar) {
        if (!gpar) return;
        var fill = parseColour(gpar.fill);
        var col = parseColour(gpar.col);
        if (fill !== undefined) sel.attr("fill", fill || "none");
        if (col !== undefined) sel.attr("stroke", col || "none");
        if (gpar.lwd !== undefined) sel.attr("stroke-width", gpar.lwd);
        if (gpar.alpha !== undefined) sel.attr("opacity", gpar.alpha);
        if (gpar.lty) sel.attr("stroke-dasharray", ltyToDash(gpar.lty));
        if (gpar.lineend) sel.attr("stroke-linecap", gpar.lineend);
        if (gpar.linejoin) sel.attr("stroke-linejoin",
            gpar.linejoin === "mitre" ? "miter" : gpar.linejoin);
    }

    function applyTextGparSvg(sel, gpar) {
        if (!gpar) return;
        var col = parseColour(gpar.col);
        if (col) sel.attr("fill", col);
        if (gpar.fontsize) sel.attr("font-size", gpar.fontsize + "px");
        if (gpar.fontfamily) sel.attr("font-family", gpar.fontfamily);
        if (gpar.fontface) {
            var face = gpar.fontface;
            if (face === "bold" || face === 2) sel.attr("font-weight", "bold");
            else if (face === "italic" || face === 3) sel.attr("font-style", "italic");
            else if (face === "bold.italic" || face === 4) {
                sel.attr("font-weight", "bold").attr("font-style", "italic");
            }
        }
        if (gpar.alpha !== undefined) sel.attr("opacity", gpar.alpha);
    }

    function applyGparCanvas(ctx, gpar) {
        if (!gpar) return;
        var fill = parseColour(gpar.fill);
        var col = parseColour(gpar.col);
        ctx.fillStyle = fill || "rgba(0,0,0,0)";
        ctx.strokeStyle = col || "rgba(0,0,0,0)";
        ctx.lineWidth = gpar.lwd || 1;
        ctx.globalAlpha = gpar.alpha !== undefined ? gpar.alpha : 1.0;
        if (gpar.lineend) ctx.lineCap = gpar.lineend;
        if (gpar.linejoin) ctx.lineJoin =
            gpar.linejoin === "mitre" ? "miter" : (gpar.linejoin || "round");
        if (gpar.lty) {
            var dash = ltyToDashArray(gpar.lty);
            ctx.setLineDash(dash || []);
        } else {
            ctx.setLineDash([]);
        }
    }

    function ltyToDash(lty) {
        var map = {
            "dashed": "6,4", "dotted": "2,2",
            "dotdash": "2,2,6,2", "longdash": "10,3",
            "twodash": "5,2,10,2", "blank": "0,100"
        };
        return map[lty] || null;
    }

    function ltyToDashArray(lty) {
        var map = {
            "dashed": [6, 4], "dotted": [2, 2],
            "dotdash": [2, 2, 6, 2], "longdash": [10, 3],
            "twodash": [5, 2, 10, 2], "blank": [0, 100]
        };
        return map[lty] || [];
    }

    function hjustToAnchor(hj) {
        if (hj <= 0.2) return "start";
        if (hj >= 0.8) return "end";
        return "middle";
    }

    function vjustToBaseline(vj) {
        if (vj <= 0.2) return "text-after-edge";
        if (vj >= 0.8) return "text-before-edge";
        return "central";
    }

    // ---- Gradient/pattern → SVG defs --------------------------------------

    function createDefs(defsSel, defsData) {
        if (!defsData) return;

        // Clip paths
        (defsData.clip_paths || []).forEach(function (cp) {
            defsSel.append("clipPath").attr("id", cp.id)
                .append("rect")
                .attr("x", cp.x).attr("y", cp.y)
                .attr("width", cp.w).attr("height", cp.h);
        });

        // Linear / radial gradients
        (defsData.gradients || []).forEach(function (g) {
            var el;
            if (g.type === "linear") {
                el = defsSel.append("linearGradient").attr("id", g.id)
                    .attr("x1", g.x1).attr("y1", g.y1)
                    .attr("x2", g.x2).attr("y2", g.y2)
                    .attr("gradientUnits", "userSpaceOnUse");
            } else {
                el = defsSel.append("radialGradient").attr("id", g.id)
                    .attr("cx", g.cx2).attr("cy", g.cy2).attr("r", g.r2)
                    .attr("fx", g.cx1).attr("fy", g.cy1).attr("fr", g.r1)
                    .attr("gradientUnits", "userSpaceOnUse");
            }
            if (g.colours && g.stops) {
                for (var i = 0; i < g.colours.length; i++) {
                    el.append("stop")
                        .attr("offset", g.stops[i])
                        .attr("stop-color", g.colours[i]);
                }
            }
        });

        // Patterns
        (defsData.patterns || []).forEach(function (p) {
            defsSel.append("pattern").attr("id", p.id)
                .attr("x", p.x || 0).attr("y", p.y || 0)
                .attr("width", p.width).attr("height", p.height)
                .attr("patternUnits", "userSpaceOnUse")
                .html(p.content || "");
        });

        // Masks
        (defsData.masks || []).forEach(function (m) {
            var mask = defsSel.append("mask").attr("id", m.id);
            if (m.content && m.content.children) {
                m.content.children.forEach(function (child) {
                    renderSvgNode(child, mask);
                });
            }
        });
    }

    // ---- Routing ----------------------------------------------------------

    function routeToLayer(node) {
        if (node.render_hint === "svg") return "svg";
        if (node.render_hint === "canvas") return "canvas";
        // Auto-routing
        if (node.type === "text") return "svg";
        if (node.type === "raster") return "svg";
        if (node.type === "points") {
            var n = node.props.x ? node.props.x.length : 0;
            return n > CANVAS_THRESHOLD ? "canvas" : "svg";
        }
        return "svg";
    }

    // ---- SVG rendering ----------------------------------------------------

    function renderSvgNode(node, parentG) {
        if (!node) return;
        if (node.type === "viewport") {
            renderSvgViewport(node, parentG);
        } else {
            drawSvgGrub(node, parentG);
        }
    }

    function renderSvgViewport(vpNode, parentG) {
        var g = parentG.append("g")
            .attr("class", "vp-" + (vpNode.name || ""))
            .attr("data-viewport", vpNode.name || "");
        if (vpNode.clip_id) {
            g.attr("clip-path", "url(#" + vpNode.clip_id + ")");
        }
        if (vpNode.mask_id) {
            g.attr("mask", "url(#" + vpNode.mask_id + ")");
        }
        (vpNode.children || []).forEach(function (child) {
            renderSvgNode(child, g);
        });
    }

    function drawSvgGrub(node, parentG) {
        var p = node.props || {};
        switch (node.type) {
            case "rect":
                parentG.append("rect")
                    .attr("x", p.x).attr("y", p.y)
                    .attr("width", p.w).attr("height", p.h)
                    .attr("data-id", node.id)
                    .call(applyGparSvg, node.gpar);
                break;

            case "roundrect":
                parentG.append("rect")
                    .attr("x", p.x).attr("y", p.y)
                    .attr("width", p.w).attr("height", p.h)
                    .attr("rx", p.r || 0).attr("ry", p.r || 0)
                    .attr("data-id", node.id)
                    .call(applyGparSvg, node.gpar);
                break;

            case "circle":
                parentG.append("circle")
                    .attr("cx", p.x).attr("cy", p.y).attr("r", p.r)
                    .attr("data-id", node.id)
                    .call(applyGparSvg, node.gpar);
                break;

            case "text":
                var txt = parentG.append("text")
                    .attr("x", p.x).attr("y", p.y)
                    .attr("text-anchor", hjustToAnchor(p.hjust || 0.5))
                    .attr("dominant-baseline", vjustToBaseline(p.vjust || 0.5))
                    .text(p.label || "")
                    .attr("data-id", node.id)
                    .call(applyTextGparSvg, node.gpar);
                if (p.rot) {
                    txt.attr("transform",
                        "rotate(" + (-p.rot) + "," + p.x + "," + p.y + ")");
                }
                break;

            case "points":
                drawSvgPoints(node, parentG);
                break;

            case "polyline":
            case "lines":
                drawSvgPolyline(node, parentG);
                break;

            case "segments":
                drawSvgSegments(node, parentG);
                break;

            case "polygon":
                drawSvgPolygon(node, parentG);
                break;

            case "path":
                drawSvgPath(node, parentG);
                break;

            case "raster":
                parentG.append("image")
                    .attr("x", p.x).attr("y", p.y)
                    .attr("width", p.w).attr("height", p.h)
                    .attr("href", p.src)
                    .attr("preserveAspectRatio", "none")
                    .attr("data-id", node.id);
                break;

            case "compound_stroke":
            case "compound_fill":
            case "compound_fill_stroke":
                drawSvgCompound(node, parentG);
                break;

            default:
                break;
        }
    }

    function drawSvgPoints(node, parentG) {
        var p = node.props;
        var xs = p.x || [], ys = p.y || [];
        var r = (p.size || 1) * 2;
        var gpar = node.gpar || {};
        var colArr = Array.isArray(gpar.col) ? gpar.col : null;
        var fillArr = Array.isArray(gpar.fill) ? gpar.fill : null;
        var lwdArr = Array.isArray(gpar.lwd) ? gpar.lwd : null;
        var g = parentG.append("g").attr("class", "grob-points grob-interactive");
        for (var i = 0; i < xs.length; i++) {
            var c = g.append("circle")
                .attr("cx", xs[i]).attr("cy", ys[i]).attr("r", r)
                .attr("data-id", node.id)
                .attr("data-index", i);
            // Apply base gpar, then override per-point if arrays
            applyGparSvg(c, gpar);
            if (colArr) c.attr("stroke", colArr[i % colArr.length] || "none");
            if (fillArr) c.attr("fill", fillArr[i % fillArr.length] || "none");
            if (lwdArr) c.attr("stroke-width", lwdArr[i % lwdArr.length]);
            if (node.data) {
                var row = {};
                for (var key in node.data) {
                    row[key] = Array.isArray(node.data[key])
                        ? node.data[key][i] : node.data[key];
                }
                c.attr("data-row", JSON.stringify(row));
            }
        }
    }

    function pointsToSvg(xs, ys) {
        var parts = [];
        for (var i = 0; i < xs.length; i++) {
            parts.push(xs[i] + "," + ys[i]);
        }
        return parts.join(" ");
    }

    function drawSvgPolyline(node, parentG) {
        var p = node.props;
        if (p.groups) {
            p.groups.forEach(function (grp) {
                parentG.append("polyline")
                    .attr("points", pointsToSvg(grp.x, grp.y))
                    .attr("fill", "none")
                    .call(applyGparSvg, node.gpar);
            });
        } else {
            parentG.append("polyline")
                .attr("points", pointsToSvg(p.x || [], p.y || []))
                .attr("fill", "none")
                .call(applyGparSvg, node.gpar);
        }
    }

    function drawSvgSegments(node, parentG) {
        var p = node.props;
        var x0s = p.x0 || [], y0s = p.y0 || [];
        var x1s = p.x1 || [], y1s = p.y1 || [];
        for (var i = 0; i < x0s.length; i++) {
            parentG.append("line")
                .attr("x1", x0s[i]).attr("y1", y0s[i])
                .attr("x2", x1s[i]).attr("y2", y1s[i])
                .call(applyGparSvg, node.gpar);
        }
    }

    function drawSvgPolygon(node, parentG) {
        var p = node.props;
        parentG.append("polygon")
            .attr("points", pointsToSvg(p.x || [], p.y || []))
            .call(applyGparSvg, node.gpar);
    }

    function drawSvgPath(node, parentG) {
        var p = node.props;
        if (p.d) {
            parentG.append("path").attr("d", p.d)
                .attr("fill-rule", p.rule === "evenodd" ? "evenodd" : "nonzero")
                .call(applyGparSvg, node.gpar);
        } else if (p.groups) {
            var d = "";
            p.groups.forEach(function (grp) {
                for (var i = 0; i < grp.x.length; i++) {
                    d += (i === 0 ? "M" : "L") + grp.x[i] + "," + grp.y[i];
                }
                d += "Z";
            });
            parentG.append("path").attr("d", d)
                .attr("fill-rule", p.rule === "evenodd" ? "evenodd" : "nonzero")
                .call(applyGparSvg, node.gpar);
        }
    }

    function drawSvgCompound(node, parentG) {
        // compound_stroke / compound_fill / compound_fill_stroke
        // Render sub-paths as a combined <path>
        var subPaths = (node.props || {}).sub_paths || [];
        var g = parentG.append("g").attr("class", "compound-path");
        subPaths.forEach(function (sub) {
            drawSvgGrub(sub, g);
        });
        // Apply overall stroke/fill from the compound node's gpar
        applyGparSvg(g, node.gpar);
    }

    // ---- Canvas rendering -------------------------------------------------

    function drawCanvasNode(node, state) {
        if (!node) return;
        var ctx = state.ctx;
        if (node.type === "viewport") {
            ctx.save();
            if (node.clip) {
                var t = node.transform || {};
                ctx.beginPath();
                ctx.rect(t.x0 || 0, t.y0 || 0, t.w || 0, t.h || 0);
                ctx.clip();
            }
            (node.children || []).forEach(function (child) {
                drawCanvasNodeRouted(child, state);
            });
            ctx.restore();
        } else {
            drawCanvasPrimitive(node, state);
        }
    }

    function drawCanvasNodeRouted(node, state) {
        if (!node) return;
        if (node.type === "viewport") {
            drawCanvasNode(node, state);
            return;
        }
        var layer = routeToLayer(node);
        if (layer === "canvas") {
            drawCanvasPrimitive(node, state);
        }
        // SVG items are handled separately in the SVG pass
    }

    function drawCanvasPrimitive(node, state) {
        var ctx = state.ctx;
        var p = node.props || {};
        ctx.save();
        applyGparCanvas(ctx, node.gpar);

        switch (node.type) {
            case "points":
                drawCanvasPoints(node, state);
                break;
            case "rect":
                ctx.beginPath();
                ctx.rect(p.x, p.y, p.w, p.h);
                if (ctx.fillStyle !== "rgba(0,0,0,0)") ctx.fill();
                if (ctx.strokeStyle !== "rgba(0,0,0,0)") ctx.stroke();
                break;
            case "circle":
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                break;
            case "polygon":
                if (p.x && p.x.length >= 3) {
                    ctx.beginPath();
                    ctx.moveTo(p.x[0], p.y[0]);
                    for (var i = 1; i < p.x.length; i++) {
                        ctx.lineTo(p.x[i], p.y[i]);
                    }
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }
                break;
            default:
                break;
        }
        ctx.restore();
    }

    function drawCanvasPoints(node, state) {
        var ctx = state.ctx;
        var xs = node.props.x || [], ys = node.props.y || [];
        var r = (node.props.size || 1) * 2;
        var gpar = node.gpar || {};
        var colArr = Array.isArray(gpar.col) ? gpar.col : null;
        var fillArr = Array.isArray(gpar.fill) ? gpar.fill : null;
        var hasPerPointStyle = colArr || fillArr;

        if (hasPerPointStyle) {
            // Per-point rendering: each point gets its own fill/stroke
            for (var i = 0; i < xs.length; i++) {
                ctx.beginPath();
                ctx.arc(xs[i], ys[i], r, 0, Math.PI * 2);
                if (fillArr) ctx.fillStyle = fillArr[i % fillArr.length] || "rgba(0,0,0,0)";
                ctx.fill();
                if (colArr) {
                    ctx.strokeStyle = colArr[i % colArr.length] || "rgba(0,0,0,0)";
                    ctx.stroke();
                }
                state.canvasItems.push({
                    x: xs[i], y: ys[i], r: r,
                    id: node.id, index: i,
                    data: node.data ? extractRow(node.data, i) : null
                });
            }
        } else {
            // Batch rendering: all points share one style
            ctx.beginPath();
            for (var j = 0; j < xs.length; j++) {
                ctx.moveTo(xs[j] + r, ys[j]);
                ctx.arc(xs[j], ys[j], r, 0, Math.PI * 2);
                state.canvasItems.push({
                    x: xs[j], y: ys[j], r: r,
                    id: node.id, index: j,
                    data: node.data ? extractRow(node.data, j) : null
                });
            }
            ctx.fill();
            if (gpar.col && parseColour(gpar.col)) {
                ctx.stroke();
            }
        }
    }

    function extractRow(data, i) {
        var row = {};
        for (var key in data) {
            row[key] = Array.isArray(data[key]) ? data[key][i] : data[key];
        }
        return row;
    }

    // ---- Quadtree (simple implementation, no D3 dependency) ---------------

    function SimpleQuadtree(items) {
        this.items = items;
    }

    SimpleQuadtree.prototype.find = function (x, y, radius) {
        var best = null, bestDist = radius * radius;
        for (var i = 0; i < this.items.length; i++) {
            var it = this.items[i];
            var dx = it.x - x, dy = it.y - y;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestDist) {
                bestDist = d2;
                best = it;
            }
        }
        return best;
    };

    // If D3 quadtree is available, use it for better performance
    function buildQuadtree(items) {
        if (typeof d3 !== "undefined" && d3.quadtree) {
            return d3.quadtree()
                .x(function (d) { return d.x; })
                .y(function (d) { return d.y; })
                .addAll(items);
        }
        return new SimpleQuadtree(items);
    }

    // ---- Interactions -----------------------------------------------------

    function bindInteractions(state, options) {
        if (!options.interactive) return;

        // Tooltip
        if (options.tooltip !== false) {
            bindTooltip(state);
        }

        // Zoom / Pan (requires D3)
        if (options.zoom !== false && typeof d3 !== "undefined" && d3.zoom) {
            bindZoom(state, options);
        }

        // Brush selection (requires D3)
        if (options.brush && typeof d3 !== "undefined" && d3.brush) {
            bindBrush(state, options);
        }
    }

    function bindTooltip(state) {
        var tooltip = state.tooltip;
        var container = state.container;

        container.addEventListener("mousemove", function (event) {
            var rect = container.getBoundingClientRect();
            var mx = event.clientX - rect.left;
            var my = event.clientY - rect.top;

            // Check SVG elements first
            var svgEl = document.elementFromPoint(event.clientX, event.clientY);
            if (svgEl && svgEl.dataset && svgEl.dataset.row) {
                var rowData = JSON.parse(svgEl.dataset.row);
                showTooltip(tooltip, rowData, mx, my, container);
                return;
            }

            // Check Canvas via quadtree
            if (state.quadtree) {
                var nearest = state.quadtree.find(mx, my, TOOLTIP_SEARCH_RADIUS);
                if (nearest && nearest.data) {
                    showTooltip(tooltip, nearest.data, mx, my, container);
                    return;
                }
            }

            hideTooltip(tooltip);
        });

        container.addEventListener("mouseleave", function () {
            hideTooltip(tooltip);
        });
    }

    function showTooltip(tooltip, data, mx, my, container) {
        var lines = [];
        for (var key in data) {
            lines.push("<b>" + key + "</b>: " + data[key]);
        }
        tooltip.innerHTML = lines.join("<br>");
        tooltip.classList.add("visible");
        // Position: offset from cursor
        var tw = tooltip.offsetWidth || 100;
        var th = tooltip.offsetHeight || 30;
        var cw = container.offsetWidth;
        var left = mx + 12;
        var top = my - th - 8;
        if (left + tw > cw) left = mx - tw - 12;
        if (top < 0) top = my + 12;
        tooltip.style.left = left + "px";
        tooltip.style.top = top + "px";
    }

    function hideTooltip(tooltip) {
        tooltip.classList.remove("visible");
    }

    function bindZoom(state, options) {
        var zoomExtent = options.zoomExtent || [0.5, 20];
        var zoom = d3.zoom()
            .scaleExtent(zoomExtent)
            .on("zoom", function (event) {
                // SVG layer
                state.svgRoot.attr("transform", event.transform);
                // Canvas layer: redraw
                redrawCanvas(state, event.transform);
            });
        d3.select(state.overlay).call(zoom);
    }

    function redrawCanvas(state, transform) {
        var ctx = state.ctx;
        var w = state.width, h = state.height;
        ctx.save();
        ctx.clearRect(0, 0, w, h);
        if (transform) {
            ctx.translate(transform.x, transform.y);
            ctx.scale(transform.k, transform.k);
        }
        // Re-draw all canvas items
        state.canvasItems.forEach(function (item) {
            ctx.beginPath();
            ctx.arc(item.x, item.y, item.r, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.restore();
    }

    function bindBrush(state, options) {
        var brush = d3.brush()
            .extent([[0, 0], [state.width, state.height]])
            .on("end", function (event) {
                if (!event.selection) {
                    emitEvent(state, "brush", { selected: [] });
                    return;
                }
                var sel = event.selection;
                var x0 = sel[0][0], y0 = sel[0][1];
                var x1 = sel[1][0], y1 = sel[1][1];
                var selected = queryRegion(state, x0, y0, x1, y1);
                emitEvent(state, "brush", { selected: selected });
            });

        d3.select(state.overlay).append("g")
            .attr("class", "brush")
            .call(brush);
    }

    function queryRegion(state, x0, y0, x1, y1) {
        var result = [];
        state.canvasItems.forEach(function (item) {
            if (item.x >= x0 && item.x <= x1 && item.y >= y0 && item.y <= y1) {
                result.push(item);
            }
        });
        return result;
    }

    function emitEvent(state, name, detail) {
        var event = new CustomEvent("gridpy:" + name, { detail: detail });
        state.container.dispatchEvent(event);
    }

    // ---- Main render entry point ------------------------------------------

    function render(container, sceneGraph, options) {
        options = options || {};
        if (options.interactive === undefined) options.interactive = true;
        var theme = options.theme || "light";

        var sg = (typeof sceneGraph === "string")
            ? JSON.parse(sceneGraph) : sceneGraph;
        var w = sg.width, h = sg.height;

        // Setup container
        if (typeof container === "string") {
            container = document.getElementById(container)
                || document.querySelector(container);
        }
        container.innerHTML = "";
        container.classList.add("gridpy-container", "gridpy-theme-" + theme);
        container.style.width = w + "px";
        container.style.height = h + "px";

        // Canvas layer
        var canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        canvas.className = "gridpy-canvas";
        container.appendChild(canvas);
        var ctx = canvas.getContext("2d");

        // SVG layer
        var svgNS = "http://www.w3.org/2000/svg";
        var svgEl = document.createElementNS(svgNS, "svg");
        svgEl.setAttribute("width", w);
        svgEl.setAttribute("height", h);
        svgEl.setAttribute("class", "gridpy-svg");
        container.appendChild(svgEl);

        var svg = (typeof d3 !== "undefined") ? d3.select(svgEl) : null;
        var defs = svg ? svg.append("defs") : null;
        var svgRoot = svg ? svg.append("g").attr("class", "gridpy-root") : null;

        // Create defs
        if (defs) createDefs(defs, sg.defs);

        // Interaction overlay (transparent SVG)
        var overlayEl = document.createElementNS(svgNS, "svg");
        overlayEl.setAttribute("width", w);
        overlayEl.setAttribute("height", h);
        overlayEl.setAttribute("class", "gridpy-overlay");
        container.appendChild(overlayEl);

        // Tooltip
        var tooltip = document.createElement("div");
        tooltip.className = "gridpy-tooltip";
        container.appendChild(tooltip);

        var state = {
            container: container,
            canvas: canvas,
            ctx: ctx,
            svg: svg,
            svgRoot: svgRoot,
            overlay: overlayEl,
            tooltip: tooltip,
            width: w,
            height: h,
            dpi: sg.dpi || 150,
            canvasItems: [],
            quadtree: null,
            sceneGraph: sg
        };

        // Render SVG pass
        if (svgRoot && sg.root) {
            renderTree(sg.root, state, svgRoot);
        }

        // Render Canvas pass (only canvas-routed items)
        if (sg.root) {
            drawCanvasNode(sg.root, state);
        }

        // Build quadtree for Canvas hit-testing
        if (state.canvasItems.length > 0) {
            state.quadtree = buildQuadtree(state.canvasItems);
        }

        // Bind interactions
        bindInteractions(state, options);

        return state;
    }

    function renderTree(node, state, parentG) {
        if (!node) return;
        if (node.type === "viewport") {
            var g = parentG.append("g")
                .attr("class", "vp-" + (node.name || ""))
                .attr("data-viewport", node.name || "");
            if (node.clip_id) {
                g.attr("clip-path", "url(#" + node.clip_id + ")");
            }
            if (node.mask_id) {
                g.attr("mask", "url(#" + node.mask_id + ")");
            }
            (node.children || []).forEach(function (child) {
                renderTree(child, state, g);
            });
        } else {
            var layer = routeToLayer(node);
            if (layer === "svg") {
                drawSvgGrub(node, parentG);
            }
            // Canvas items are drawn in a separate pass
        }
    }

    // ---- Public API -------------------------------------------------------

    return {
        render: render,
        CANVAS_THRESHOLD: CANVAS_THRESHOLD,
        version: "0.1.0"
    };

})();
