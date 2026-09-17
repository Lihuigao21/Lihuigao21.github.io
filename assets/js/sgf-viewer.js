(function () {
  "use strict";

  function parseValue(text, start) {
    var value = "";
    var i = start + 1;
    while (i < text.length) {
      if (text[i] === "\\" && i + 1 < text.length) {
        i += 1;
        if (text[i] !== "\r" && text[i] !== "\n") value += text[i];
      } else if (text[i] === "]") {
        return { value: value, end: i + 1 };
      } else {
        value += text[i];
      }
      i += 1;
    }
    return { value: value, end: i };
  }

  function parseMainLine(text) {
    var cursor = { i: 0 };

    function node() {
      var props = {};
      cursor.i += 1;
      while (cursor.i < text.length && text[cursor.i] !== ";" && text[cursor.i] !== "(" && text[cursor.i] !== ")") {
        if (/\s/.test(text[cursor.i])) { cursor.i += 1; continue; }
        var match = text.slice(cursor.i).match(/^([A-Za-z]+)/);
        if (!match) { cursor.i += 1; continue; }
        var key = match[1].toUpperCase();
        cursor.i += match[1].length;
        props[key] = props[key] || [];
        while (text[cursor.i] === "[") {
          var parsed = parseValue(text, cursor.i);
          props[key].push(parsed.value);
          cursor.i = parsed.end;
        }
      }
      return props;
    }

    function tree() {
      var sequence = [];
      var longestChild = [];
      if (text[cursor.i] !== "(") return sequence;
      cursor.i += 1;
      while (cursor.i < text.length && text[cursor.i] !== ")") {
        if (/\s/.test(text[cursor.i])) { cursor.i += 1; continue; }
        if (text[cursor.i] === ";") { sequence.push(node()); continue; }
        if (text[cursor.i] === "(") {
          var child = tree();
          if (child.length > longestChild.length) longestChild = child;
          continue;
        }
        cursor.i += 1;
      }
      if (text[cursor.i] === ")") cursor.i += 1;
      return sequence.concat(longestChild);
    }

    while (cursor.i < text.length && text[cursor.i] !== "(") cursor.i += 1;
    return tree();
  }

  function coord(value, size) {
    if (!value || value.length < 2) return null;
    var x = value.charCodeAt(0) - 97;
    var y = value.charCodeAt(1) - 97;
    return x >= 0 && y >= 0 && x < size && y < size ? [x, y] : null;
  }

  function neighbors(x, y, size) {
    return [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]].filter(function (p) {
      return p[0] >= 0 && p[1] >= 0 && p[0] < size && p[1] < size;
    });
  }

  function groupAt(board, x, y, size) {
    var color = board[y][x];
    var stack = [[x, y]];
    var seen = {};
    var stones = [];
    var liberties = {};
    while (stack.length) {
      var point = stack.pop();
      var key = point[0] + "," + point[1];
      if (seen[key]) continue;
      seen[key] = true;
      stones.push(point);
      neighbors(point[0], point[1], size).forEach(function (next) {
        var value = board[next[1]][next[0]];
        if (!value) liberties[next[0] + "," + next[1]] = true;
        else if (value === color) stack.push(next);
      });
    }
    return { stones: stones, liberties: Object.keys(liberties).length };
  }

  function makeBoard(size) {
    return Array.from({ length: size }, function () { return Array(size).fill(null); });
  }

  function applyMove(board, move, size) {
    if (!move.point) return;
    var x = move.point[0];
    var y = move.point[1];
    board[y][x] = move.color;
    var other = move.color === "B" ? "W" : "B";
    neighbors(x, y, size).forEach(function (p) {
      if (board[p[1]][p[0]] !== other) return;
      var group = groupAt(board, p[0], p[1], size);
      if (group.liberties === 0) group.stones.forEach(function (s) { board[s[1]][s[0]] = null; });
    });
  }

  function init(root) {
    var source = root.dataset.sgf;
    var canvas = root.querySelector("canvas");
    var status = root.querySelector("[data-sgf-status]");
    var comment = root.querySelector("[data-sgf-comment]");
    var slider = root.querySelector("input[type=range]");
    var copyButton = root.querySelector("[data-sgf-copy]");
    var rawText = "";
    var moves = [];
    var size = 19;
    var current = 0;

    function draw() {
      var ratio = window.devicePixelRatio || 1;
      var width = Math.max(280, canvas.clientWidth);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(width * ratio);
      var ctx = canvas.getContext("2d");
      ctx.scale(ratio, ratio);
      ctx.fillStyle = "#d6a85f";
      ctx.fillRect(0, 0, width, width);
      var pad = width * 0.055;
      var step = (width - pad * 2) / (size - 1);
      ctx.strokeStyle = "#3b2b18";
      ctx.lineWidth = Math.max(1, width / 600);
      for (var n = 0; n < size; n += 1) {
        var pos = pad + n * step;
        ctx.beginPath(); ctx.moveTo(pad, pos); ctx.lineTo(width - pad, pos); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pos, pad); ctx.lineTo(pos, width - pad); ctx.stroke();
      }
      var stars = size === 19 ? [3, 9, 15] : size === 13 ? [3, 6, 9] : [2, 4, 6];
      stars.forEach(function (x) { stars.forEach(function (y) {
        ctx.beginPath(); ctx.arc(pad + x * step, pad + y * step, Math.max(2.5, step * 0.09), 0, Math.PI * 2); ctx.fillStyle = "#302316"; ctx.fill();
      }); });
      var board = makeBoard(size);
      for (var i = 0; i < current; i += 1) applyMove(board, moves[i], size);
      for (var y = 0; y < size; y += 1) for (var x = 0; x < size; x += 1) {
        if (!board[y][x]) continue;
        ctx.beginPath(); ctx.arc(pad + x * step, pad + y * step, step * 0.43, 0, Math.PI * 2);
        ctx.fillStyle = board[y][x] === "B" ? "#171717" : "#f4f1e8"; ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,.45)"; ctx.stroke();
      }
      if (current && moves[current - 1].point) {
        var last = moves[current - 1];
        ctx.beginPath(); ctx.arc(pad + last.point[0] * step, pad + last.point[1] * step, step * 0.12, 0, Math.PI * 2);
        ctx.fillStyle = last.color === "B" ? "#f5f5f5" : "#222"; ctx.fill();
      }
      slider.value = current;
      var move = current ? moves[current - 1] : null;
      status.textContent = current + " / " + moves.length + (move ? " · " + (move.color === "B" ? "黑" : "白") + (move.point ? "落子" : "停一手") : " · 初始局面");
      comment.textContent = move && move.comment ? move.comment : "当前手无附加评论。";
      root.querySelectorAll("button[data-step]").forEach(function (button) {
        var stepValue = button.dataset.step;
        button.disabled = (current === 0 && (stepValue === "first" || stepValue === "prev")) || (current === moves.length && (stepValue === "next" || stepValue === "last"));
      });
    }

    fetch(source).then(function (response) {
      if (!response.ok) throw new Error("无法载入 SGF");
      return response.text();
    }).then(function (text) {
      rawText = text;
      var nodes = parseMainLine(text);
      var rootProps = nodes[0] || {};
      size = parseInt(rootProps.SZ && rootProps.SZ[0], 10) || 19;
      moves = nodes.slice(1).map(function (node) {
        var color = node.B ? "B" : node.W ? "W" : null;
        if (!color) return null;
        var value = (node[color] && node[color][0]) || "";
        return { color: color, point: coord(value, size), comment: (node.C && node.C[0]) || "" };
      }).filter(Boolean);
      slider.max = moves.length;
      draw();
    }).catch(function (error) {
      status.textContent = error.message;
      root.classList.add("sgf-viewer-error");
    });

    root.addEventListener("click", function (event) {
      var button = event.target.closest("button[data-step]");
      if (!button) return;
      if (button.dataset.step === "first") current = 0;
      if (button.dataset.step === "prev") current = Math.max(0, current - 1);
      if (button.dataset.step === "next") current = Math.min(moves.length, current + 1);
      if (button.dataset.step === "last") current = moves.length;
      draw();
    });
    slider.addEventListener("input", function () { current = Number(slider.value); draw(); });
    copyButton.addEventListener("click", function () {
      navigator.clipboard.writeText(rawText).then(function () { copyButton.textContent = "已复制 SGF"; setTimeout(function () { copyButton.textContent = "复制 SGF"; }, 1600); });
    });
    root.tabIndex = 0;
    root.addEventListener("keydown", function (event) {
      if (event.key === "ArrowLeft") { current = Math.max(0, current - 1); draw(); event.preventDefault(); }
      if (event.key === "ArrowRight") { current = Math.min(moves.length, current + 1); draw(); event.preventDefault(); }
    });
    window.addEventListener("resize", draw);
  }

  document.querySelectorAll("[data-sgf-viewer]").forEach(init);
}());
