/* Synthetic random-feature committee; no dynamics or chemical predictions. */
(function () {
  "use strict";
  const BLUE = "#315f85", BROWN = "#925e35", INK = "#292d30", GRAY = "#8d9fac";

  function reference(q) {
    return {energy: 0.12*q**4-0.18*q*q+0.08*Math.sin(3*q),
      force: -0.48*q**3+0.36*q-0.24*Math.cos(3*q)};
  }

  function validate(data) {
    if (!data || data.schema !== "dpgen-committee-teaching-v1" || data.models.length !== 4) throw Error("Invalid teaching model data.");
    data.models.forEach(function (model) {
      if (model.centers.length !== 24 || model.widths.length !== 24 || model.weights.length !== 25) throw Error("Invalid feature dimensions.");
      model.centers.concat(model.widths, model.weights).forEach(function (v) { if (!Number.isFinite(v)) throw Error("Nonfinite model parameter."); });
      if (model.widths.some(function (v) {return v <= 0;})) throw Error("Feature widths must be positive.");
    });
    if (new Set(data.models.map(function (m) {return m.seed;})).size !== 4) throw Error("Four distinct seeds are required.");
    return data;
  }

  function point(data, q, bias) {
    const energy = [], force = [], truth = reference(q);
    const slope = bias ? data.config.common_bias_slope : 0;
    data.models.forEach(function (model) {
      let e = model.weights[24], f = 0;
      for (let j = 0; j < 24; j++) {
        const d = q-model.centers[j], s = model.widths[j];
        const phi = Math.exp(-0.5*(d/s)**2);
        e += model.weights[j]*phi;
        f += model.weights[j]*phi*d/(s*s);
      }
      energy.push(e+slope*q); force.push(f-slope);
    });
    const meanEnergy = energy.reduce(function (a, v) {return a+v;}, 0)/4;
    const meanForce = force.reduce(function (a, v) {return a+v;}, 0)/4;
    const sigma = Math.sqrt(force.reduce(function (a, v) {return a+(v-meanForce)**2;}, 0)/4);
    return {q: q, energy: energy, force: force, mean_energy: meanEnergy,
      mean_force: meanForce, sigma_force: sigma, reference_energy: truth.energy,
      reference_force: truth.force, signed_force_error: meanForce-truth.force,
      absolute_force_error: Math.abs(meanForce-truth.force)};
  }

  function series(data, bias, n) {
    n = n || data.config.display_count;
    const range = data.config.display_range, rows = [];
    for (let i = 0; i < n; i++) rows.push(point(data, range[0]+i*(range[1]-range[0])/(n-1), bias));
    return rows;
  }

  function classify(rows, lo, hi) {
    if (!(Number.isFinite(lo) && Number.isFinite(hi) && 0 <= lo && lo < hi)) throw Error("Require 0 <= lower < upper threshold.");
    const counts = {below: 0, candidate: 0, above: 0};
    rows.forEach(function (r) {counts[r.sigma_force < lo ? "below" : r.sigma_force < hi ? "candidate" : "above"]++;});
    return counts;
  }

  function format(v) {return Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(3) : v.toFixed(4);}

  function draw(canvas, data, rows, inspected, lo, hi) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.getBoundingClientRect().width;
    const height = width < 480 ? 520 : 470;
    canvas.style.height = height+"px";
    canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    ctx.font = "12px sans-serif"; ctx.lineJoin = "round";
    const left = width < 480 ? 46 : 58, right = width-14;
    const a = {left: left, right: right, top: 32, bottom: height/2-32};
    const b = {left: left, right: right, top: height/2+38, bottom: height-38};
    const range = data.config.display_range;
    const x = function (q) {return left+(q-range[0])/(range[1]-range[0])*(right-left);};
    let fmin = Infinity, fmax = -Infinity, diagmax = hi;
    rows.forEach(function (r) {
      r.force.concat([r.reference_force]).forEach(function (f) {fmin = Math.min(fmin, f); fmax = Math.max(fmax, f);});
      diagmax = Math.max(diagmax, r.sigma_force, r.absolute_force_error);
    });
    const padding = (fmax-fmin)*0.08; fmin -= padding; fmax += padding; diagmax *= 1.10;
    function y(box, v, low, high) {return box.bottom-(v-low)/(high-low)*(box.bottom-box.top);}
    function axes(box, low, high, title) {
      ctx.fillStyle = "#e5edf1"; ctx.fillRect(x(-0.9), box.top, x(0.9)-x(-0.9), box.bottom-box.top);
      ctx.strokeStyle = "#c7cbce"; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath();ctx.moveTo(left,box.top);ctx.lineTo(left,box.bottom);ctx.lineTo(right,box.bottom);ctx.stroke();
      ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.fillText(title, 2, box.top-15);
      ctx.font = "11px sans-serif"; ctx.textAlign = "right";
      for (let i=0; i<=3; i++) {
        const v = low+i*(high-low)/3, yy = y(box,v,low,high);
        ctx.fillText(Math.abs(v)<0.01 ? v.toFixed(2) : v.toFixed(1),left-5,yy+4);
      }
      ctx.textAlign = "center";
      [-2,-1,0,1,2].forEach(function (q) {ctx.fillText(String(q),x(q),box.bottom+17);});
      ctx.fillText("Coordinate q", (left+right)/2, box.bottom+31);
      ctx.font = "12px sans-serif";
    }
    function line(box, values, low, high, color, lineWidth, dash) {
      ctx.save(); ctx.beginPath();ctx.rect(box.left,box.top,box.right-box.left,box.bottom-box.top);ctx.clip();
      ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dash || []);ctx.beginPath();
      rows.forEach(function(r,i){const yy=y(box,values(r),low,high);if(i===0)ctx.moveTo(x(r.q),yy);else ctx.lineTo(x(r.q),yy);});
      ctx.stroke();ctx.restore();
    }
    axes(a,fmin,fmax,"Force: four members, mean, reference");
    for(let j=0;j<4;j++)line(a,function(r){return r.force[j];},fmin,fmax,GRAY,.85);
    line(a,function(r){return r.reference_force;},fmin,fmax,INK,1.5,[4,3]);
    line(a,function(r){return r.mean_force;},fmin,fmax,BLUE,2);
    ctx.fillStyle=INK;
    data.training.q.forEach(function(q,i){ctx.beginPath();ctx.arc(x(q),y(a,data.training.force[i],fmin,fmax),2.4,0,2*Math.PI);ctx.fill();});
    axes(b,0,diagmax,"Force: committee spread and actual error");
    line(b,function(r){return r.sigma_force;},0,diagmax,BLUE,2);
    line(b,function(r){return r.absolute_force_error;},0,diagmax,BROWN,1.7);
    [lo,hi].forEach(function(v){ctx.strokeStyle="#90979c";ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(left,y(b,v,0,diagmax));ctx.lineTo(right,y(b,v,0,diagmax));ctx.stroke();});
    [a,b].forEach(function(box){ctx.strokeStyle="#62696d";ctx.setLineDash([2,3]);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(inspected.q),box.top);ctx.lineTo(x(inspected.q),box.bottom);ctx.stroke();});
    [[a,inspected.mean_force,fmin,fmax,BLUE],[b,inspected.sigma_force,0,diagmax,BLUE],[b,inspected.absolute_force_error,0,diagmax,BROWN]].forEach(function(v){ctx.fillStyle=v[4];ctx.beginPath();ctx.arc(x(inspected.q),y(v[0],v[1],v[2],v[3]),3.2,0,2*Math.PI);ctx.fill();});
    ctx.setLineDash([]);
  }

  function init(widget) {
    const status=widget.querySelector("[data-dpgen-status]");
    const src=widget.getAttribute("data-source");
    fetch(src).then(function(r){if(!r.ok)throw Error("Teaching data HTTP "+r.status);return r.json();}).then(function(raw){
      const data=validate(raw), q=widget.querySelector('[data-dpgen-control="q"]'),
        lo=widget.querySelector('[data-dpgen-control="lo"]'), hi=widget.querySelector('[data-dpgen-control="hi"]'),
        bias=widget.querySelector('[data-dpgen-control="bias"]'), canvas=widget.querySelector("canvas");
      let rows=series(data,false);
      function update(event){
        lo.value=Math.min(.5,Math.max(0,Number(lo.value)||0));
        hi.value=Math.min(.6,Math.max(.005,Number(hi.value)||.15));
        if(event && event.target===lo && +lo.value>=+hi.value)hi.value=(+lo.value+0.005).toFixed(3);
        if(event && event.target===hi && +hi.value<=+lo.value)lo.value=Math.max(0,+hi.value-0.005).toFixed(3);
        rows=series(data,bias.checked);
        const r=point(data,+q.value,bias.checked), counts=classify(rows,+lo.value,+hi.value);
        widget.querySelector('[data-dpgen-output="q"]').textContent=(+q.value).toFixed(2);
        widget.querySelector('[data-dpgen-output="numbers"]').textContent="Reference force: "+format(r.reference_force)+"; mean force: "+format(r.mean_force)+"; spread: "+format(r.sigma_force)+"; actual error: "+format(r.absolute_force_error)+".";
        widget.querySelector('[data-dpgen-output="energy"]').textContent="Reference energy: "+format(r.reference_energy)+"; mean energy: "+format(r.mean_energy)+". All quantities are dimensionless.";
        widget.querySelector('[data-dpgen-output="counts"]').textContent="Synthetic grid (601 points): "+counts.below+" below lower threshold; "+counts.candidate+" candidate; "+counts.above+" at or above upper threshold.";
        widget.querySelector('[data-dpgen-output="class"]').textContent="Inspected point: "+(r.sigma_force<+lo.value?"below lower threshold":r.sigma_force<+hi.value?"candidate interval":"at or above upper threshold")+".";
        status.textContent=bias.checked?"Common bias ON: add b(q) = 0.4q to every energy and -0.4 to every force. The spread and classifications stay unchanged; the error need not increase at every q.":"Common bias OFF. Four seeded random-feature fits share the same training data. These editable thresholds illustrate standard-style triage, not the molecular project's ranking rule. Grid points are not MD snapshots or accepted quantum labels.";
        draw(canvas,data,rows,r,+lo.value,+hi.value);
        widget.dpgenState={data:data,rows:rows,point:r,counts:counts,lo:+lo.value,hi:+hi.value,bias:bias.checked};
      }
      widget.classList.add("is-ready");
      [q,lo,hi,bias].forEach(function(control){control.addEventListener("input",update);control.addEventListener("change",update);});
      widget.querySelector('[data-dpgen-control="reset"]').addEventListener("click",function(){q.value=0;lo.value=.03;hi.value=.15;bias.checked=false;update();});
      if(window.ResizeObserver)new ResizeObserver(function(){update();}).observe(canvas.parentElement);
      update();
    }).catch(function(err){widget.classList.add("has-error");status.textContent="Interactive teaching example unavailable; the static figure remains below. "+err.message;});
  }

  window.DPGENCommitteeMath={reference:reference,point:point,series:series,classify:classify,validate:validate};
  function start(){document.querySelectorAll("[data-dpgen-widget]").forEach(init);}
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",start);else start();
}());
