/* Exact thermal SHO formulas. Fourier convention: exp(-i omega t). */
(() => {
  "use strict";
  const root=document.querySelector('[data-sho-explorer]');
  if(!root)return;
  const ns='http://www.w3.org/2000/svg';
  const controls=Object.fromEntries([...root.querySelectorAll('[data-sho]')].map(e=>[e.dataset.sho,e]));
  const state={x:2,s:0,observable:'q',hidden:new Set()};
  const series=[
    {key:'re',label:'Re Cₛ / Cₛ spectrum',color:'var(--sho-blue)',dash:''},
    {key:'im',label:'Im Cₛ',color:'var(--sho-brown)',dash:'6 4'},
    {key:'kubo',label:'Kubo',color:'var(--sho-green)',dash:''},
    {key:'sym',label:'Anticommutator',color:'var(--sho-gray)',dash:'2 4'},
    {key:'half',label:'Midpoint C½',color:'var(--sho-purple)',dash:'9 3 2 3'},
    {key:'classical',label:'Classical',color:'var(--sho-teal)',dash:'10 6'}
  ];
  function element(tag,attrs={},text) {
    const e=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
    if(text!==undefined)e.textContent=text;
    return e;
  }
  function number(v){return Math.abs(v)<1e-12?'0':Number(v.toPrecision(3)).toString();}
  function formulas(theta,s=state.s) {
    const x=state.x,n=1/Math.expm1(x),b=2*n+1;
    if(state.observable==='q') {
      const plus=(n+1)*Math.exp(-s*x),minus=n*Math.exp(s*x);
      return {re:(plus+minus)*Math.cos(theta),im:(plus-minus)*Math.sin(theta),
        kubo:2/x*Math.cos(theta),sym:b*Math.cos(theta),
        half:Math.cos(theta)/Math.sinh(x/2),classical:2/x*Math.cos(theta)};
    }
    const diagonal=4*n*(n+1),plus=2*(n+1)**2*Math.exp(-2*s*x),minus=2*n*n*Math.exp(2*s*x);
    return {re:diagonal+(plus+minus)*Math.cos(2*theta),im:(plus-minus)*Math.sin(2*theta),
      kubo:diagonal+2*b/x*Math.cos(2*theta),sym:diagonal+2*((n+1)**2+n*n)*Math.cos(2*theta),
      half:diagonal*(1+Math.cos(2*theta)),classical:4/x**2*(1+Math.cos(2*theta))};
  }
  function spectralLines() {
    const x=state.x,n=1/Math.expm1(x),b=2*n+1,nonlinear=state.observable==='q2';
    if(!nonlinear)return {
      frequencies:[-1,1], re:[n*Math.exp(state.s*x),(n+1)*Math.exp(-state.s*x)],
      kubo:[1/x,1/x],sym:[b/2,b/2],half:[1/(2*Math.sinh(x/2)),1/(2*Math.sinh(x/2))],classical:[1/x,1/x]};
    const d=4*n*(n+1);
    return {frequencies:[-2,0,2],re:[2*n*n*Math.exp(2*state.s*x),d,2*(n+1)**2*Math.exp(-2*state.s*x)],
      kubo:[b/x,d,b/x],sym:[(n+1)**2+n*n,d,(n+1)**2+n*n],
      half:[2*n*(n+1),d,2*n*(n+1)],classical:[2/x**2,4/x**2,2/x**2]};
  }
  function axes(svg,width,height,domainX,domainY,caption,ycaption,xTicks) {
    const frame={left:62,right:width-15,top:31,bottom:height-49};
    const x=v=>frame.left+(v-domainX[0])/(domainX[1]-domainX[0])*(frame.right-frame.left);
    const y=v=>frame.bottom-(v-domainY[0])/(domainY[1]-domainY[0])*(frame.bottom-frame.top);
    svg.append(element('rect',{class:'sho-axis',x:frame.left,y:frame.top,width:frame.right-frame.left,height:frame.bottom-frame.top}));
    for(let i=0;i<5;i++) {
      const value=domainY[0]+i/4*(domainY[1]-domainY[0]);
      svg.append(element('line',{class:'sho-grid',x1:frame.left,y1:y(value),x2:frame.right,y2:y(value)}));
      svg.append(element('text',{x:frame.left-8,y:y(value)+4,'text-anchor':'end'},number(value)));
    }
    xTicks.forEach(t=>{
      svg.append(element('line',{class:'sho-axis',x1:x(t.value),y1:frame.bottom,x2:x(t.value),y2:frame.bottom+4}));
      svg.append(element('text',{x:x(t.value),y:frame.bottom+19,'text-anchor':t.value===domainX[0]?'start':t.value===domainX[1]?'end':'middle'},t.label));
    });
    svg.append(element('text',{class:'sho-axis-title',x:frame.left,y:17},ycaption));
    svg.append(element('text',{class:'sho-axis-title',x:(frame.left+frame.right)/2,y:height-6,'text-anchor':'middle'},caption));
    return {frame,x,y};
  }
  function drawCurvePlot(kind) {
    const holder=root.querySelector(`[data-sho-plot="${kind}"]`),width=Math.max(180,holder.clientWidth),height=kind==='time'?290:215;
    const svg=element('svg',{class:'sho-chart',viewBox:`0 0 ${width} ${height}`,height,role:'img','aria-label':kind==='time'?'Exact real-time correlation curves':'Equal-time correlation versus thermal split'});
    const unit=state.observable==='q'?'ℓ²':'ℓ⁴';
    const active=series.filter(d=>!state.hidden.has(d.key)&&(kind==='time'||d.key!=='im'));
    const xmax=kind==='time'?2*Math.PI:1;
    const points=Array.from({length:241},(_,i)=>{const coord=i/240*xmax;return {coord,values:kind==='time'?formulas(coord):formulas(0,coord)};});
    const values=points.flatMap(p=>active.map(d=>p.values[d.key]));values.push(0);
    let low=Math.min(...values),high=Math.max(...values),span=high-low||1;
    low-=span*.09;high+=span*.09;
    const xTicks=kind==='time'?(width<420?[{value:0,label:'0'},{value:Math.PI,label:'π'},{value:2*Math.PI,label:'2π'}]:[{value:0,label:'0'},{value:Math.PI/2,label:'π/2'},{value:Math.PI,label:'π'},{value:3*Math.PI/2,label:'3π/2'},{value:2*Math.PI,label:'2π'}]):(width<420?[0,.5,1]:[0,.25,.5,.75,1]).map(value=>({value,label:number(value)}));
    const scales=axes(svg,width,height,[0,xmax],[low,high],kind==='time'?'Real time Ωt (dimensionless)':'Thermal split s (dimensionless)',kind==='time'?`C(t) / ${unit}`:`Cₛ(0) / ${unit}`,xTicks);
    active.forEach(d=>{
      const path=points.map((p,i)=>(i?'L':'M')+scales.x(p.coord).toFixed(2)+','+scales.y(p.values[d.key]).toFixed(2)).join(' ');
      const line=element('path',{class:'sho-line',d:path,'stroke-dasharray':d.dash,'data-series':d.key});line.style.setProperty('--series',d.color);svg.append(line);
    });
    if(kind==='split'&&!state.hidden.has('re'))svg.append(element('circle',{class:'sho-selected',cx:scales.x(state.s),cy:scales.y(formulas(0).re),r:4}));
    holder.replaceChildren(svg);
  }
  function drawSpectrum() {
    const holder=root.querySelector('[data-sho-plot="spectrum"]'),width=Math.max(180,holder.clientWidth),height=235;
    const svg=element('svg',{class:'sho-chart',viewBox:`0 0 ${width} ${height}`,height,role:'img','aria-label':'Discrete spectral line weights; the shared two pi factor is omitted'});
    const data=spectralLines(),active=series.filter(d=>d.key!=='im'&&!state.hidden.has(d.key));
    const all=active.flatMap(d=>data[d.key]),high=Math.max(...all,0)||1;
    const frequencyMax=state.observable==='q'?1:2;
    const xticks=data.frequencies.map(value=>({value,label:String(value)}));
    const {x,y,frame}=axes(svg,width,height,[-frequencyMax-.55,frequencyMax+.55],[0,high*1.15],'Frequency ω / Ω',`Line weight / ${state.observable==='q'?'ℓ²':'ℓ⁴'}`,xticks);
    data.frequencies.forEach((freq,index)=>{
      active.forEach((d,j)=>{
        const offset=(j-(active.length-1)/2)*6,px=x(freq)+offset,weight=data[d.key][index];
        const group=element('g',{'data-spectral-series':d.key,'data-frequency':freq,'data-weight':weight});group.style.setProperty('--series',d.color);
        group.append(element('title',{},`${d.label}: ω/Ω=${freq}, weight=${number(weight)}`));
        group.append(element('line',{class:'sho-stick',x1:px,y1:y(0),x2:px,y2:y(weight),'stroke-dasharray':d.dash}));
        group.append(element('circle',{class:'sho-stick-marker',cx:px,cy:y(weight),r:2.5}));svg.append(group);
      });
    });
    svg.append(element('desc',{},'Line positions are displaced slightly sideways within each frequency group to separate the series; every line belongs to the labeled frequency.'));
    holder.replaceChildren(svg);
  }
  function draw() {
    root.querySelector('[data-sho-value="x"]').textContent=state.x.toFixed(2);
    root.querySelector('[data-sho-value="s"]').textContent=state.s.toFixed(2);
    root.querySelectorAll('.sho-legend button').forEach(b=>b.setAttribute('aria-pressed',String(!state.hidden.has(b.dataset.key))));
    drawCurvePlot('time');drawCurvePlot('split');drawSpectrum();
    const values=formulas(0),unit=state.observable==='q'?'ℓ²':'ℓ⁴';
    root.querySelector('[data-sho-readout]').textContent=`At t = 0: Cₛ = ${number(values.re)}, Kubo = ${number(values.kubo)}, classical = ${number(values.classical)} (units ${unit}).`;
  }
  const legend=root.querySelector('.sho-legend');
  series.forEach(d=>{const button=document.createElement('button');button.type='button';button.dataset.key=d.key;button.setAttribute('aria-pressed','true');
    const swatch=document.createElement('span');swatch.className='sho-swatch';swatch.style.setProperty('--series',d.color);swatch.dataset.style=d.key==='sym'?'dotted':d.dash?'dashed':'solid';swatch.setAttribute('aria-hidden','true');
    button.append(swatch,document.createTextNode(d.label));button.addEventListener('click',()=>{state.hidden.has(d.key)?state.hidden.delete(d.key):state.hidden.add(d.key);draw();});legend.append(button);});
  Object.entries(controls).forEach(([key,input])=>input.addEventListener('input',()=>{state[key]=key==='observable'?input.value:Number(input.value);draw();}));
  let width=0;new ResizeObserver(entries=>{const next=Math.round(entries[0].contentRect.width);if(next!==width){width=next;draw();}}).observe(root);
  draw();
})();
