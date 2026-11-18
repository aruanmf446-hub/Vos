/* Local Colab target editor. No network requests or inference in this UI. */
function vosTargetEditor(data) {
  return new Promise(resolve => {
    const host=document.createElement('section');
    host.style.cssText='font:15px sans-serif;padding:16px;background:#f5f7fa;color:#16202a;max-width:900px';
    host.innerHTML=`<h3>Selecionar personagem — experimental, opcional</h3>
      <p>Marque o alvo em verde e outra pessoa/cenário em vermelho. Os pontos podem ser acrescentados em qualquer quadro-chave.
      Na revisão, o vermelho translúcido indica a região que será substituída. Confira todos os quadros.</p>
      <label>Nome do alvo <input data-role="name" placeholder="Ex.: Wanda"></label>
      <p><button data-role="prev">← Anterior</button> <input data-role="frame" type="range" min="0" max="${data.images.length-1}" value="0">
      <button data-role="next">Próximo →</button> <output data-role="position"></output></p>
      <p><label>Ferramenta <select data-role="tool"><option value="positive">Ponto positivo: alvo</option><option value="negative">Ponto negativo: preservar</option>
      ${data.mode==='review'?'<option value="paint">Pincel: incluir na máscara</option><option value="erase">Borracha: preservar região</option>':''}</select></label>
      <label>Tamanho do pincel <input data-role="brush" type="number" min="1" max="150" value="12"></label>
      <button data-role="clear">Limpar pontos deste quadro</button></p>
      <canvas data-role="canvas" style="width:100%;max-width:640px;touch-action:none;border:1px solid #999"></canvas>
      <p data-role="warning" style="color:#a01919"></p>
      <p><label><input data-role="review" type="checkbox"> Revisei este quadro: somente o alvo está na máscara</label></p>
      <p><button data-role="save">${data.mode==='review'?'Salvar correções e aprovar todos os quadros':'Salvar pontos dos quadros-chave'}</button>
      <button data-role="cancel">Cancelar</button></p><p data-role="status" aria-live="polite"></p>`;
    document.body.appendChild(host);
    const el=role=>host.querySelector(`[data-role="${role}"]`),canvas=el('canvas'),ctx=canvas.getContext('2d');
    const mask=document.createElement('canvas'),mc=mask.getContext('2d',{willReadFrequently:true});
    canvas.width=mask.width=data.size[0];canvas.height=mask.height=data.size[1];
    const points=JSON.parse(JSON.stringify(data.keyframes)),corrections={},reviewed=new Set();
    let frame=0,bg=null,loading=true,sequence=0,drawing=false,last=null;
    el('name').value=data.name||'';el('review').parentElement.hidden=data.mode!=='review';
    const load=src=>new Promise((ok,fail)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=fail;i.src=src;});
    function redraw() {
      ctx.clearRect(0,0,canvas.width,canvas.height);if(bg)ctx.drawImage(bg,0,0,canvas.width,canvas.height);
      if(data.mode==='review') {
        const rgba=mc.getImageData(0,0,mask.width,mask.height),overlay=document.createElement('canvas');overlay.width=mask.width;overlay.height=mask.height;
        for(let i=0;i<rgba.data.length;i+=4){const selected=rgba.data[i]>=128;rgba.data[i]=255;rgba.data[i+1]=40;rgba.data[i+2]=30;rgba.data[i+3]=selected?105:0;}
        overlay.getContext('2d').putImageData(rgba,0,0);ctx.drawImage(overlay,0,0);
      }
      for(const [x,y,label] of (points[String(frame)]?.points||[])) {
        ctx.fillStyle=label?'#00ff80':'#ff0030';ctx.strokeStyle='#101010';ctx.lineWidth=2;
        ctx.beginPath();ctx.arc(x*canvas.width,y*canvas.height,Math.max(4,canvas.width/100),0,2*Math.PI);ctx.fill();ctx.stroke();
      }
    }
    async function navigate(n) {
      loading=true;canvas.dataset.loading="true";const request=++sequence;frame=Math.max(0,Math.min(data.images.length-1,n));
      el('frame').value=frame;el('position').textContent=`Quadro ${frame} / ${data.images.length-1} · t=${data.timestamps[frame]} s`;
      el('review').checked=reviewed.has(frame);
      el('warning').textContent=(data.issues||[]).filter(i=>i.frame===frame).flatMap(i=>i.reasons).join('; ');
      const image=await load(data.images[frame]);let imageMask=null;
      if(data.mode==='review')imageMask=await load(corrections[frame]||data.masks[frame]);
      if(request!==sequence)return;
      bg=image;mc.fillStyle='black';mc.fillRect(0,0,mask.width,mask.height);if(imageMask)mc.drawImage(imageMask,0,0);
      loading=false;canvas.dataset.loading="false";canvas.dataset.frame=String(frame);redraw();
    }
    function rememberMask() {
      const pixels=mc.getImageData(0,0,mask.width,mask.height);
      for(let i=0;i<pixels.data.length;i+=4){const v=pixels.data[i]>=128?255:0;pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=v;pixels.data[i+3]=255;}
      mc.putImageData(pixels,0,0);corrections[frame]=mask.toDataURL('image/png');
      reviewed.delete(frame);el('review').checked=false;
    }
    const location=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width-1,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height-1,(e.clientY-r.top)*canvas.height/r.height))];};
    function paint(e) {
      const xy=location(e),tool=el('tool').value;
      if(tool!=='paint'&&tool!=='erase')return;
      const radius=Math.max(1,Math.min(150,Number(el('brush').value)||12));
      mc.strokeStyle=mc.fillStyle=tool==='paint'?'white':'black';mc.lineWidth=radius*2;mc.lineCap='round';
      if(last){mc.beginPath();mc.moveTo(...last);mc.lineTo(...xy);mc.stroke();}
      mc.beginPath();mc.arc(...xy,radius,0,2*Math.PI);mc.fill();last=xy;rememberMask();redraw();
    }
    canvas.onpointerdown=e=>{
      if(loading)return;canvas.setPointerCapture(e.pointerId);const [x,y]=location(e),tool=el('tool').value;
      if(tool==='positive'||tool==='negative') {
        (points[String(frame)]||= {points:[]}).points.push([x/canvas.width,y/canvas.height,tool==='positive'?1:0]);
        reviewed.delete(frame);el('review').checked=false;redraw();
      } else {drawing=true;last=null;paint(e);}
    };
    canvas.onpointermove=e=>{if(drawing)paint(e);};canvas.onpointerup=canvas.onpointercancel=()=>{drawing=false;last=null;};
    el('prev').onclick=()=>navigate(frame-1);el('next').onclick=()=>navigate(frame+1);el('frame').oninput=()=>navigate(Number(el('frame').value));
    el('clear').onclick=()=>{delete points[String(frame)];reviewed.delete(frame);el('review').checked=false;redraw();};
    el('review').onchange=()=>{if(!loading){el('review').checked?reviewed.add(frame):reviewed.delete(frame);el('status').textContent=`${reviewed.size}/${data.images.length} quadros revisados`;}else el('review').checked=false;};
    el('cancel').onclick=()=>{host.remove();resolve({cancelled:true});};
    el('save').onclick=()=>{
      if(loading)return;
      if(data.mode==='prompts'&&!(points['0']?.points||[]).some(p=>p[2]===1)){el('status').textContent='Marque o alvo com um ponto positivo no quadro 0.';return;}
      // Point edits made during review must be propagated by SAM2 before approval.
      if(data.mode==='review'&&JSON.stringify(points)!==JSON.stringify(data.keyframes)) {
        el('status').textContent='Pontos alterados: salve-os pelo editor de seleção e execute SAM2 novamente; revise depois.';
        host.remove();resolve({name:el('name').value,keyframes:points,corrections,reviewed:[],needs_tracking:true});return;
      }
      if(data.mode==='review'&&reviewed.size!==data.images.length){el('status').textContent='Revise e marque todos os quadros. Não há aprovação automática.';return;}
      const result={name:el('name').value,keyframes:points,corrections,reviewed:[...reviewed]};host.remove();resolve(result);
    };
    navigate(0).catch(e=>{el('status').textContent='Falha ao abrir os quadros: '+e;});
  });
}
