/* Actual headless Chromium interaction, synthetic two-person images. No models. */
const assert=require('node:assert/strict');
const path=require('node:path');
const {chromium}=require('playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:"chromium",executablePath:process.env.VOS_CHROMIUM_PATH||undefined,args:['--no-sandbox']});
  try {
    const page=await browser.newPage({viewport:{width:900,height:900}});
    const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.setContent('<!doctype html><html><body></body></html>');
    await page.addScriptTag({path:path.resolve(__dirname,'../static/target_editor.js')});
    const data=await page.evaluate(()=>{
      const canvas=document.createElement('canvas');canvas.width=64;canvas.height=40;const ctx=canvas.getContext('2d');
      ctx.fillStyle='#303030';ctx.fillRect(0,0,64,40);ctx.fillStyle='red';ctx.fillRect(5,10,11,21);ctx.fillStyle='green';ctx.fillRect(45,10,11,21);const image=canvas.toDataURL();
      ctx.fillStyle='black';ctx.fillRect(0,0,64,40);ctx.fillStyle='white';ctx.fillRect(5,10,11,21);const mask=canvas.toDataURL();
      return {name:'Wanda fixture',size:[64,40],images:[image,image,image],masks:[mask,mask,mask],timestamps:['0','1/60','1/30'],keyframes:{},issues:[]};
    });
    async function open(mode,config=data){await page.evaluate(({mode,config})=>{window.result=null;window.pending=vosTargetEditor({...config,mode}).then(r=>window.result=r);},{mode,config});await ready(0);}
    async function ready(n){await page.waitForFunction(n=>document.querySelector('canvas[data-role="canvas"]')?.dataset.frame===String(n)&&document.querySelector('canvas[data-role="canvas"]').dataset.loading==='false',n);}
    async function click(x,y){const box=await page.locator('[data-role=canvas]').boundingBox();await page.mouse.click(box.x+x*box.width,box.y+y*box.height);}
    await open('prompts');
    await page.locator('[data-role=save]').click();assert.equal(await page.evaluate(()=>window.result),null);
    await click(.15,.5);await page.locator('[data-role=tool]').selectOption('negative');await click(.8,.5);
    await page.locator('[data-role=next]').click();await ready(1);await page.locator('[data-role=tool]').selectOption('positive');await click(.16,.5);
    await page.locator('[data-role=save]').click();const selected=await page.evaluate(()=>window.pending);
    assert.equal(selected.keyframes['0'].points.length,2);assert.equal(selected.keyframes['0'].points[1][2],0);assert.equal(selected.keyframes['1'].points[0][2],1);
    await open('review',{...data,keyframes:selected.keyframes});
    await page.locator('[data-role=save]').click();assert.equal(await page.evaluate(()=>window.result),null);
    await page.locator('[data-role=review]').check();await page.locator('[data-role=next]').click();await ready(1);
    await page.locator('[data-role=tool]').selectOption('paint');await page.locator('[data-role=brush]').fill('3');await click(.3,.5);
    await page.locator('[data-role=review]').check();
    // Editing an already-reviewed mask revokes the checkbox.
    await click(.31,.5);assert.equal(await page.locator('[data-role=review]').isChecked(),false);
    await page.locator('[data-role=review]').check();await page.locator('[data-role=next]').click();await ready(2);await page.locator('[data-role=review]').check();
    await page.locator('[data-role=save]').click();const reviewed=await page.evaluate(()=>window.pending);
    assert.deepEqual(reviewed.reviewed,[0,1,2]);assert.ok(reviewed.corrections['1']);assert.equal(reviewed.corrections['0'],undefined);
    // Exported brush mask is strictly binary; the other actor remains excluded.
    const pixels=await page.evaluate(async src=>{const i=new Image();i.src=src;await i.decode();const c=document.createElement('canvas');c.width=64;c.height=40;const ctx=c.getContext('2d');ctx.drawImage(i,0,0);return [...ctx.getImageData(0,0,64,40).data];},reviewed.corrections['1']);
    assert.ok(pixels.every((v,n)=>n%4===3?v===255:v===0||v===255));assert.equal(pixels[(20*64+50)*4],0);
    await open('prompts');await page.locator('[data-role=cancel]').click();assert.equal((await page.evaluate(()=>window.pending)).cancelled,true);
    assert.deepEqual(errors,[]);
    console.log('Chromium: selection, negative points, keyframes, brush, review invalidation, binary masks and cancel passed');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
