"""Optional Chromium E2E. See docs/FRAME_STUDIO_TESTS.md for setup."""
import asyncio,sys,json,zipfile,io
from pathlib import Path
from PIL import Image
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from frame_studio.tests.helpers import make_video
from playwright.async_api import async_playwright
ROOT=Path(os.getenv('BROWSER_RESULTS','/tmp/vos-browser-results'));ROOT.mkdir(parents=True,exist_ok=True)
BASE=os.getenv('STUDIO_URL','http://127.0.0.1:8765').rstrip('/')
PASSWORD=os.getenv('STUDIO_PASSWORD','teste-ui')
source=ROOT/'source.mp4'
if not source.exists():make_video(source,audio=True)
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
  page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[]
  page.on('pageerror',lambda error:errors.append(str(error)))
  await page.goto(BASE);await page.locator('#password').fill('incorreta');await page.locator('#loginForm button').click();await page.wait_for_function("()=>document.getElementById('loginError').textContent.includes('incorreta')")
  await page.locator('#password').fill(PASSWORD);await page.locator('#loginForm button').click();await page.locator('#studio').wait_for(state='visible')
  await page.locator('#videoInput').set_input_files(str(source));await page.locator('#editor').wait_for(state='visible',timeout=30000)
  for frame,count in [(6,2),(12,3)]:
   await page.locator('#cursor').evaluate("(el,n)=>{el.value=n;el.dispatchEvent(new Event('input',{bubbles:true}));}",frame)
   await page.locator('#cutHere').click();await page.wait_for_function(f"()=>document.querySelectorAll('.scene-item').length==={count}")
  await page.locator('#extractScope').select_option('scene');await page.locator('#extractScene').select_option('scene_002');await page.wait_for_timeout(400)
  await page.locator('#extractButton').click();await page.locator('#zipResult').wait_for(state='visible',timeout=30000)
  url=await page.locator('#zipDownload').get_attribute('href');response=await page.request.get(BASE+url);data=await response.body();assert response.status==200
  with zipfile.ZipFile(io.BytesIO(data)) as archive:
   manifest=json.loads(archive.read('manifest.json'));assert len(manifest['frames'])==6
   edited=ROOT/'edited.zip'
   with zipfile.ZipFile(edited,'w') as out:
    for item in archive.infolist():
     content=archive.read(item)
     if item.filename==manifest['frames'][2]['file']:
      b=io.BytesIO();Image.new('RGB',(160,96),(220,20,210)).save(b,format='PNG');content=b.getvalue()
     out.writestr(item.filename,content)
  await page.locator('[data-tab="import"]').click();await page.locator('#zipInput').set_input_files(str(edited));await page.locator('#frameComparison').wait_for(state='visible',timeout=30000)
  await page.wait_for_function("()=>document.getElementById('editedFrame').complete&&document.getElementById('editedFrame').naturalWidth>0")
  await page.screenshot(path=str(ROOT/'import-desktop.png'),full_page=True)
  await page.locator('#applyImport').click();await page.wait_for_function("()=>document.getElementById('applyImport').textContent.includes('aplicado')")
  await page.locator('#previewImport').click();await page.locator('#renderResult').wait_for(state='visible',timeout=30000)
  await page.wait_for_function("()=>document.getElementById('resultVideo').readyState>=2")
  duration=await page.locator('#resultVideo').evaluate('(v)=>v.duration');assert abs(duration-.2)<.002,duration
  await page.locator('#buildScope').select_option('full');await page.locator('#buildFps').select_option('60');await page.wait_for_timeout(450);await page.locator('#buildButton').click()
  await page.wait_for_function("()=>document.getElementById('resultScope').textContent==='Vídeo completo'",timeout=30000)
  await page.wait_for_function("()=>document.getElementById('resultVideo').readyState>=2")
  await page.locator('#resultVideo').evaluate('(v)=>{v.muted=true;v.playbackRate=1;return v.play();}')
  await page.wait_for_function("()=>document.getElementById('resultVideo').ended",timeout=8000)
  full_duration=await page.locator('#resultVideo').evaluate('(v)=>v.duration');assert abs(full_duration-.6)<.002
  assert await page.locator('#validationSummary .stat strong').first.text_content()=='36'
  await page.screenshot(path=str(ROOT/'assembly-desktop.png'),full_page=True)
  await page.set_viewport_size({'width':390,'height':844});await page.screenshot(path=str(ROOT/'assembly-mobile.png'),full_page=True)
  width=await page.evaluate('document.documentElement.scrollWidth');assert width<=391,width
  await page.locator('[data-tab="extract"]').click();await page.screenshot(path=str(ROOT/'extract-mobile.png'),full_page=True)
  assert not errors,errors
  result={'browser':'Chromium','flow':'login → upload → manual cuts → scene ZIP → edit one frame → import → compare → apply → scene preview → full 60 FPS export → normal playback','scene_duration':duration,'full_duration':full_duration,'output_frames':36,'mobile_width':width,'javascript_errors':errors}
  (ROOT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2));await browser.close()
asyncio.run(main())
