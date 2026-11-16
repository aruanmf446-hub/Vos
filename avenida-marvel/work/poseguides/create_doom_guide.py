from pathlib import Path
from PIL import Image,ImageDraw
import json,math
root=Path(__file__).parent
# Manually estimated screen-space geometry from the active source crop only.
# Coordinates describe neutral pose, not facial identity or appearance.
rows=[
(272,(351,85),(296,10,410,164),(380,165),(443,191),(497,308),(460,380),(445,394),(361,186),(343,293),(332,380),(324,394)),
(282,(365,85),(307,10,424,166),(389,168),(452,195),(492,306),(430,380),(412,393),(370,188),(351,293),(345,378),(333,390)),
(292,(389,99),(332,24,448,176),(410,178),(473,200),(506,303),(455,356),(435,367),(389,192),(370,292),(385,368),(372,383)),
(302,(377,105),(320,31,437,182),(402,187),(460,215),(497,303),(405,350),(375,352),(382,201),(363,299),(357,369),(341,382)),
(312,(355,113),(296,46,414,188),(385,187),(437,214),(474,309),(390,356),(362,365),(366,207),(354,303),(355,371),(339,384)),
(322,(355,100),(297,26,415,176),(380,179),(432,205),(460,286),(370,336),(337,346),(364,197),(349,286),(356,352),(338,365)),
(332,(372,113),(313,38,431,187),(401,187),(464,223),(500,294),(427,334),(395,341),(381,206),(369,302),(376,348),(359,361)),
(342,(385,115),(327,39,443,189),(413,190),(466,218),(517,281),(486,325),(480,347),(392,211),(381,302),(387,328),(367,337)),
(348,(377,108),(318,33,438,183),(401,186),(468,214),(522,282),(487,320),(482,340),(383,207),(376,294),(398,322),(376,332)),
]
canvas=Image.new('RGB',(1728,1092),'white');records=[]
for n,row in enumerate(rows):
 frame,head,hbox,neck,ns,ne,nw,nh,fs,fe,fw,fh=row
 im=Image.new('RGB',(576,364),'white');d=ImageDraw.Draw(im)
 # Plain neutral torso and skeletal joints. Deliberately no source pixels/facial features.
 torso=[(neck[0]-27,neck[1]+7),(ns[0]+8,ns[1]-6),(min(575,ns[0]+85),410),(neck[0]-37,410)]
 d.polygon(torso,fill=(238,238,238),outline=(115,115,115))
 d.line([fs,fe,fw,fh],fill=(160,160,160),width=10)
 d.line([ns,ne,nw,nh],fill=(35,104,169),width=11)
 d.line([neck,head],fill=(75,75,75),width=10)
 d.ellipse(hbox,fill=(248,248,248),outline=(40,40,40),width=3)
 # Orientation arrow inside head encodes left-facing profile without eyes or facial details.
 end=(head[0]-43,head[1]+3)
 d.line([head,end],fill=(170,70,30),width=4)
 d.polygon([end,(end[0]+12,end[1]-7),(end[0]+12,end[1]+7)],fill=(170,70,30))
 for p in [neck,ns,ne,nw,fs,fe,fw]:
  d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=(255,255,255),outline=(55,55,55),width=2)
 # Generic open hands follow the observed arm gesture; fingers have no identity information.
 for center,wrist,col in [(nh,nw,(35,104,169)),(fh,fw,(140,140,140))]:
  cx,cy=center;d.ellipse((cx-12,cy-8,cx+12,cy+8),fill=col)
  if frame>=302:
   vx,vy=cx-wrist[0],cy-wrist[1];ang=math.atan2(vy,vx)
   for off in [-.5,-.25,0,.25,.5]:
    length=17-abs(off)*6
    tip=(round(cx+length*math.cos(ang+off)),round(cy+length*math.sin(ang+off)))
    d.line([center,tip],fill=col,width=3)
 d.text((8,8),f'{n+1}  frame {frame}  ({frame/30:.2f}s)',fill=(80,80,80))
 d.text((8,25),'neutral pose / facing screen-left',fill=(120,120,120))
 d.rectangle((0,0,575,363),outline=(200,200,200),width=1)
 canvas.paste(im,((n%3)*576,(n//3)*364))
 records.append({'panel':n+1,'source_frame':frame,'source_time_seconds':frame/30,'coordinate_canvas':[576,364],'head_center':list(head),'head_bbox':list(hbox),'facing':'screen-left profile','yaw_degrees':-75,'neck':list(neck),'near_shoulder':list(ns),'near_elbow':list(ne),'near_wrist':list(nw),'near_hand_center':list(nh),'far_shoulder':list(fs),'far_elbow':list(fe),'far_wrist':list(fw),'far_hand_center':list(fh),'torso_polygon':[list(p) for p in torso],'crop':'waist-up; lower body outside image','estimate_method':'manual screen-space measurements; neutral geometry only','facial_identity_data':False})
canvas.save(root/'doom-walk.png')
(root/'doom-walk.json').write_text(json.dumps({'grid':[3,3],'panel_size':[576,364],'output_size':[1728,1092],'contains_source_photo_pixels':False,'poses':records},indent=2))
print(root/'doom-walk.png');print(root/'doom-walk.json')
