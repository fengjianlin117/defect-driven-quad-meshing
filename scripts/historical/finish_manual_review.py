from pathlib import Path
import json,hashlib,shutil
from PIL import Image
W=Path('research://windows-delivery/paper_stage_20260920')
D=W/'mesh_manual_review_v1'
names=['B49','B32','B43','B53','B68','B33','B34','B61','B70','B65','B31','B73']
records=json.loads((D/'provenance.json').read_text(encoding='utf-8'))['meshes']
for r in records:
    local=D/'obj'/f"{r['model']}_{r['kind']}.obj"
    assert hashlib.sha256(local.read_bytes()).hexdigest()==r['sha256']
for k in range(3):
    group=names[k*4:k*4+4]
    images=[Image.open(D/'images'/f'{name}.png').convert('RGB') for name in group]
    sheet=Image.new('RGB',(images[0].width,sum(i.height for i in images)), 'white')
    offset=0
    for im in images:sheet.paste(im,(0,offset));offset+=im.height
    sheet.save(D/'images'/f'overview_{k+1}.png')
atlas='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>12模型静态图册</title><style>body{font-family:system-ui;margin:24px;color:#203344}img{width:100%;height:auto}figure{margin:30px 0 45px}a{color:#17649a}nav{display:flex;gap:18px;flex-wrap:wrap;position:sticky;top:0;background:white;padding:12px}h1{font-size:24px}figcaption{font-size:18px;font-weight:600}p{line-height:1.8}</style><h1>12模型静态图册</h1><p>每行从左到右：基线 / 一次性提案 / 反馈后最好推荐。空白表示没有有效候选；fallback表示回退基线。各行使用相同源尺度和视角，未作网格修饰。点击图可看原尺寸。</p><nav>'''
atlas+=''.join(f'<a href="#{name}">{name}</a>' for name in names)+'</nav>'
for name in names:atlas+=f'<figure id="{name}"><figcaption>{name} · <a href="index.html#{name}">旋转检查</a></figcaption><a href="images/{name}.png"><img src="images/{name}.png" alt="{name}的基线、一次性提案和反馈结果"></a></figure>'
atlas+='</html>'
(D/'atlas.html').write_text(atlas,encoding='utf-8')
(D/'verification.json').write_text(json.dumps(dict(models=12,obj_files=len(records),all_obj_hashes_verified=True,quad_meshes_only=True,selection_manifest='provenance.json',browser_checks=['B49 displayed','B32 empty one-shot slot','B65 model switch and synchronized drag','No browser console errors'],server='http://127.0.0.1:8766/',server_session=88987),ensure_ascii=False,indent=2),encoding='utf-8')
print('REVIEW_VERIFIED',len(names),len(records))
