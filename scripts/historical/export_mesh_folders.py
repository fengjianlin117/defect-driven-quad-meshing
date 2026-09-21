"""Export the reviewed meshes and their exact triangle inputs into per-model folders."""
from pathlib import Path
import json,hashlib,shutil

ROOT=Path('research://windows-delivery')
REVIEW=ROOT/'paper_stage_20260920/mesh_manual_review_v1'
WSL=Path('//wsl.localhost/Ubuntu')
PLANS=WSL/'home/lenovo/projects/rubost-structural-pipeline/research_runs/2026-09-20_paper_stage_v1/coordinator_v5_development/plans.json'
DEST=ROOT/'十二模型_原网格与三阶段结果_20260920'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    plans=json.loads(PLANS.read_text(encoding='utf-8'))
    provenance=json.loads((REVIEW/'provenance.json').read_text(encoding='utf-8'))
    DEST.mkdir(exist_ok=False)
    manifest=[];missing=[]
    def copy_checked(source,destination,model,stage,expected=None):
        source_hash=digest(source)
        if expected is not None:assert source_hash==expected
        shutil.copy2(source,destination)
        assert digest(destination)==source_hash
        manifest.append(dict(model=model,stage=stage,source=str(source),file=str(destination.relative_to(DEST)),sha256=source_hash))
    for p in plans:
        name=p['model'];folder=DEST/name;folder.mkdir()
        source=WSL/p['source'].lstrip('/')
        copy_checked(source,folder/'00_原始三角网格.obj',name,'source')
        rows={r['kind']:r for r in provenance['meshes'] if r['model']==name}
        notes=[f'{name} 网格文件说明','',
               '文件是原始OBJ的逐字节副本；未缩放、旋转、平滑、修补或重新三角化。',
               '00_原始三角网格.obj 是本实验实际使用的输入三角网格。','']
        for kind,filename,title in [
            ('baseline','01_基线.obj','基线'),
            ('one_shot','02_一次性提案.obj','一次性提案'),
            ('feedback','03_反馈最终结果.obj','反馈最终结果')]:
            r=rows.get(kind)
            if r is None:
                assert kind=='one_shot'
                message=(f'{name} 的一次性提案阶段没有通过统一几何检查的候选。\n'
                         '与人工检查页面保持一致，此处不以基线或其他方法的网格代替。\n'
                         '原始失败记录保留在研究结果目录中；这不代表已删除失败实验。\n')
                (folder/'02_一次性提案_无有效结果.txt').write_text(message,encoding='utf-8-sig')
                notes.append('一次性提案：没有有效候选，见02说明文件。');missing.append(name)
                continue
            reviewed=REVIEW/'obj'/f'{name}_{kind}.obj'
            copy_checked(reviewed,folder/filename,name,kind,r['sha256'])
            notes.append(f"{filename}：{r['quads']}个四边形；候选 {r['candidate']}；{r['note']}。")
        notes+=['','一次性列选取初始阶段几何有效、完整参考缺陷最小的候选，是否通过原保护条件单独注明。',
                '反馈列沿用已冻结的最终推荐；回退或保留初始结果时会与另一个文件相同。',
                '三阶段实际面数可能不同，本文件夹用于人工检查，不等于同面数排名。']
        (folder/'说明.txt').write_text('\n'.join(notes)+'\n',encoding='utf-8-sig')
    assert len(plans)==12 and len(manifest)==45 and set(missing)=={'B32','B43','B33'}
    result=dict(models=12,obj_files=len(manifest),all_copies_hash_verified=True,missing_valid_one_shot=missing,
                selection=provenance['selection'],files=manifest)
    (DEST/'文件来源与校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'文件夹说明.txt').write_text(
        '每个模型一个文件夹，按00原始三角网格、01基线、02一次性提案、03反馈最终结果排列。\n'
        '共12个模型、45份OBJ，已逐个验证复制前后SHA256一致。\n'
        'B32、B43、B33没有有效一次性结果，对应位置放置说明文件。\n'
        '反馈最终结果可能回退到基线或保留初始提案，各模型说明.txt中已标记。\n',encoding='utf-8-sig')
    print(json.dumps(dict(directory=str(DEST),models=12,obj_files=45,hash_verified=True,missing_valid_one_shot=missing),ensure_ascii=False))

if __name__=='__main__':main()
