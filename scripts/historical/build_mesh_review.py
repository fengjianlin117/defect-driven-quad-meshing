"""Prepare the requested twelve-model, at-most-three-mesh manual review."""
from pathlib import Path
import json,shutil,urllib.request,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from audit_evidence import OUT,read,save,sha
import strong_static_controls as static
W=Path('research://windows-delivery/paper_stage_20260920')
D=W/'mesh_manual_review_v1'
R=OUT/'coordinator_v5_development/runs'

def draw(ax,vertices,faces,center,radius):
    polys=[vertices[np.asarray(f,int)] for f in faces]
    normals=np.array([np.cross(x[1]-x[0],x[2]-x[0]) for x in polys])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    light=np.array([.3,-.4,1.]);light/=np.linalg.norm(light)
    brightness=.65+.3*np.abs(normals@light)
    colors=np.c_[np.array([.70,.80,.86])[None,:]*brightness[:,None],np.ones(len(polys))]
    ax.add_collection3d(Poly3DCollection(polys,facecolors=colors,edgecolors='#293742',linewidths=.34))
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
    ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=40);ax.set_axis_off()

def main():
    D.mkdir(exist_ok=False);(D/'obj').mkdir();(D/'assets').mkdir();(D/'images').mkdir()
    plans=read(OUT/'coordinator_v5_development/plans.json');models=[];manifest=[]
    for p in plans:
        name=p['model'];folder=R/name/'full';summary=read(folder/'summary.json')['candidates'];decision=read(folder/'decision.json')
        base=next(c for c in summary if c['id']=='baseline')
        initial=[c for c in summary if c['id'].startswith('initial_') and c['strict_output_pass'] and c['quads']<=2048]
        first=min(initial,key=lambda c:(c['all_main_deficit'],c['symmetric_rms_h'],c['quads'],c['id'])) if initial else None
        final=next(c for c in summary if c['id']==decision['recommended_id'])
        accepted=set(decision['eligible_ids']);source=static.load_obj(p['source']);center=(source.vertices.min(0)+source.vertices.max(0))/2;scale=float(np.ptp(source.vertices,axis=0).max())
        graph=read(p['graph']);features=[]
        for e in graph['source_edges']:
            if e['layer']=='main':features.extend(((source.vertices[np.asarray(e['vertices'],int)]-center)/scale).round(7).tolist())
        panels=[];raw=[]
        for kind,title,candidate in zip(['baseline','one_shot','feedback'],['基线','一次性提案','反馈后最好推荐'],[base,first,final]):
            if candidate is None:
                panels.append(dict(kind=kind,title=title,available=False,note='初始阶段没有通过统一几何检查的候选，未用基线冒充。'))
                raw.append(None);continue
            path=Path(p['source']).parent/'baseline/quad.obj' if candidate['id']=='baseline' else Path(candidate['quad_path'])
            mesh=static.load_obj(path,require_triangles=False);assert all(len(f)==4 for f in mesh.faces)
            local=D/'obj'/f'{name}_{kind}.obj';shutil.copy2(path,local);assert sha(local)==sha(path)
            if kind=='baseline':note='原方向场＋后端输出'
            elif kind=='one_shot':note='通过原保护条件' if candidate['id'] in accepted else '几何检查通过；未通过原保护条件'
            elif candidate['id']=='baseline':note='反馈未选出可接受改善，回退到基线'
            elif candidate['id'].startswith('initial_'):note='反馈后仍保留一次性提案'
            else:note='反馈产生的新结果；通过原保护条件'
            panels.append(dict(kind=kind,title=title,available=True,note=note,candidate=candidate['id'],quads=mesh.face_count,
                deficit=candidate['all_main_deficit'],rms=candidate['symmetric_rms_h'],accepted=candidate['id'] in accepted,
                vertices=((mesh.vertices-center)/scale).round(8).tolist(),faces=[list(map(int,f)) for f in mesh.faces],
                obj=f'obj/{name}_{kind}.obj'))
            raw.append(mesh)
            manifest.append(dict(model=name,kind=kind,candidate=candidate['id'],source_path=str(path),review_path=str(local),sha256=sha(path),
                quads=mesh.face_count,strict=candidate['strict_output_pass'],original_protection_eligible=candidate['id'] in accepted,
                summary_path=str(folder/'summary.json'),decision_path=str(folder/'decision.json'),note=note))
        model=dict(name=name,panels=panels,features=features)
        models.append(model)
        fig=plt.figure(figsize=(15,5),layout='constrained')
        for j,(panel,mesh) in enumerate(zip(panels,raw)):
            ax=fig.add_subplot(1,3,j+1,projection='3d')
            if mesh is None:
                ax.set_axis_off();ax.text2D(.5,.5,'No geometrically valid\none-shot candidate',ha='center',va='center',transform=ax.transAxes,color='#a34e13',fontsize=12)
                ax.set_title('One-shot proposal',fontsize=13);continue
            draw(ax,mesh.vertices,mesh.faces,center,scale*.53)
            label=['Baseline','One-shot proposal','After feedback'][j]
            status=' | fallback' if j==2 and panel['candidate']=='baseline' else ' | initial retained' if j==2 and panel['candidate'].startswith('initial_') else ' | protection not passed' if j==1 and not panel['accepted'] else ''
            ax.set_title(f"{label}: {panel['quads']} quads{status}\nD={panel['deficit']:.3f}; RMS/h={panel['rms']:.5f}",fontsize=11)
        fig.suptitle(f'{name} — same source coordinates, scale and camera',fontsize=15)
        fig.savefig(D/'images'/f'{name}.png',dpi=160);plt.close(fig)
        print('REVIEW_MODEL',name,[(x['kind'],x.get('quads')) for x in panels],flush=True)
    save(D/'provenance.json',dict(selection=dict(
        baseline='Original frozen same-field backend baseline, not a newly count-calibrated control.',
        one_shot='From initial phase of frozen v5 full run: basic+both-diagonal intersection-valid, <=2048 quads; minimize full-reference D, then RMS, Q, id. Original protection eligibility shown separately.',
        feedback='Frozen v5 full decision recommended_id: best under unchanged protection and geometry gates, including initial retention or baseline fallback.',
        missing='No valid initial candidate remains empty; no substitutions. At most three quad meshes per model.',
        display='Original OBJ copied byte-for-byte. Viewer transforms only by common source center/scale; true quad edges, no triangulation diagonals. Figures use identical source scale/camera per row.',
        warning='Actual counts differ. Manual review panel is not a claim of equal-count superiority.'),meshes=manifest))
    (D/'mesh-data.js').write_text('window.MESH_REVIEW_DATA='+json.dumps(models,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    url='https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js'
    data=urllib.request.urlopen(url,timeout=45).read();assert len(data)>100000
    (D/'assets/three.min.js').write_bytes(data)
    save(D/'assets/source.json',dict(url=url,sha256=hashlib.sha256(data).hexdigest(),version='0.160.0',purpose='Local offline mesh rendering library'))
    shutil.copy2(Path(__file__).with_name('mesh_review_template.html'),D/'index.html')
    lines=['# 12模型人工网格检查','',
           '打开index.html可逐个模型旋转、放大，并同步三个视角。images目录有每个模型的统一视角PNG。obj目录保存原始四边形OBJ副本。','',
           '每个模型最多三份：原基线；一次性阶段几何有效且缺陷D最小的候选（保护未通过会标记）；反馈阶段按原保护与有效性规则选出的最终最好推荐。反馈失败时明确标记回退，不视为改善。初始没有有效结果时留空。','',
           '展示不同实际面数，只供人工检查，不等于同面数质量排名。完整来源、选择规则和SHA256见provenance.json。','',
           '|模型|基线Q|一次性Q|反馈后Q|','|---|---:|---:|---:|']
    for model in models:lines.append('|'+model['name']+'|'+'|'.join(str(p.get('quads','无有效候选')) for p in model['panels'])+'|')
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('MESH_REVIEW_COMPLETE',len(models),len(manifest),flush=True)

if __name__=='__main__':main()
