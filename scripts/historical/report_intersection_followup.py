"""Report every strict-audit disagreement, including baseline and competitor results."""
from pathlib import Path
import sys,shutil
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from audit_evidence import OUT,read,save,sha
W=Path('research://windows-delivery/paper_stage_20260920')

def main():
    root=OUT/'intersection_soup_audit_v2'; rows=read(root/'summary.json');integ=read(root/'integrity.json')
    dest=OUT/'intersection_review_v2';dest.mkdir(exist_ok=False)
    unique={r['sha256']:r for r in rows};bad=[r for r in unique.values() if not r['clear_both_triangulations']]
    disagreements=[r for r in rows if r.get('basic_output_pass') is True and not r['clear_both_triangulations']]
    save(dest/'basic_pass_but_intersection_nonclear.json',disagreements)
    audit={r['sha256']:r for r in rows};recs=read(OUT/'checkpoint_03/recommendations.json');full=[r for r in recs if r['variant']=='full']
    assert all(r['both_completed'] for r in rows)
    baseline=[r for r in rows if r['role']=='baseline'];assert len(baseline)==12
    result=dict(**integ,unique_geometry_with_intersections=len(bad),source_clear=sum(r['clear_both_triangulations'] for r in rows if r['role']=='source'),
        baseline_clear=sum(r['clear_both_triangulations'] for r in baseline),
        full_recommendations_clear=sum(audit[sha(r['quad'])]['clear_both_triangulations'] for r in full),
        full_nonbaseline_and_clear=sum(r['recommended']!='baseline' and audit[sha(r['quad'])]['clear_both_triangulations'] for r in full),
        basic_pass_but_nonclear_unique=len({r['sha256'] for r in disagreements}),
        interpretation='Post hoc developmental audit; original decision records remain unchanged. This is not a new method selection rule.')
    save(dest/'summary.json',result)
    external=[]
    for variant in ['default','sharp','sharp_sat']:
        groups={}
        for r in read(OUT/'quadriflow_development_v1/runs/summary.json'):
            if r['variant']==variant:groups[r['model']]=r
        terminal=list(groups.values());nclear=0;both=0;matched=0
        for r in terminal:
            clear=Path(r['quad_path']).is_file() and audit[sha(r['quad_path'])]['clear_both_triangulations']
            nclear+=bool(clear);ok=r.get('basic_output_pass') is True and clear;both+=bool(ok)
            matched+=bool(ok and abs(r['quads']/r['target_quads']-1)<=.05)
        external.append(dict(variant=variant,terminal_geometry_clear=nclear,terminal_basic_and_clear=both,terminal_basic_clear_and_count_within_5_percent=matched))
    save(dest/'external_terminal_audit.json',external)
    p=next(r for r in full if r['model']=='B31');q=audit[sha(p['quad'])];checks=read(q['result'])
    sys.path.insert(0,str(OUT/'coordinator_v4_development/method'));from weak_layout_pipeline.pipeline.mesh import load_obj
    mesh=load_obj(p['quad'],require_triangles=False);pairs=checks[1]['pairs'];ids=sorted(set(i for pair in pairs for i in pair['polygons']))
    polygons=[mesh.vertices[mesh.faces[i]] for i in ids];verts=np.concatenate(polygons)
    center=verts.mean(0);radius=np.max(np.ptp(verts,axis=0))*.65
    fig=plt.figure(figsize=(11,5),layout='constrained')
    colors=['#ef4444','#2563eb']
    for j,angle in enumerate([(20,30),(65,115)]):
        ax=fig.add_subplot(1,2,j+1,projection='3d')
        for k,(idx,polygon) in enumerate(zip(ids,polygons)):
            ax.add_collection3d(Poly3DCollection([polygon],facecolors=colors[k%2],edgecolors=colors[k%2],alpha=.4,linewidths=1.5))
            c=polygon.mean(0);ax.text(*c,str(idx),fontsize=10)
        for pair in pairs:
            for ti in pair['triangles']:
                quad=np.roll(mesh.faces[ti//2],-1)
                tri=quad[[0,1,2] if ti%2==0 else [0,2,3]];points=mesh.vertices[np.r_[tri,tri[0]]]
                ax.plot(*points.T,color='#111827',lw=.7)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1));ax.view_init(*angle);ax.set_title('B31 intersecting polygons '+', '.join(map(str,ids)))
    fig.suptitle('Independent exact-predicate audit: two intersecting triangle pairs\nAlternate diagonal only; polygon IDs are zero-based; visualization is not the proof')
    fig.savefig(dest/'B31_intersection.png',dpi=180);fig.savefig(dest/'B31_intersection.svg');plt.close(fig)
    save(dest/'B31_witness.json',dict(record=p['record'],quad=p['quad'],sha256=sha(p['quad']),checks=checks,
        polygon_vertices=[dict(polygon_id=i,vertex_ids=mesh.faces[i].tolist(),coordinates=mesh.vertices[mesh.faces[i]].tolist()) for i in ids]))
    text=['# 全局相交审计补充：拓扑无关的复核','',
        '检查点03中的Surface_mesh检查存在替代对角线插入失败；本补充用CGAL官方triangle_soup_self_intersections完整检查同一批三角片，不修改几何、旧报告或原方法决策。','',
        f"149份独特几何的298次检查全部完成；此前可完成的{integ['parity_checks']}次检查，具体相交三角片对集合全部一致。12个人工夹具检查通过。",'',
        f"12个源模型与12个基线均在两种剖分下无相交/退化。149份独特几何中{len(bad)}份检测到相交；包括{result['basic_pass_but_nonclear_unique']}份旧基本检查已通过的输出。",'',
        'B31全方法推荐为493Q，沿对角线0检查无相交，沿对角线1有两对相交三角片，涉及原四边形348和492（零起始）。旧基本检查不足以排除这类问题。B31基线的旧“不完整”经本检查确认为两种剖分均无相交。','',
        '因此，v4的12个推荐中11个通过此项审计，9个非基线推荐中只有8个同时通过。原始推荐保留供审查；不得将9/12称为通过完整几何审查的成功率。','',
        '外部方法同标准复核（每模型的最后一次尝试，含过程失败）：','',
        '|QuadriFlow配置|基本检查且无相交/12|再满足数量误差≤5%/12|','|---|---:|---:|']
    for r in external:text.append(f"|{r['variant']}|{r['terminal_basic_and_clear']}|{r['terminal_basic_clear_and_count_within_5_percent']}|")
    text+=['','不得将这个开发数据上的新审计直接解释为方法优越性。新的统一有效性判定应在下一方法版本中预先固定，并对所有候选、基线和外部方法一致执行。对曲面双线性四边形仍无全局单射保证。','',
        'B31_witness.json保存文件散列、原四边形顶点坐标与精确检查的相交对；PNG/SVG用于定位，计算证据为原始检查JSON。','']
    (dest/'REVIEW.md').write_text('\n'.join(text),encoding='utf-8')
    shutil.copytree(dest,W/'intersection_review_v2');print('INTERSECTION_REVIEW',result,external,flush=True)

if __name__=='__main__':main()
