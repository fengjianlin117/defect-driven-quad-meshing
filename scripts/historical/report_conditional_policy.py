"""Report conditional execution and show every outcome category, including failure."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
from audit_evidence import OUT,read,save,sha
import strong_static_controls as static
W=Path('research://windows-delivery/paper_stage_20260920')
D=OUT/'conditional_policy_report_v1'
R=OUT/'conditional_policy_development_v1/runs'

def main():
    integrity=read(R/'integrity.json');assert integrity['complete']
    rows=read(R/'summary.json');D.mkdir(exist_ok=False)
    save(D/'recommendations.json',rows);save(D/'integrity.json',integrity)
    labels={'baseline_no_diagnostic_feature_defect':'基线诊断无需干预','initial_proposal_accepted':'一次性提案接受','feedback_or_bounded_fallback':'进入反馈'}
    lines=['# 按需介入：一次性主体与条件反馈','',
           '按用户提出的论文叙事实现并冻结新的控制策略：基线没有诊断特征缺陷且有效时直接保留；否则执行原一次性提案阶段；初始阶段没有可接受新候选时才进入反馈。缺陷、保护与有效性条件均未放宽。', '',
           '12个已见模型的执行结果：1个无需新增求解，4个在一次性阶段接受，7个进入反馈，其中6个得到新候选、1个回退原基线。总逻辑尝试由无条件反馈的83次降到65次，减少18次（21.7%）；这不是21.7%的实测时间加速。', '',
           '执行对精确历史输入进行缓存重放，0次新后端求解。它验证控制流、输出身份和逻辑调用数，不是独立随机重跑或未见泛化。7个进入反馈的案例均与原完整反馈输出逐项一致。', '',
           '|模型|退出阶段|Q|D|RMS/h|条件调用数|无条件调用数|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['model']}|{labels[r['exit_stage']]}{'，回退' if r['exit_stage']=='feedback_or_bounded_fallback' and r['selected']=='baseline' else ''}|{r['quads']}|{r['deficit']:.5f}|{r['rms']:.6f}|{r['logical_attempts']}|{r['always_feedback_attempts']}|")
    lines += ['', '早停取舍：B70结果相同但节约4次尝试；B73放弃很小的后续改善；B68早停565Q而持续反馈838Q，质量和数量同时变化，不能据此作同数量优劣判断；B61早停556Q，而继续反馈512Q有更低缺陷与RMS。这些取舍必须保留。', '',
              '当前“接受”仍是相对基线的合计保护与严格输出检查，尚非标定的CAD工程公差。一次性阶段包含两种尺寸分支、各至多一次数量校正，不等于一次原生求解。', '',
              '示意图按事先明确的四种路径各取一个例子：B34无需干预，B70一次性接受，B65反馈接受，B43反馈失败回退。图不是全数据统计或同面数胜负证据；所用模型、结果、视角与路径见figure_sources.json。','']
    (D/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    plans={p['model']:p for p in read(OUT/'coordinator_v5_development/plans.json')}
    refs={r['model']:r for r in rows};sources=[]
    fig=plt.figure(figsize=(11,12),layout='constrained')
    names=['B34','B70','B65','B43'];route=['No intervention','One-shot accepted','Feedback accepted','Fallback after failure']
    for i,name in enumerate(names):
        p=plans[name];r=refs[name];basepath=Path(p['source']).parent/'baseline/quad.obj'
        summary=read(R/name/'summary.json')['candidates']
        selected=next(c for c in summary if c['id']==r['selected'])
        output=basepath if r['selected']=='baseline' else Path(selected['quad_path'])
        source=static.load_obj(p['source']);graph=read(p['graph'])
        center=(source.vertices.min(0)+source.vertices.max(0))/2;radius=np.ptp(source.vertices,axis=0).max()*.55
        paths=[Path(p['source']),basepath,output]
        for j,path in enumerate(paths):
            mesh=static.load_obj(path,require_triangles=(j==0));ax=fig.add_subplot(4,3,i*3+j+1,projection='3d')
            polys=[mesh.vertices[np.asarray(face,dtype=int)] for face in mesh.faces]
            coll=Poly3DCollection(polys,facecolor='#d9e6ec',edgecolor='none' if j==0 else '#334155',linewidth=.24,alpha=1)
            ax.add_collection3d(coll)
            if j==0:
                edges=[source.vertices[np.asarray(e['vertices'],int)] for e in graph['source_edges'] if e['layer']=='main']
                ax.add_collection3d(Line3DCollection(edges,colors='#c2410c',linewidths=1.15))
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=40);ax.set_axis_off()
            title=[f'{name}: source features',f'Existing baseline ({mesh.face_count} Q)',f'{route[i]} ({mesh.face_count} Q)'][j]
            ax.set_title(title,fontsize=10,pad=0)
        sources.append(dict(model=name,route=route[i],source=str(paths[0]),baseline=str(paths[1]),selected=str(paths[2]),hashes=[sha(f) for f in paths],view=dict(elevation=25,azimuth=40),decision=r['decision_path']))
    fig.suptitle('Demand-driven feature and sizing intervention\nSeen development examples; views illustrate execution paths, not count-matched superiority',fontsize=12)
    fig.savefig(D/'conditional_paths.png',dpi=170);fig.savefig(D/'conditional_paths.svg');plt.close(fig)
    save(D/'figure_sources.json',sources)
    shutil.copytree(D,W/D.name)
    print('CONDITIONAL_REPORT',integrity,flush=True)

if __name__=='__main__':main()
