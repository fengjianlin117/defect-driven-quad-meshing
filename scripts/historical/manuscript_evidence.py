"""Bind manuscript numerical statements to authoritative development records."""
from pathlib import Path
import shutil
from audit_evidence import OUT,read,save,sha
W=Path('research://windows-delivery/paper_stage_20260920')

def main():
    d=W/'manuscript';claims=[]
    def add(identifier,claim,path,values,scope):claims.append(dict(id=identifier,claim=claim,record=str(path),sha256=sha(path),values=values,scope=scope))
    path=OUT/'checkpoint_03/ablation_summary.json';r=read(path)
    add('C01','5个方法版本的推荐、尝试和复用表',path,r,'12个已见模型；先full后消融，复用不增加独立证据；推荐尚未通过新增相交门槛。')
    path=OUT/'checkpoint_03/recommendations.json';r=read(path);full={x['model']:x for x in r if x['variant']=='full'}
    add('C02','B32/B33恢复数量和缺陷；6个推荐有新增缺陷',path,
        dict(models={m:{k:full[m][k] for k in ['baseline_quads','quads','baseline_deficit','all_main_deficit']} for m in ['B32','B33']},
            recommended_with_new_deficiencies=sum(x['newly_deficient_edges']>0 for x in full.values())),
        '增加面数成本必须同时报告；不是同成本优越性或跨族成功率。')
    path=OUT/'checkpoint_01/candidate_comparison.json';r=read(path)
    add('C03','B49三个528Q条件与B53尺寸反馈反例',path,[x for x in r if (x['model']=='B49' and x['condition'] in ['U','A','S']) or (x['model']=='B53' and x['condition'] in ['A','S'])],
        '单模型机制结果；B53比较并非精确同面数，保留实际面数。')
    path=OUT/'coordinator_v4_development/runs/B53/full/summary.json';r=read(path)
    add('C04','B53初始与约束反馈同为503Q的缺陷',path,[x for x in r['candidates'] if x['id'] in ['initial_allocated_1','r0_constraint_0']],
        '冻结v4的一对条件，不把该对当作总体胜率。')
    path=OUT/'intersection_review_v2/summary.json';r=read(path)
    add('C05','149份几何、298次检查、18份相交、7份旧基本通过、11个推荐clear',path,r,
        '三角片嵌入审计，无法证明双线性曲面单射；输出数不是独立模型数。')
    path=OUT/'intersection_review_v2/B31_witness.json';r=read(path)
    add('C06','B31相交原四边形348和492',path,dict(quad=r['quad'],checks=r['checks']),
        '具体已见反例；零起始面编号；原方法记录未改写。')
    path=OUT/'intersection_review_v2/external_terminal_audit.json';r=read(path)
    add('C07','QuadriFlow三配置终次有效性12/5/6，兼数量7/4/5',path,r,
        '旧注册目标的外部开发矩阵，尚非v4推荐实际面数匹配曲线；无端到端成本胜负。')
    path=OUT/'cad_dimension_proxies_v1/integrity.json';r=read(path)
    add('C08','77个候选引用、31份独特几何的尺寸代理',path,r,
        '无工程单位/精确B-Rep公差保证，31不是独立模型数。')
    path=OUT/'checkpoint_02/sensitivity_pairs.json';r=read(path)
    add('C09','B49反馈9/9、B53尺寸反馈6/9测量设置',path,r,
        '固定输出的测量敏感性，不是算法阈值重跑或工程公差。')
    path=OUT/'coordinator_v4_cachefree_replay/integrity.json';r=read(path)
    add('C10','B53无历史缓存8调用、全部几何和指标一致',path,r,
        '相同输入重放，不是独立随机种子或不同场稳定性。')
    path=OUT/'checkpoint_03_final/integrity.json';r=read(path)
    add('C11','旧4613项文件未变',path,dict(old_protected_files=r['old_protected_files'],changed_old_files=r['changed_old_files']),
        '完整性证据不构成研究假设成立的证据。')
    path=OUT/'strong_static_controls_v1/protocol.json';r=read(path)
    add('C12','72格强静态比较的冻结协议',path,r,
        '研究假设检验；运行状态与结果另列，不能从协议推断胜负。')
    save(d/'定量主张证据映射.json',claims)
    lines=['# 论文v0.1定量主张与证据','',
        '主稿为内部开发初稿，尚不能投稿。这里列出实验数值的直接来源；算法常数对应冻结方法源码和protocol。强静态结果完成后单独生成附页，不回填预设胜率。','']
    for c in claims:
        lines+=['## '+c['id']+' '+c['claim'],'',c['scope'],'',f"记录：`{c['record']}`",f"SHA256：`{c['sha256']}`",'']
    lines+=['## 算法、样本和引用的边界','',
        '- 缺陷定义与采样：coordinator_v4_development/method/defect_predictor_v1.py；无固定比例的必要性选择实际使用defect_driven_selection.py，不调用旧预算比例接口。',
        '- 源几何尺寸常数：geometry_budget.py；反馈与上限：coordination.py、coordinator.py。',
        '- 相容性证书：constraint_certificates；原MIQ约束规则与独立路径检查对应记录。',
        '- 数据身份：最初交接及audit/source_plans.json。所有12例均已见，尚无新测试。',
        '- 文献正文核读与下载散列：LITERATURE_AUDIT.md、literature/sources.json。近年相关工作查新未完成。',
        '- method_v5/coordinator_v5.py是尚未测试、冻结、运行的工作草案；不属于论文v0.1已验证方法，也不能混入结果。','']
    (d/'定量主张证据映射.md').write_text('\n'.join(lines),encoding='utf-8')
    target=OUT/'manuscript_v0_1';target.mkdir(exist_ok=False)
    for f in d.iterdir():
        if f.is_file():shutil.copy2(f,target/f.name)
    save(target/'hashes.json',{str(f):sha(f) for f in target.iterdir() if f.is_file()})
    print('MANUSCRIPT_EVIDENCE',len(claims),'mapped',flush=True)

if __name__=='__main__':main()
