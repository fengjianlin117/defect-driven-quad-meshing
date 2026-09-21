"""Persist versions, dependency hashes and checkpoint continuation state."""
from pathlib import Path
import subprocess,sys,platform,shutil,datetime,json
from audit_evidence import ROOT,OLD,OUT,read,save,sha
W=Path('research://windows-delivery/paper_stage_20260920')
def command(argv):
    p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def main():
    d=OUT/'checkpoint_02';d.mkdir(exist_ok=False);env=d/'environment';env.mkdir()
    versions=dict(python=sys.version,python_executable=sys.executable,platform=platform.platform(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        packages=command([sys.executable,'-m','pip','freeze','--all']),compiler=command(['g++','--version']),cmake=command(['cmake','--version']),
        repo_head=command(['git','-C',str(ROOT),'rev-parse','HEAD']),repo_status=command(['git','-C',str(ROOT),'status','--short']))
    binaries=[ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'/n for n in ['miq_adaptive','qex_adapter']]
    binaries += [OUT/'external/QuadriFlow/build/quadriflow',OUT/'external/QuadriFlow/build_sat/minisat']
    versions['binaries']=[dict(path=str(p),sha256=sha(p),ldd=command(['ldd',str(p)])) for p in binaries]
    save(env/'versions.json',versions)
    deps=Path('research://backend-dependencies/libs')
    lib_files=[deps/'libigl/include/igl/copyleft/comiso/miq.cpp',ROOT/'backend/CMakeLists.txt',ROOT/'weak_layout_pipeline/backend/run_weak_layout_miq.cpp']
    save(env/'critical_source_hashes.json',{str(p):sha(p) for p in lib_files})
    pairs=read(OUT/'diagnostic_sensitivity_v1/paired_summary.json');integrity=read(OUT/'diagnostic_sensitivity_v1/integrity.json');replay=read(OUT/'checkpoint_replay_v1/summary.json')
    save(d/'sensitivity_pairs.json',pairs);save(d/'sensitivity_integrity.json',integrity);save(d/'replay_summary.json',replay)
    old_changed=[p for p,h in read(OLD/'final_verified_hashes.json').items() if not Path(p).is_file() or sha(p)!=h]
    tracked_changed=[p for p,h in read(OUT/'audit/starting_tracked_hashes.json').items() if sha(ROOT/p)!=h]
    assert not old_changed and not tracked_changed
    save(d/'integrity.json',dict(protected_files=4613,changed=old_changed,changed_tracked_files=tracked_changed,
        experiment_backend_calls=71,replay_backend_calls=2,diagnostic_remeasurement_cases=17,settings_each=9,independent_new_test_models=0,goal_status='active'))
    text=['# 检查点02：诊断敏感性、重放与继续执行入口','',
        '本goal保持active。当前执行进程均已完成；不要按旧session ID恢复，也不要在已有实验输出目录重跑。',
        '', '## 新完成核验','',
        '17个固定输出，各9组诊断设置（距离0.05/0.1/0.2h × 方向10/15/20度），原0.1h/15度指标全部复现；无新后端调用，也未据此替换阈值。',
        '', '|模型|反馈|相对原allocated对照缺陷更低的设置数|', '|---|---|---:|']
    for x in pairs:text.append(f'|{x["model"]}|{x["feedback"]}|{x["lower_deficit_settings"]}/9|')
    text+=['','S是尺寸反馈，C是约束反馈。这是固定候选的测量敏感性，不是完整方法的阈值重跑，也不建立CAD工程公差。实际面数仍按检查点01报告，不能因9/9就省略面数差异。',
        '', 'B49新反馈528Q和B32 QuadriFlow sharp516Q各自精确重放，OBJ均逐字节一致。两次重放不增加独立模型数，也不能代替不同种子/场的稳定性。',
        '', '截至本检查点：实验后端调用71，重放2，共73；新未见模型0；旧4613保护文件和开始时仓库跟踪文件再次核验无变更。',
        '', '## 下一项必须继续的研究工作','',
        '1. 统一有预算的实测反馈策略：保留初始uniform/allocated的有效中间候选；区分新增必要约束、不可解除的组内硬冲突、尺寸重分配和全局数量不足；引入有界回退。当前两个反馈试验仍是分开的单步开发机制。',
        '2. 在完整12例上做方法消融与步长/数量敏感性，特别保留B32/B43/B33结构失败、B65保护失败。共同全参考评价需继续补强；不得看测试结果放宽保护。',
        '3. 外部对照已有默认/sharp/sharp+SAT的12例开发矩阵；尚非最终同实际面数质量曲线。要保留各配置强项，验证构建/运行局限，补适用的CAD方法。',
        '4. 补全全局相交检查、关键CAD尺寸指标、随机稳定性与独立族数据选择协议。只有最终方法冻结后才能运行新的未见测试；现有12例永远保持已见身份。',
        '5. 继续核读一手文献与方法新颖性；再写完整论文、图表和可迁移复现包。当前是阶段证据，不是完整投稿证据。',
        '', '## 可直接使用的路径','',
        f'- 新证据根：`{OUT}`。',
        '- 首读 `checkpoint_01/CHECKPOINT_REPORT.md`、本文件、`CLAIM_EVIDENCE_MATRIX.md`、`LITERATURE_AUDIT.md`。',
        '- `feedback_pilot_v1`：冻结尺寸反馈、两个相同实验臂在输出前去重、全部6次尝试。',
        '- `feedback_matched_controls_v1`：3例原尺寸与均匀尺寸的6次实际面数对照。',
        '- `constraint_feedback_pilot_v1`：12例全部诊断；B53/B65各两路径，共4调用；组内冲突的跳过原因保留。',
        '- `quadriflow_development_v1`：12例3配置共55调用；2个SAT终次过程失败以及所有几何失败保留。',
        '- `constraint_certificates`：独立路径证书与实际MIQ源代码规则；`localization`：12例逐边诊断。',
        '- `diagnostic_sensitivity_v1`：17候选的原始距离NPZ、153项诊断与汇总。',
        '- `checkpoint_replay_v1`：两次关键重放。',
        '- `scripts`：Python执行入口；Windows同目录也有脚本。',
        '', '运行环境在本目录environment/versions.json。原实验脚本均故意拒绝覆盖输出；需要重放时使用新的独立目录及明确路径映射。目前源文件依赖绝对路径，可迁移打包尚未完成。',
        '', '无commit/push，无对外投稿、邮件或私有数据上传。已使用的网络仅用于公开一手文献与官方QuadriFlow代码。']
    (d/'CONTINUE.md').write_text('\n'.join(text),encoding='utf-8')
    shutil.copytree(d,W/'checkpoint_02')
    shutil.copytree(W/'scripts',OUT/'scripts',dirs_exist_ok=True)
    shutil.copy2(d/'CONTINUE.md',W/'ACTIVE_RESEARCH_STATE.md')
    print('CHECKPOINT_02_SAVED',flush=True)
if __name__=='__main__':main()
