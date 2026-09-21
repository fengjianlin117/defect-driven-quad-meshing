from candidate_decision import decide,count_correction_target


def candidate(name,n,deficit,rms,valid=True,**extra):
    return dict(id=name,quads=n,all_main_deficit=deficit,symmetric_rms_h=rms,basic_output_pass=valid,process_success=True,**extra)


def test_better_feature_fit_does_not_hide_overbudget_or_broken_output():
    rows=[candidate('baseline',500,30,.05),candidate('broken',600,1,.01,False),candidate('expensive',3000,0,.001),candidate('joint',700,10,.02)]
    result=decide(rows,1000)
    assert result['recommended_id']=='joint'
    assert set(result['eligible_ids'])=={'baseline','joint'}
    assert 'actual_count_exceeds_resource_cap' in next(r for r in result['audit'] if r['id']=='expensive')['reasons']


def test_tradeoff_remains_visible_and_intermediate_uv_warning_is_not_veto():
    rows=[candidate('uniform',500,20,.01),candidate('allocated',480,10,.03,warnings=['uv_flips','projected_corner_quality'])]
    result=decide(rows,1000)
    assert result['recommended_id']=='allocated' and set(result['pareto_ids'])=={'uniform','allocated'}


def test_baseline_remains_available_and_no_candidate_is_not_success():
    baseline=candidate('baseline',500,10,.01)
    assert decide([baseline,candidate('joint',600,20,.02)],1000)['recommended_id']=='baseline'
    assert decide([candidate('bad',500,1,.01,False)],1000)['recommended_id'] is None


def test_actual_resource_cap_triggers_correction_inside_count_tolerance():
    assert count_correction_target(2049,2048,2048)==.98*2048
    assert count_correction_target(2008,2048,2048) is None
    assert count_correction_target(800,1000,2048)==1000
