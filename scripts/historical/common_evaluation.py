"""Backend-neutral final-surface measurements, with immutable reference samples."""
import sys
import numpy as np
from audit_evidence import OLD
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from weak_layout_pipeline.pipeline.evaluate import topology_metrics,quality_metrics
from weak_layout_pipeline.pipeline.sampling import area_samples,surface_distances
from evaluation_review import review_mesh
from defect_predictor_v1 import measure_edges

def evaluate(source,quad,graph,h):
    topology=topology_metrics(quad);quality=quality_metrics(quad,source)
    review=review_mesh(quad,source,topology,quality,uv=None)
    records=measure_edges(source,quad,graph,h)
    sd=surface_distances(area_samples(source,4096,190919)[0],quad)/h
    qd=surface_distances(area_samples(quad,4096,190920)[0],source)/h
    return dict(quads=quad.face_count,basic_output_pass=review['basic_output_checks_pass'],
        all_main_deficit=sum(r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction']) for r in records if r['layer']=='main'),
        symmetric_rms_h=float(np.sqrt((np.mean(sd*sd)+np.mean(qd*qd))/2)),source_p95_h=float(np.percentile(sd,95)),
        output_p95_h=float(np.percentile(qd,95)),topology=topology,quality=quality,review=review,edge_defects=records,
        intermediate_uv='Not applicable to backend-neutral final-output evaluation.')
