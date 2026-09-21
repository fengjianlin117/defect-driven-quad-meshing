import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary

from models import Network_predict_angle
from models import MorseLoss_quad_mesh as MorseLoss
import utils.utils as utils
import quad_mesh_args
from convergence_stop import (AutoStopConfig, ConvergenceStopper, build_fixed_probe,
                              final_morse_weight, ProbeStateIntegrityError,
                              run_fixed_field_only,
                              run_fixed_probe, validate_training_contract,
                              validate_final_artifact_destination,
                              write_final_artifacts)
from geometry_initialization import (
    StructuredResearchTrace, attach_terminal_research_provenance,
    build_research_field_provenance, build_research_publication_paths,
    load_geometry_initialization, prefit_geometry_angle_decoder,
    resolve_research_trace_path,
    sha256_file, validate_research_contract,
    validate_research_output_destinations, write_research_checkpoint,
)

import quad_mesh_dataset as dataset

# get training parameters
args = quad_mesh_args.get_args()
end_to_end_started_monotonic = time.monotonic()
auto_stop_config = AutoStopConfig(
    min_steps=args.auto_min_steps,
    check_interval=args.auto_check_interval,
    loss_relative_tolerance=args.auto_loss_relative_tolerance,
)
training_mode, n_iterations, requested_steps = validate_training_contract(args, auto_stop_config)
research_checkpoint_steps = validate_research_contract(
    args, training_mode, n_iterations, requested_steps)
file_name = os.path.splitext(args.data_path.split('/')[-1])[0]
logdir = os.path.join(args.logdir, file_name)
publication_paths = build_research_publication_paths(
    logdir, args.research_publication_logdir)
geometry_initialization = None
if args.angle_init == 'geometry':
    geometry_initialization = load_geometry_initialization(
        args.geometry_init_file, args.data_path)

research_trace_path = resolve_research_trace_path(args, logdir)
publication_paths.validate_trace_path(research_trace_path)
research_input_sha256 = (
    geometry_initialization.mesh_sha256
    if geometry_initialization is not None
    else sha256_file(args.data_path)
    if (research_checkpoint_steps or research_trace_path is not None
        or publication_paths.enabled)
    else None
)
validate_final_artifact_destination(logdir)
validate_research_output_destinations(
    logdir, research_checkpoint_steps, research_trace_path)

os.makedirs(logdir, exist_ok=True)

# set up logging
log_file = utils.setup_logdir_only_log(logdir, args)

research_trace = StructuredResearchTrace(research_trace_path, metadata={
    'data_path': os.path.abspath(args.data_path),
    'mesh_sha256': research_input_sha256,
    'checkpoint_steps': research_checkpoint_steps,
    'elapsed_seconds_origin': 'immediately after argument parsing',
    'optimizer_update_trace_interval': 10,
})
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# get data loaders
utils.same_seed(args.seed)
train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
# get model
net = Network_predict_angle(in_dim=3, angle_in_dim=12, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl,
                            decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type,
                            sphere_init_params=args.sphere_init_params, udf=args.udf)

net.to(device)
if args.load_path is not None:
    net.load_state_dict(torch.load(args.load_path))
    print('Loaded model from %s' % args.load_path)
summary(net.decoder, (1, 1024, 3))

n_parameters = utils.count_parameters(net)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

geometry_prefit_report = None
if geometry_initialization is not None:
    geometry_prefit_report = prefit_geometry_angle_decoder(
        net, geometry_initialization, train_set, device,
        learning_rate=args.geometry_prefit_lr,
        max_steps=args.geometry_prefit_max_steps,
        target_degrees=args.geometry_prefit_target_deg,
    )
    utils.log_string(
        'Geometry angle prefit: {} updates, final mean cross error {:.6f} deg'.format(
            geometry_prefit_report['completed_updates'],
            geometry_prefit_report['final']['mean_cross_error_degrees']),
        log_file)
    research_trace.record('geometry_prefit', report=geometry_prefit_report)

# Setup Adam optimizers
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
print('n_iterations: ', n_iterations)
print('training mode: ', training_mode)
if training_mode == 'fixed':
    print('fixed optimizer updates: ', requested_steps)

net.to(device)

num_batches = len(train_dataloader)
refine_flag = True
min_cd = np.inf
SAVE_BEST = False

##################################################################################
# get the vertices neighbors of the mesh
vertex_neighbors = utils.get_sample_vers_neighbors_for_face_center_points_or_vertices(args.data_path)
vertex_neighbors_list = utils.calculate_same_neighbors_verts(vertex_neighbors)
###################################################################################
axis_angle_R_mat_list = utils.get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, args.data_path)

criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                      div_type=args.morse_type,
                      vertex_neighbors_list=vertex_neighbors_list,
                      vertex_neighbors=vertex_neighbors, axis_angle_R_mat_list=axis_angle_R_mat_list, device=device,
                      use_batched_hessian=not args.disable_batched_hessian
                      )

final_score_weights = [float(value) for value in args.loss_weights]
final_score_weights[5] = final_morse_weight(
    n_iterations - 1, n_iterations, args.decay_params, args.morse_decay,
    final_score_weights[5])

# Fixed-step mode deliberately gives the convergence gate no observations.
if training_mode == 'auto':
    try:
        face_areas = np.asarray(train_set.mesh.area_faces, dtype=np.float64)
        convergence_stopper = ConvergenceStopper(auto_stop_config, face_areas, auto_enabled=True)
    except Exception as setup_error:
        convergence_stopper = ConvergenceStopper(
            auto_stop_config, face_areas=None, auto_enabled=True)
        convergence_stopper.record_direction_diagnostic_error(
            0, 'face_area_setup', setup_error)
else:
    convergence_stopper = ConvergenceStopper(
        auto_stop_config, face_areas=None, auto_enabled=False)

fixed_probe = None
needs_fixed_probe = (
    (training_mode == 'auto' and convergence_stopper.auto_enabled)
    or bool(research_checkpoint_steps)
)
if needs_fixed_probe:
    try:
        fixed_probe = build_fixed_probe(train_set, args.n_points, args.auto_probe_seed)
    except Exception as setup_error:
        if research_checkpoint_steps:
            raise RuntimeError('mandatory research fixed-probe setup failed') from setup_error
        convergence_stopper.disable_after_probe_error(0, 'fixed_probe_setup', setup_error)

completed_steps = 0
stop_reason = None
terminal_observation = None
published_research_checkpoints = {}

# For each epoch. Validation above intentionally restricts this to one epoch and
# batch size one so update counts and the deployed Morse index stay unambiguous.
for epoch in range(args.num_epochs):
    for batch_idx, data in enumerate(train_dataloader):
        if batch_idx != 0 and (batch_idx % 500 == 0 or batch_idx == len(train_dataloader) - 1):
            SAVE_BEST = True

        optimizer.zero_grad(set_to_none=True)
        net.train()

        mnfld_points, mnfld_n_gt, nonmnfld_points, near_points, local_coord_u, local_coord_v = data[
            'points'].to(device), data['mnfld_n'].to(device), data['nonmnfld_points'].to(device), data[
            'near_points'].to(device), data['local_coordinates_u'].to(device), data['local_coordinates_v'].to(device)

        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()

        features = torch.cat((mnfld_points, mnfld_n_gt, local_coord_u, local_coord_v), dim=-1)

        output_pred, mnfld_pts_theta_output_pred = net(nonmnfld_points, mnfld_points,
                                                       near_points=near_points if args.morse_near else None,
                                                       angle_features=features)

        loss_dict = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                              near_points=near_points if args.morse_near else None, batch_idx=batch_idx,
                              logdir=logdir, filename=file_name, save_best=SAVE_BEST,
                              mnfld_pts_theta_output_pred=mnfld_pts_theta_output_pred,
                              local_coord_u=local_coord_u, local_coord_v=local_coord_v)

        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        loss_dict["loss"].backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

        # Capture the weights which produced this loss before the schedule moves
        # to the next-step state.
        weights_used = [float(value.detach().cpu().item()) if torch.is_tensor(value)
                        else float(value) for value in criterion.weights]
        SAVE_BEST = False
        optimizer.step()
        completed_steps += 1

        # The original zero-based schedule mutation remains immediately after
        # optimizer.step. Every convergence probe is strictly after this call.
        criterion.update_morse_weight(epoch * args.n_samples + batch_idx, n_iterations,
                                      args.decay_params)  # assumes batch size of 1

        # Output training stats using the weights which actually formed loss_dict.
        if batch_idx % 10 == 0:
            utils.log_string("Weights: {}, lr={:.3e}".format(weights_used, lr), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + L_thetaNeighbor: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                loss_dict["loss"].item(), weights_used[0] * loss_dict["sdf_term"].item(),
                       weights_used[1] * loss_dict["inter_term"].item(),
                       weights_used[3] * loss_dict["eikonal_term"].item(), weights_used[5] * loss_dict["morse_term"].item(),
                       weights_used[2] * loss_dict["theta_hessian_term"].item(),
                       weights_used[4] * loss_dict['theta_neighbors_term'].item()
            ),
                log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + L_thetaNeighbor: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                loss_dict["eikonal_term"].item(), loss_dict["morse_term"].item(),
                loss_dict['theta_hessian_term'].item(), loss_dict['theta_neighbors_term'].item()),
                log_file)
            utils.log_string('', log_file)

        if research_trace.enabled and batch_idx % 10 == 0:
            research_trace.record(
                'optimizer_update', step=completed_steps, epoch=epoch,
                batch_index=batch_idx, learning_rate=float(lr.item()),
                elapsed_seconds=time.monotonic() - end_to_end_started_monotonic,
                weights=weights_used,
                losses={
                    'loss': float(loss_dict['loss'].item()),
                    'sdf_term': float(loss_dict['sdf_term'].item()),
                    'inter_term': float(loss_dict['inter_term'].item()),
                    'eikonal_term': float(loss_dict['eikonal_term'].item()),
                    'morse_term': float(loss_dict['morse_term'].item()),
                    'theta_hessian_term': float(loss_dict['theta_hessian_term'].item()),
                    'theta_neighbors_term': float(loss_dict['theta_neighbors_term'].item()),
                },
            )

        automatic_probe_due = convergence_stopper.should_probe(completed_steps)
        checkpoint_due = completed_steps in research_checkpoint_steps
        if automatic_probe_due or checkpoint_due:
            # One state-preserving fixed probe serves convergence and research.
            # Only probe/evaluation failures belong to the automatic fail-open
            # contract. Artifact, trace, and log I/O failures remain fatal and
            # retain their real exception domain.
            del output_pred, mnfld_pts_theta_output_pred, features, loss_dict
            probe_error_record = None
            try:
                probe_metrics, cross_field = run_fixed_probe(
                    net, criterion, fixed_probe, args, device, final_score_weights)
                decision = None
                if automatic_probe_due:
                    decision = convergence_stopper.observe(
                        completed_steps, cross_field,
                        probe_metrics['uniform_final_weight_score'])
            except ProbeStateIntegrityError:
                # Continuing would invalidate every later optimizer update.
                raise
            except Exception as probe_error:
                if checkpoint_due:
                    raise RuntimeError(
                        'mandatory research fixed probe failed at step {}'.format(
                            completed_steps)) from probe_error
                probe_error_record = convergence_stopper.disable_after_probe_error(
                    completed_steps, 'automatic_fixed_probe', probe_error)

            if probe_error_record is not None:
                utils.log_string(
                    'Automatic convergence probe disabled after {}: {}'.format(
                        probe_error_record['exception_type'],
                        probe_error_record['message']), log_file)
            else:
                if automatic_probe_due:
                    utils.log_string(
                        'Convergence probe step {}: eligible={}, passes={}'.format(
                            completed_steps, decision['eligible'], decision['passes']),
                        log_file)
                    research_trace.record(
                        'automatic_probe', step=completed_steps,
                        decision=decision, probe=probe_metrics,
                        elapsed_seconds=time.monotonic() - end_to_end_started_monotonic)
                    if decision['passes']:
                        stop_reason = 'loss_converged'
                terminal_observation = {
                    'step': completed_steps,
                    'probe': probe_metrics,
                    'cross_field': cross_field,
                }

                if checkpoint_due:
                    checkpoint_probe = dict(probe_metrics)
                    if cross_field is None:
                        try:
                            field_metrics, cross_field = run_fixed_field_only(
                                net, fixed_probe, args, device)
                        except ProbeStateIntegrityError:
                            raise
                        except Exception as field_error:
                            raise RuntimeError(
                                'mandatory research checkpoint field fallback failed '
                                'at step {}'.format(completed_steps)) from field_error
                        checkpoint_probe['field_only_fallback'] = field_metrics
                        terminal_observation['probe'][
                            'terminal_direction_field_fallback'] = field_metrics
                        terminal_observation['cross_field'] = cross_field
                    checkpoint_report = {
                        'mode': training_mode,
                        'completed_updates': completed_steps,
                        'morse_schedule_horizon': n_iterations,
                        'state_semantics': (
                            'after optimizer update and update_morse_weight; '
                            'eval-mode fixed probe'),
                        'seed': args.seed,
                        'probe_seed': args.auto_probe_seed,
                        'mesh_sha256': research_input_sha256,
                        'face_count': int(cross_field.shape[0]),
                        'grid_res': args.grid_res,
                        'uniform_final_score_weights': final_score_weights,
                        'fixed_probe': checkpoint_probe,
                        'automatic_observation': decision,
                        'automatic_stop_semantics': {
                            'checkpoint_artifact_participates': False,
                            'coincident_fixed_probe_observation_participates': bool(
                                automatic_probe_due),
                        },
                        'end_to_end_elapsed_seconds_at_probe': (
                            time.monotonic() - end_to_end_started_monotonic),
                    }
                    checkpoint_report.update(build_research_field_provenance(
                        seed=args.seed, mesh_sha256=research_input_sha256,
                        grid_res=args.grid_res, angle_init=args.angle_init,
                        field_state='step_{}'.format(completed_steps),
                        geometry_initialization=geometry_prefit_report,
                    ))
                    checkpoint_field_path, checkpoint_report_path, _ = (
                        write_research_checkpoint(
                            logdir, completed_steps, cross_field,
                            checkpoint_report))
                    published_research_checkpoints[completed_steps] = {
                        'field': str(publication_paths.publication_reference(
                            checkpoint_field_path)),
                        'report': str(publication_paths.publication_reference(
                            checkpoint_report_path)),
                    }
                    utils.log_string(
                        'Research checkpoint step {}: {}'.format(
                            completed_steps, checkpoint_field_path), log_file)
                    research_trace.record(
                        'research_checkpoint', step=completed_steps,
                        field_path=publication_paths.publication_reference(
                            checkpoint_field_path),
                        report_path=publication_paths.publication_reference(
                            checkpoint_report_path),
                        elapsed_seconds=(
                            time.monotonic() - end_to_end_started_monotonic))

        if training_mode == 'fixed' and completed_steps == requested_steps:
            break
        if stop_reason == 'loss_converged':
            break
    if training_mode == 'fixed' and completed_steps == requested_steps:
        break
    if stop_reason == 'loss_converged':
        break

if training_mode == 'fixed' and completed_steps != requested_steps:
    raise RuntimeError('fixed-step training completed {} updates, expected {}'.format(
        completed_steps, requested_steps))
if training_mode == 'auto' and stop_reason is None and completed_steps != n_iterations:
    raise RuntimeError('automatic training ended before its hard horizon without loss_converged')

missing_research_checkpoints = sorted(
    set(research_checkpoint_steps).difference(published_research_checkpoints))
if missing_research_checkpoints:
    raise RuntimeError('requested research checkpoints were not published: {}'.format(
        missing_research_checkpoints))

# Fixed and fail-open runs do not necessarily finish on an automatic probe
# point. Release their last training graph before allocating the terminal loss
# probe (especially important for the largest meshes).
output_pred = None
mnfld_pts_theta_output_pred = None
features = None
loss_dict = None
mnfld_points = None
mnfld_n_gt = None
nonmnfld_points = None
near_points = None
local_coord_u = None
local_coord_v = None
data = None

# Reuse the last automatic observation only when it is exactly the terminal
# post-update/post-schedule state. Fixed mode always reaches this path and hence
# cannot enter or be affected by the convergence gate.
if (terminal_observation is None
        or terminal_observation['step'] != completed_steps):
    if fixed_probe is None:
        try:
            fixed_probe = build_fixed_probe(train_set, args.n_points, args.auto_probe_seed)
        except Exception as terminal_setup_error:
            convergence_stopper.disable_after_probe_error(
                completed_steps, 'terminal_fixed_probe_setup', terminal_setup_error)
            raise RuntimeError(
                'training completed, but terminal fixed-probe construction failed; '
                'no convergence terminal field or report was published') from terminal_setup_error
    try:
        probe_metrics, cross_field = run_fixed_probe(
            net, criterion, fixed_probe, args, device, final_score_weights)
    except ProbeStateIntegrityError:
        # A field-only retry is unsafe when restoration itself was incomplete.
        raise
    except Exception as terminal_probe_error:
        record = convergence_stopper.disable_after_probe_error(
            completed_steps, 'terminal_loss_probe', terminal_probe_error)
        utils.log_string(
            'Terminal loss probe failed (mandatory field export follows): {}: {}'.format(
                record['exception_type'], record['message']), log_file)
        probe_metrics = {
            'mode': 'eval',
            'loss_probe_available': False,
            'uniform_final_weight_score': None,
            'loss_probe_error': record,
        }
        cross_field = None
    terminal_observation = {
        'step': completed_steps,
        'probe': probe_metrics,
        'cross_field': cross_field,
    }

# Direction drift never controls the loss gate, but a valid terminal cross
# field remains the required product of this training command. Export failure
# therefore fails finalization and the atomic bundle is never published.
if terminal_observation['cross_field'] is None:
    try:
        field_probe_metrics, cross_field = run_fixed_field_only(
            net, fixed_probe, args, device)
    except ProbeStateIntegrityError:
        raise
    except Exception as terminal_field_error:
        record = convergence_stopper.record_direction_diagnostic_error(
            completed_steps, 'mandatory_terminal_field', terminal_field_error)
        terminal_field_message = (
            'loss stopping decision completed, but the mandatory terminal cross '
            'field could not be exported; no terminal artifact bundle was published: '
            '{}: {}'.format(record['exception_type'], record['message']))
        utils.log_string(terminal_field_message, log_file)
        raise RuntimeError(terminal_field_message) from terminal_field_error
    terminal_observation['cross_field'] = cross_field
    terminal_observation['probe']['terminal_direction_field_fallback'] = field_probe_metrics

termination = ('automatic' if stop_reason == 'loss_converged'
               else 'fixed_steps' if training_mode == 'fixed'
               else 'hard_horizon')
terminal_report = {
    'schema_version': 1,
    'mode': training_mode,
    'termination': termination,
    # This is deliberately null for fixed-step and hard-horizon termination.
    'stop_reason': stop_reason,
    'completed_updates': completed_steps,
    'morse_schedule_horizon': n_iterations,
    'state_semantics': (
        'after {} optimizer updates; after update_morse_weight(current_iteration={}, '
        'n_iterations={}); eval-mode fixed probe'.format(
            completed_steps, completed_steps - 1, n_iterations)),
    'fixed_steps': args.fixed_steps,
    'probe_seed': args.auto_probe_seed,
    'uniform_final_score_weights': final_score_weights,
    'final_probe': terminal_observation['probe'],
    'automatic_loss_convergence': convergence_stopper.report(),
}
research_opt_in = bool(
    research_trace.enabled or research_checkpoint_steps
    or geometry_prefit_report is not None
    or publication_paths.enabled)
if research_opt_in:
    terminal_report['end_to_end_elapsed_seconds'] = (
        time.monotonic() - end_to_end_started_monotonic)
if geometry_prefit_report is not None:
    terminal_report['geometry_initialization'] = geometry_prefit_report
if research_checkpoint_steps:
    terminal_report['research_checkpoints'] = {
        'requested_steps': list(research_checkpoint_steps),
        'published': published_research_checkpoints,
        'automatic_stop_semantics': {
            'checkpoint_artifacts_participate': False,
        },
    }
if research_trace.enabled:
    terminal_report['research_trace'] = {
        'schema_version': 'trimesh2quad.neurcross.research_trace.v1',
        'path': str(publication_paths.publication_reference(
            research_trace.path)),
    }
attach_terminal_research_provenance(
    terminal_report, enabled=research_opt_in, seed=args.seed,
    mesh_sha256=research_input_sha256, grid_res=args.grid_res,
    angle_init=args.angle_init, termination=termination,
    geometry_initialization=geometry_prefit_report,
)
field_path, report_path, _ = write_final_artifacts(
    logdir, terminal_observation['cross_field'], terminal_report,
    lowercase_research_sha256=research_opt_in)
utils.log_string('Training termination: {} after {} updates'.format(
    termination, completed_steps), log_file)
utils.log_string('Terminal post-update eval field: {}'.format(field_path), log_file)
utils.log_string('Terminal convergence report: {}'.format(report_path), log_file)
log_file.close()
research_trace.record(
    'training_termination', termination=termination,
    completed_updates=completed_steps, stop_reason=stop_reason,
    final_probe=terminal_observation['probe'],
    elapsed_seconds=time.monotonic() - end_to_end_started_monotonic)
research_trace.record(
    'final_artifacts',
    field_path=publication_paths.publication_reference(field_path),
    report_path=publication_paths.publication_reference(report_path),
    elapsed_seconds=time.monotonic() - end_to_end_started_monotonic)
research_trace.close()
