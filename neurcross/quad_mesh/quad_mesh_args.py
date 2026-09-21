import argparse


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return parsed


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be a non-negative number')
    return parsed

def add_args(parser):
    parser.add_argument('--logdir', type=str, default='outputs/neurcross', help='log directory')
    parser.add_argument('--model_name', type=str, default='model', help='trained model name')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--data_path', type=str, required=True, help='path to input mesh')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='numbers of epochs')
    parser.add_argument('--n_points', type=int, default=15000, help='number of points in each point cloud')
    parser.add_argument('--grid_res', type=int, default=256, help='uniform grid resolution')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='gaussian',
                        help='how to sample points off the manifold - grid | gaussian | combined')

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='always be 1')
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--disable_batched_hessian', action='store_true',
                        help='fall back to three eager Hessian VJPs')
    parser.add_argument('--fixed_steps', '--fixed-steps', type=_positive_int, default=None,
                        help='run exactly N optimizer updates and bypass automatic convergence stopping')
    parser.add_argument('--auto_min_steps', '--auto-min-steps', type=_positive_int, default=1500,
                        help='earliest post-update step eligible for automatic fixed-probe loss stopping')
    parser.add_argument('--auto_check_interval', '--auto-check-interval', type=_positive_int, default=250,
                        help='optimizer updates between deterministic convergence probes')
    parser.add_argument('--auto_loss_relative_tolerance',
                        '--auto-loss-relative-tolerance', type=_nonnegative_float, default=0.10,
                        help='maximum three-probe relative loss span: (max-min)/max(min,float64_eps)')
    parser.add_argument('--auto_probe_seed', '--auto-probe-seed', type=_nonnegative_int, default=104729,
                        help='seed for the probe-local NumPy generator (global RNGs are untouched)')

    # Optional research controls.  Their defaults deliberately preserve the
    # deployed random-initialisation/training path.
    parser.add_argument('--angle_init', '--angle-init', choices=('random', 'geometry'), default='random',
                        help='angle decoder initialization: deployed random weights or a geometry target prefit')
    parser.add_argument('--geometry_init_file', '--geometry-init-file', type=str, default=None,
                        help='geometry_init.npz or its companion geometry_init.json')
    parser.add_argument('--geometry_prefit_lr', '--geometry-prefit-lr', type=float, default=1e-3,
                        help='learning rate for the isolated decoder_angle prefit')
    parser.add_argument('--geometry_prefit_max_steps', '--geometry-prefit-max-steps',
                        type=_positive_int, default=500,
                        help='maximum isolated decoder_angle prefit updates')
    parser.add_argument('--geometry_prefit_target_deg', '--geometry-prefit-target-deg',
                        type=_nonnegative_float, default=5.0,
                        help='stop the prefit once mean 4-RoSy error reaches this many degrees')
    parser.add_argument('--research_checkpoint_steps', '--research-checkpoint-steps',
                        nargs='+', type=_positive_int, default=[], metavar='STEP',
                        help='post-update steps whose fixed-probe field/report are atomically preserved')
    parser.add_argument('--research_trace', '--research-trace', action='store_true',
                        help='write a no-replace structured JSONL training trace')
    parser.add_argument('--research_trace_path', '--research-trace-path', type=str, default=None,
                        help='explicit JSONL trace path (default: <shape logdir>/research_trace.jsonl)')
    parser.add_argument('--research_publication_logdir', '--research-publication-logdir',
                        type=str, default=None,
                        help=('final absolute shape logdir used only for lexical research '
                              'artifact references; it is never created or written'))

    # Network architecture and loss
    parser.add_argument('--init_type', type=str, default='siren',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=4, help='number of decoder hidden layers')
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu | softplus')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 0.1],
                        help='radius and scaling')
    parser.add_argument('--udf', action='store_true')
    parser.add_argument('--output_any', action='store_true')

    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_morse_w_theta')
    parser.add_argument('--decay_params', nargs='+', type=float, default=[3, 0.2, 3, 0.4, 0.001, 0],
                        help='epoch number to evaluate')
    parser.add_argument('--morse_type', type=str, default='l1', help='divergence term norm l1 | l2')
    parser.add_argument('--morse_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[7e3, 6e2, 10, 5e1, 30, 3],
                        help='loss terms weights sdf | inter | normal | eikonal | div | morse')
    parser.add_argument('--morse_near', action='store_true')
    parser.add_argument('--weight_for_morse', action='store_true',
                        help='if true, Weighting A according to the distance of the sampling point')
    parser.add_argument('--use_morse_nonmnfld_grad', type=bool, default=True, help='if True, use morse loss on nonmnfld')
    parser.add_argument('--relax_morse', type=float, default=0.5, help='the max value of relax Morse')
    parser.add_argument('--use_vertices', type=bool, default=False, help='if False, sample points to overfitting')
    parser.add_argument('--featureLine_threshold', type=float, default=1.0)


    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args
