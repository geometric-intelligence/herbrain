import herbrain.lddmm as lddmm
from pregnancy.visualization.paraview import generate_visualization
from strings import cp_str


def main(
        data_set, times, output_dir, config_id="0", structure="PostHipp",
        registration_args=None, spline_args=None):
    if registration_args is None:
        registration_args = {}
    if spline_args is None:
        spline_args = {}

    times = (times - times.min()) / (times.max() - times.min())
    target_weights = [1 / len(data_set)] * len(data_set)

    # registration to optimize control points
    source = data_set[0]['shape']
    target = data_set[-1]['shape']
    registration_dir = output_dir / structure / config_id / 'initial_registration'
    lddmm.registration(source, target, registration_dir, **registration_args)

    # geodesic or spline regression (depending on freeze_external_forces)
    all_spline_args = registration_args.copy()
    all_spline_args.update(spline_args)
    all_spline_args['initial_control_points'] = registration_dir / cp_str
    regression_dir = output_dir / structure / config_id / 'regression'
    lddmm.spline_regression(
        source=data_set[0]['shape'], target=data_set,
        output_dir=regression_dir, subject_id=[''],
        times=times.tolist(), t0=min(times),
        target_weights=target_weights, **all_spline_args)
    generate_visualization(registration_dir, regression_dir, data_set, times)
