import pandas as pd
from pathlib import Path

import herbrain.lddmm as lddmm
from strings import cp_str


def get_data_set(
        data_dir: Path, hemisphere: str, structure: str, covariates: pd.DataFrame,
        tmin: int, tmax: int, day_ref: int):
    data_list = covariates[covariates[structure]]  # remove excluded
    data_list = data_list[
        ((tmin < data_list['gestWeek']) & (data_list['gestWeek'] < tmax))
        | (data_list['day'] == day_ref)]  # select phase + template at day ref
    data_path = data_dir / structure / 'raw'
    dataset = [
        {
            'shape': data_path / f'{hemisphere}_{structure}_t{k:02}.vtk'
         } for k in data_list['day']]
    return dataset, data_list['times']


def main(
        covariates, output_dir, config_id="0", structure="PostHipp", tmin=1, tmax=25,
        day_ref=3, registration_args=None, spline_args=None):
    if registration_args is None:
        registration_args = {}
    if spline_args is None:
        spline_args = {}

    # Filter meshes between to tmin, tmax and ref day
    data_set, times = get_data_set(
        output_dir, 'left', structure, covariates, tmin, tmax, day_ref)
    times = (times - times.min()) / (times.max() - times.min())
    target_weights = [1 / len(data_set)] * len(data_set)

    # registration to optimize control points
    source = data_set[0]['shape']
    target = data_set[-1]['shape']
    registration_dir = output_dir / structure / config_id / 'inital_registration'
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


if __name__ == '__main__':
    from herbrain.pregnancy.configurations import configurations
    project_dir = Path('/user/nguigui/home/Documents/UCSB')
    covariate = pd.read_csv(project_dir / 'covariates.csv')
    out_dir = project_dir / 'meshes_nico'
    # for config in configurations:
    #     main(covariate, out_dir, **config)
    main(covariate, out_dir, **configurations[-1])
