import json
import re
from pathlib import Path

import strings


def generate_visualization(registration_dir, regression_dir, data_set, times):
    filenames = dict(
        registration={
            'registrationName': 'registration',
            'FileNames': [
                (registration_dir / strings.registration_str.format(i)).as_posix()
                for i in range(11)]
        },
        initial_cp_registration={
            'registrationName': 'initial_control_points_registration',
            'FileNames': [(registration_dir / 'initial_control_points.vtk').as_posix()]
        },
        target_shape={
            'registrationName': 'target_shape',
            'FileNames': [(registration_dir / 'target_shape.vtk').as_posix()]},
        source_shape={
            'registrationName': 'source',
            'FileNames': [(registration_dir / strings.template_str).as_posix()]},
        initial_cp_regression={
            'registrationName': 'initial_control_points_regression',
            'FileNames': [(regression_dir / 'initial_control_points.vtk').as_posix()]},
        regression_shapes={
            'registrationName': 'regression',
            'FileNames': [
                (regression_dir / strings.regression_str.format(i, t)).as_posix() for
                i, t in enumerate(sorted(times))]},
        raw_data_shapes={
            'registrationName': 'raw_data',
            'FileNames': [k['shape'].as_posix() for k in data_set]}
    )

    with open(registration_dir.parent / f'visualisation_names.json', 'w') as fp:
        json.dump(filenames, fp)

    template_path = Path(__file__).parent / 'template.py'
    with open(template_path, 'r') as file:
        content = file.read()

    pattern = re.compile('result_path_place_holder')
    updated_content = pattern.sub(registration_dir.parent.as_posix(), content)
    with open(registration_dir.parent / 'viz.py', 'w') as file:
        file.write(updated_content)
