import pathlib
import subprocess
import lddmm

import herbrain.strings as strings


def main(
        source, target, atlas, name, output_dir, registration_args,
        transport_args, shoot_args, main_reg_dir=None):
    # estimation of time deformation
    time_reg_dir = output_dir / name / f'time_reg_{name}'
    time_reg_dir.mkdir(parents=True, exist_ok=True)

    lddmm.registration(
        source, target, time_reg_dir, **registration_args)

    if main_reg_dir is None:
        # estimation of subject-to-patient deformation
        main_reg_dir = output_dir / name / f'main_reg_{name}'
        main_reg_dir.mkdir(parents=True, exist_ok=True)

        lddmm.registration(
            atlas, source, main_reg_dir, **registration_args)

    # parallel transport of time deformation along subject-to-patient deformation
    momenta_to_transport = (time_reg_dir / strings.momenta_str).as_posix()
    control_points_to_transport = (time_reg_dir / strings.cp_str).as_posix()
    control_points = (main_reg_dir / strings.cp_str).as_posix()
    momenta = (main_reg_dir / strings.cp_str).as_posix()

    transport_dir = output_dir / name / 'transport'
    transport_dir.mkdir(parents=True, exist_ok=True)
    lddmm.transport(
        control_points, momenta, control_points_to_transport, momenta_to_transport,
        transport_dir, **transport_args)

    # Shoot transported momenta from atlas
    transported_cp = transport_dir / 'final_cp.txt'
    transported_mom = transport_dir / 'transported_momenta.txt'
    lddmm.shoot(
        control_points=transported_cp.as_posix(),
        momenta=transported_mom.as_posix(),
        output_dir=transport_dir, **shoot_args)

    shoot_name = transport_dir / strings.shoot_str.format(transport_args['n_rungs'])
    subprocess.call(['cp', shoot_name, transport_dir / f'transported_shoot_{name}.vtk'])
    return transported_mom
