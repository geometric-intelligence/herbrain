import subprocess
from pathlib import Path

import pandas as pd

import herbrain.lddmm as lddmm
import preprocessing
import strings

project_dir = Path('/user/nguigui/home/Documents/UCSB/menstrual')
data_dir = project_dir / 'a_meshed'
output_dir = project_dir / 'meshes_nico'
tmp_dir = Path('/tmp')
ref_labels = preprocessing.get_ref_labels()


def get_day(side, struct, day):
    name = f'{side}_{struct}_t{day:02}.vtk'
    return output_dir / structure / 'raw' / name


side_ = 'left'
structure = 'ERC'
preprocessing.main(1, 60, 1, side_, data_dir, output_dir)

# registration of day 1 - main geodesic
atlas = get_day(side_, structure, day=1)
target = get_day(side_, structure, day=31)

registration_dir = output_dir / structure / 'initial_registration'
registration_args = dict(
    kernel_width=4., regularisation=1., max_iter=2000,
    freeze_control_points=False, attachment_kernel_width=1., metric='varifold',
    tol=1e-10, filter_cp=True, threshold=0.75,
    use_rk4_for_shoot=True, use_rk2_for_shoot=False)

lddmm.registration(atlas, target, registration_dir, **registration_args)

# registration, transport, and shoot each day
registration_args.update(use_rk4_for_shoot=False, use_rk2_for_shoot=True)
registration_seq_dir = output_dir / structure / 'seq_registration'
registration_seq_dir.mkdir(exist_ok=True)

source = get_day(side_, structure, day=31)
n_rungs = 11
transport_args = {'kernel_type': 'torch', 'kernel_width': 4, 'n_rungs': n_rungs}

shoot_args = {
    'source': source, 'use_rk2_for_flow': True, 'kernel_width': 4.,
    'number_of_time_steps': n_rungs + 1, 'write_params': False}

momenta = registration_dir / strings.momenta_str
control_points = registration_dir / strings.cp_str

for d in range(31, 61):
    target = get_day(side_, structure, day=d)
    output_name = tmp_dir / structure / f'reg_{d}'
    lddmm.registration(
        source, target, output_name, **registration_args)

    to_move = [strings.cp_str, strings.momenta_str, strings.residual_str]
    move_to = [f'cp_{d}.txt', f'momenta_{d}.txt', f'registration_error_{d}.txt']

    for s, t in zip(to_move, move_to):
        subprocess.call(['mv', output_name / s, registration_seq_dir / t])

    output_name = (tmp_dir / f'{d}_transport_S')
    subprocess.call(['mkdir', output_name])

    target_dir = output_dir / structure / 'transport'
    momenta_to_transport = (registration_seq_dir / f'momenta_{d}.txt').as_posix()
    control_points_to_transport = (registration_seq_dir / f'cp_{d}.txt').as_posix()

    subprocess.call(['mkdir', target_dir])

    lddmm.transport(
        control_points, momenta, control_points_to_transport, momenta_to_transport,
        output_name, **transport_args)

    to_move = ['final_cp.txt', 'transported_momenta.txt']
    transported_cp = target_dir / f'transported_cp_{d}.txt'
    transported_mom = target_dir / f'transported_momenta_{d}.txt'
    move_to = [transported_cp, transported_mom]

    for src, dest in zip(to_move, move_to):
        subprocess.call(['mv', output_name / src, dest])

    # Shoot transported momenta from atlas
    lddmm.shoot(
        control_points=transported_cp.as_posix(),
        momenta=transported_mom.as_posix(),
        output_dir=output_name, **shoot_args)

    shoot_name = output_name / strings.shoot_str.format(n_rungs)
    subprocess.call(['mv', shoot_name, target_dir / f'transported_shoot_{d}.vtk'])
    subprocess.call(['rm', '-r', output_name])


hormones = pd.read_csv(project_dir / 'hormones.csv', index_col=0)
hormones.loc[hormones.index <= 30, 'cycle'] = 'free'
hormones.loc[hormones.index > 30, 'cycle'] = 'birthcontrol'
# fig, ax = plt.subplots(figsize=(8, 6))
# hormones.groupby('cycle').plot(x='CycleDay', y='Prog', ax=ax)
# plt.show()

times = hormones.CycleDay
times = (times - times.min()) / (times.max() - times.min())
