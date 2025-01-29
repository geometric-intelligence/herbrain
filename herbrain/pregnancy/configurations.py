"""Parameters to study different time frames of pregnancy with LDDMM."""

configurations = []

# 1. PostHipp from preconception to 2nd trimester
configurations.append(dict(
    config_id="preTo2nd",
    dataset=dict(
        structure="PostHipp",
        tmin=1,
        tmax=25,
        time_var='gestWeek',
        day_ref=3,
        variable='times'
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1.,
        max_iter=2000,
        freeze_control_points=False,
        attachment_kernel_width=1.,
        metric='varifold',
        tol=1e-16,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 2. preconception to 1st trimester
configurations.append(dict(
    config_id="firstTrim",
    dataset=dict(
        structure="PostHipp",
        tmin=1.5,
        tmax=13,
        time_var='gestWeek',
        day_ref=4,
        variable='times'
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=4.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 3. 1st trimester to 2nd
configurations.append(dict(
    config_id="secondTrim",
    dataset=dict(
        structure="PostHipp",
        tmin=12,
        tmax=25,
        time_var='gestWeek',
        day_ref=8,
        variable='times'
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold', attachment_kernel_width=4.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 4. 3rd trimester : shapes are not well enough semented
# configurations.append(dict(
#     config_id="thirdTrim",
#     structure="PostHipp",
#     tmin=25,
#     tmax=40,
#     time_var='gestWeek',
#     day_ref=12,
#     variable='times',
#     registration_args=dict(
#         kernel_width=4.,
#         regularisation=1,
#         max_iter=2000,
#         freeze_control_points=False,
#         metric='varifold',
#         attachment_kernel_width=4.,
#         tol=1e-10,
#         filter_cp=True,
#         threshold=0.75
#     ),
#     spline_args=dict(
#         initial_step_size=100,
#         regularisation=1.,
#         freeze_external_forces=True,
#         freeze_control_points=True,
#     )
# ))

# 4. PostPartum
configurations.append(dict(
    config_id="PostPartum",
    dataset=dict(
        structure="PostHipp",
        tmin=40,
        tmax=163,
        time_var='gestWeek',
        day_ref=19,
        variable='times'
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=4.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 5. PostHipp all wrt prog
configurations.append(dict(
    config_id="Prog",
    dataset=dict(
        structure="PostHipp",
        tmin=-5,
        tmax=163,
        time_var='gestWeek',
        day_ref=1,
        variable='prog',
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=1.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.25
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 6. PostHipp all wrt estro
configurations.append(dict(
    config_id="Estro",
    dataset=dict(
        structure="PostHipp",
        tmin=-5,
        tmax=163,
        day_ref=1,
        time_var='gestWeek',
        variable='estro'
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=1.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.25
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 7. PostHipp all wrt lh
configurations.append(dict(
    config_id="LH",
    dataset=dict(structure="PostHipp",
        tmin=-5,
        tmax=163,
        time_var='gestWeek',
        day_ref=1,
        variable='lh'
    ),
    registration_args=dict(
        kernel_width=4.,
        regularisation=1,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=1.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.25
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
        filter_cp=False,
    )
))

# 8. PHC full gestation wrt time
configurations.append(dict(
    config_id="gestation",
    dataset=dict(structure="PHC",
        tmin=1,
        tmax=40,
        time_var='gestWeek',
        day_ref=2,
        variable='times'
    ),
    registration_args=dict(
        kernel_width=6.,
        regularisation=1.,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=1.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 9. PHC full gestation wrt prog
configurations.append(dict(
    config_id="gestation_prog",
    dataset=dict(
        structure="PHC",
        tmin=1,
        tmax=40,
        time_var='gestWeek',
        day_ref=2,
        variable='prog')
    ,
    registration_args=dict(
        kernel_width=6.,
        regularisation=1.,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=1.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))

# 8. PHC full gestation wrt estro
configurations.append(dict(
    config_id="gestation_estro",
    dataset=dict(
        structure="PHC",
        tmin=1,
        tmax=40,
        time_var='gestWeek',
        day_ref=2,
        variable='estro'
    ),
    registration_args=dict(
        kernel_width=6.,
        regularisation=1.,
        max_iter=2000,
        freeze_control_points=False,
        metric='varifold',
        attachment_kernel_width=1.,
        tol=1e-10,
        filter_cp=True,
        threshold=0.75
    ),
    spline_args=dict(
        initial_step_size=100,
        regularisation=1.,
        freeze_external_forces=True,
        freeze_control_points=True,
    )
))
