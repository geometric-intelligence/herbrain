# state file generated using paraview version 5.9.1
import json
with open(
        '/user/nguigui/home/Documents/UCSB/meshes_nico/PostHipp/LH'
        '/visualisation_names.json', 'r') as fp:
    filenames = json.load(fp)

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1507, 734]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [277.92604064941406, 179.04769134521484,
                                5.427828788757324]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [276.3549570947753, 183.19231182409987, -70.45550712051084]
renderView1.CameraFocalPoint = [277.9260406494143, 179.04769134521467,
                                5.427828788757304]
renderView1.CameraViewUp = [0.21736492870542645, 0.9748724815155914, 0.0487455900827761]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 19.67352811328448

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1507, 734)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Legacy VTK Reader'
registration = LegacyVTKReader(**filenames['registration'])

# create a new 'Legacy VTK Reader'
regression = LegacyVTKReader(**filenames['regression_shapes'])

# create a new 'Legacy VTK Reader'
target_shape = LegacyVTKReader(**filenames['target_shape'])

# create a new 'Legacy VTK Reader'
raw_data = LegacyVTKReader(**filenames['raw_data_shapes'])

# create a new 'Legacy VTK Reader'
source = LegacyVTKReader(**filenames['source_shape'])

# create a new 'Legacy VTK Reader'
initial_control_points_regression = LegacyVTKReader(**filenames['initial_cp_regression'])

# create a new 'Legacy VTK Reader'
initial_control_points_registration = LegacyVTKReader(**filenames['initial_cp_registration'])

# create a new 'Glyph'
vel = Glyph(registrationName='vel', Input=initial_control_points_regression,
            GlyphType='Arrow')
vel.OrientationArray = ['POINTS', 'Velocity']
vel.ScaleArray = ['POINTS', 'Velocity']
vel.ScaleFactor = 1.6
vel.GlyphTransform = 'Transform2'

# create a new 'Glyph'
vel_1 = Glyph(registrationName='vel', Input=initial_control_points_registration,
              GlyphType='Arrow')
vel_1.OrientationArray = ['POINTS', 'Velocity']
vel_1.ScaleArray = ['POINTS', 'Velocity']
vel_1.ScaleFactor = 1.6
vel_1.GlyphTransform = 'Transform2'

# create a new 'Glyph'
momenta = Glyph(registrationName='momenta', Input=initial_control_points_regression,
                GlyphType='Arrow')
momenta.OrientationArray = ['POINTS', 'Momentum']
momenta.ScaleArray = ['POINTS', 'Momentum']
momenta.ScaleFactor = 1.6
momenta.GlyphTransform = 'Transform2'

# create a new 'Glyph'
momenta_1 = Glyph(registrationName='momenta', Input=initial_control_points_registration,
                  GlyphType='Arrow')
momenta_1.OrientationArray = ['POINTS', 'Momentum']
momenta_1.ScaleArray = ['POINTS', 'Momentum']
momenta_1.ScaleFactor = 1.6
momenta_1.GlyphTransform = 'Transform2'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from initial_control_points_registration
initial_control_points_registrationDisplay = Show(initial_control_points_registration,
                                                  renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'Momentum'
momentumLUT = GetColorTransferFunction('Momentum')
momentumLUT.RGBPoints = [0.006285316954890572, 0.231373, 0.298039, 0.752941,
                         4.673110955213255, 0.865003, 0.865003, 0.865003,
                         9.339936593471618, 0.705882, 0.0156863, 0.14902]
momentumLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
initial_control_points_registrationDisplay.Representation = 'Point Gaussian'
initial_control_points_registrationDisplay.ColorArrayName = ['POINTS', 'Momentum']
initial_control_points_registrationDisplay.LookupTable = momentumLUT
initial_control_points_registrationDisplay.SelectTCoordArray = 'None'
initial_control_points_registrationDisplay.SelectNormalArray = 'None'
initial_control_points_registrationDisplay.SelectTangentArray = 'None'
initial_control_points_registrationDisplay.OSPRayScaleArray = 'Momentum'
initial_control_points_registrationDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
initial_control_points_registrationDisplay.SelectOrientationVectors = 'Momentum'
initial_control_points_registrationDisplay.ScaleFactor = 3.558497
initial_control_points_registrationDisplay.SelectScaleArray = 'Momentum'
initial_control_points_registrationDisplay.GlyphType = 'Arrow'
initial_control_points_registrationDisplay.GlyphTableIndexArray = 'Momentum'
initial_control_points_registrationDisplay.GaussianRadius = 0.5
initial_control_points_registrationDisplay.SetScaleArray = ['POINTS', 'Momentum']
initial_control_points_registrationDisplay.ScaleTransferFunction = 'PiecewiseFunction'
initial_control_points_registrationDisplay.OpacityArray = ['POINTS', 'Momentum']
initial_control_points_registrationDisplay.OpacityTransferFunction = 'PiecewiseFunction'
initial_control_points_registrationDisplay.DataAxesGrid = 'GridAxesRepresentation'
initial_control_points_registrationDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
initial_control_points_registrationDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5,
                                                                         0.0,
                                                                         60.00000000000003,
                                                                         1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
initial_control_points_registrationDisplay.ScaleTransferFunction.Points = [
    -1.0873324533164452, 0.0, 0.5, 0.0, 3.108503832605042, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
initial_control_points_registrationDisplay.OpacityTransferFunction.Points = [
    -1.0873324533164452, 0.0, 0.5, 0.0, 3.108503832605042, 1.0, 0.5, 0.0]

# show data from vel_1
vel_1Display = Show(vel_1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
vel_1Display.Representation = 'Surface'
vel_1Display.ColorArrayName = ['POINTS', 'Momentum']
vel_1Display.LookupTable = momentumLUT
vel_1Display.SelectTCoordArray = 'None'
vel_1Display.SelectNormalArray = 'None'
vel_1Display.SelectTangentArray = 'None'
vel_1Display.OSPRayScaleArray = 'Momentum'
vel_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
vel_1Display.SelectOrientationVectors = 'Momentum'
vel_1Display.ScaleFactor = 3.6595428466796878
vel_1Display.SelectScaleArray = 'Momentum'
vel_1Display.GlyphType = 'Arrow'
vel_1Display.GlyphTableIndexArray = 'Momentum'
vel_1Display.GaussianRadius = 0.18297714233398438
vel_1Display.SetScaleArray = ['POINTS', 'Momentum']
vel_1Display.ScaleTransferFunction = 'PiecewiseFunction'
vel_1Display.OpacityArray = ['POINTS', 'Momentum']
vel_1Display.OpacityTransferFunction = 'PiecewiseFunction'
vel_1Display.DataAxesGrid = 'GridAxesRepresentation'
vel_1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
vel_1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003, 1.0,
                                           0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
vel_1Display.ScaleTransferFunction.Points = [-1.0873324533164452, 0.0, 0.5, 0.0,
                                             3.108503832605042, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
vel_1Display.OpacityTransferFunction.Points = [-1.0873324533164452, 0.0, 0.5, 0.0,
                                               3.108503832605042, 1.0, 0.5, 0.0]

# show data from momenta_1
momenta_1Display = Show(momenta_1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
momenta_1Display.Representation = 'Surface'
momenta_1Display.ColorArrayName = ['POINTS', 'Momentum']
momenta_1Display.LookupTable = momentumLUT
momenta_1Display.SelectTCoordArray = 'None'
momenta_1Display.SelectNormalArray = 'None'
momenta_1Display.SelectTangentArray = 'None'
momenta_1Display.OSPRayScaleArray = 'Momentum'
momenta_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
momenta_1Display.SelectOrientationVectors = 'Momentum'
momenta_1Display.ScaleFactor = 3.5830474853515626
momenta_1Display.SelectScaleArray = 'Momentum'
momenta_1Display.GlyphType = 'Arrow'
momenta_1Display.GlyphTableIndexArray = 'Momentum'
momenta_1Display.GaussianRadius = 0.17915237426757813
momenta_1Display.SetScaleArray = ['POINTS', 'Momentum']
momenta_1Display.ScaleTransferFunction = 'PiecewiseFunction'
momenta_1Display.OpacityArray = ['POINTS', 'Momentum']
momenta_1Display.OpacityTransferFunction = 'PiecewiseFunction'
momenta_1Display.DataAxesGrid = 'GridAxesRepresentation'
momenta_1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
momenta_1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003,
                                               1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
momenta_1Display.ScaleTransferFunction.Points = [-1.2058828128559589, 0.0, 0.5, 0.0,
                                                 1.8833669421108907, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
momenta_1Display.OpacityTransferFunction.Points = [-1.2058828128559589, 0.0, 0.5, 0.0,
                                                   1.8833669421108907, 1.0, 0.5, 0.0]

# show data from registration
registrationDisplay = Show(registration, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
registrationDisplay.Representation = 'Wireframe'
registrationDisplay.ColorArrayName = ['POINTS', '']
registrationDisplay.SelectTCoordArray = 'None'
registrationDisplay.SelectNormalArray = 'None'
registrationDisplay.SelectTangentArray = 'None'
registrationDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
registrationDisplay.SelectOrientationVectors = 'None'
registrationDisplay.ScaleFactor = 3.520928955078125
registrationDisplay.SelectScaleArray = 'None'
registrationDisplay.GlyphType = 'Arrow'
registrationDisplay.GlyphTableIndexArray = 'None'
registrationDisplay.GaussianRadius = 0.17604644775390627
registrationDisplay.SetScaleArray = ['POINTS', '']
registrationDisplay.ScaleTransferFunction = 'PiecewiseFunction'
registrationDisplay.OpacityArray = ['POINTS', '']
registrationDisplay.OpacityTransferFunction = 'PiecewiseFunction'
registrationDisplay.DataAxesGrid = 'GridAxesRepresentation'
registrationDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
registrationDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003,
                                                  1.0, 0.5, 0.0]

# show data from source
sourceDisplay = Show(source, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sourceDisplay.Representation = 'Surface'
sourceDisplay.AmbientColor = [0.0, 0.0, 0.4980392156862745]
sourceDisplay.ColorArrayName = ['POINTS', '']
sourceDisplay.DiffuseColor = [0.0, 0.0, 0.4980392156862745]
sourceDisplay.SelectTCoordArray = 'None'
sourceDisplay.SelectNormalArray = 'None'
sourceDisplay.SelectTangentArray = 'None'
sourceDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
sourceDisplay.SelectOrientationVectors = 'None'
sourceDisplay.ScaleFactor = 3.520928955078125
sourceDisplay.SelectScaleArray = 'None'
sourceDisplay.GlyphType = 'Arrow'
sourceDisplay.GlyphTableIndexArray = 'None'
sourceDisplay.GaussianRadius = 0.17604644775390627
sourceDisplay.SetScaleArray = ['POINTS', '']
sourceDisplay.ScaleTransferFunction = 'PiecewiseFunction'
sourceDisplay.OpacityArray = ['POINTS', '']
sourceDisplay.OpacityTransferFunction = 'PiecewiseFunction'
sourceDisplay.DataAxesGrid = 'GridAxesRepresentation'
sourceDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
sourceDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003, 1.0,
                                            0.5, 0.0]

# show data from regression
regressionDisplay = Show(regression, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
regressionDisplay.Representation = 'Wireframe'
regressionDisplay.AmbientColor = [0.0, 0.0, 1.0]
regressionDisplay.ColorArrayName = ['POINTS', '']
regressionDisplay.DiffuseColor = [0.0, 0.0, 1.0]
regressionDisplay.SelectTCoordArray = 'None'
regressionDisplay.SelectNormalArray = 'None'
regressionDisplay.SelectTangentArray = 'None'
regressionDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
regressionDisplay.SelectOrientationVectors = 'None'
regressionDisplay.ScaleFactor = 3.6701690673828127
regressionDisplay.SelectScaleArray = 'None'
regressionDisplay.GlyphType = 'Arrow'
regressionDisplay.GlyphTableIndexArray = 'None'
regressionDisplay.GaussianRadius = 0.18350845336914062
regressionDisplay.SetScaleArray = ['POINTS', '']
regressionDisplay.ScaleTransferFunction = 'PiecewiseFunction'
regressionDisplay.OpacityArray = ['POINTS', '']
regressionDisplay.OpacityTransferFunction = 'PiecewiseFunction'
regressionDisplay.DataAxesGrid = 'GridAxesRepresentation'
regressionDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
regressionDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003,
                                                1.0, 0.5, 0.0]

# show data from initial_control_points_regression
initial_control_points_regressionDisplay = Show(initial_control_points_regression,
                                                renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
initial_control_points_regressionDisplay.Representation = 'Surface'
initial_control_points_regressionDisplay.ColorArrayName = ['POINTS', 'Momentum']
initial_control_points_regressionDisplay.LookupTable = momentumLUT
initial_control_points_regressionDisplay.SelectTCoordArray = 'None'
initial_control_points_regressionDisplay.SelectNormalArray = 'None'
initial_control_points_regressionDisplay.SelectTangentArray = 'None'
initial_control_points_regressionDisplay.OSPRayScaleArray = 'Momentum'
initial_control_points_regressionDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
initial_control_points_regressionDisplay.SelectOrientationVectors = 'Momentum'
initial_control_points_regressionDisplay.ScaleFactor = 3.0369147
initial_control_points_regressionDisplay.SelectScaleArray = 'Momentum'
initial_control_points_regressionDisplay.GlyphType = 'Arrow'
initial_control_points_regressionDisplay.GlyphTableIndexArray = 'Momentum'
initial_control_points_regressionDisplay.GaussianRadius = 0.15184573499999998
initial_control_points_regressionDisplay.SetScaleArray = ['POINTS', 'Momentum']
initial_control_points_regressionDisplay.ScaleTransferFunction = 'PiecewiseFunction'
initial_control_points_regressionDisplay.OpacityArray = ['POINTS', 'Momentum']
initial_control_points_regressionDisplay.OpacityTransferFunction = 'PiecewiseFunction'
initial_control_points_regressionDisplay.DataAxesGrid = 'GridAxesRepresentation'
initial_control_points_regressionDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
initial_control_points_regressionDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5,
                                                                       0.0,
                                                                       60.00000000000003,
                                                                       1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
initial_control_points_regressionDisplay.ScaleTransferFunction.Points = [
    -1.1737351132147464, 0.0, 0.5, 0.0, 2.970480772097517, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
initial_control_points_regressionDisplay.OpacityTransferFunction.Points = [
    -1.1737351132147464, 0.0, 0.5, 0.0, 2.970480772097517, 1.0, 0.5, 0.0]

# show data from momenta
momentaDisplay = Show(momenta, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
momentaDisplay.Representation = 'Surface'
momentaDisplay.ColorArrayName = ['POINTS', 'Momentum']
momentaDisplay.LookupTable = momentumLUT
momentaDisplay.SelectTCoordArray = 'None'
momentaDisplay.SelectNormalArray = 'None'
momentaDisplay.SelectTangentArray = 'None'
momentaDisplay.OSPRayScaleArray = 'Momentum'
momentaDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
momentaDisplay.SelectOrientationVectors = 'Momentum'
momentaDisplay.ScaleFactor = 3.3323394775390627
momentaDisplay.SelectScaleArray = 'Momentum'
momentaDisplay.GlyphType = 'Arrow'
momentaDisplay.GlyphTableIndexArray = 'Momentum'
momentaDisplay.GaussianRadius = 0.16661697387695312
momentaDisplay.SetScaleArray = ['POINTS', 'Momentum']
momentaDisplay.ScaleTransferFunction = 'PiecewiseFunction'
momentaDisplay.OpacityArray = ['POINTS', 'Momentum']
momentaDisplay.OpacityTransferFunction = 'PiecewiseFunction'
momentaDisplay.DataAxesGrid = 'GridAxesRepresentation'
momentaDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
momentaDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003, 1.0,
                                             0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
momentaDisplay.ScaleTransferFunction.Points = [-1.1737351132147464, 0.0, 0.5, 0.0,
                                               2.970480772097517, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
momentaDisplay.OpacityTransferFunction.Points = [-1.1737351132147464, 0.0, 0.5, 0.0,
                                                 2.970480772097517, 1.0, 0.5, 0.0]

# show data from vel
velDisplay = Show(vel, renderView1, 'GeometryRepresentation')

# get separate color transfer function/color map for 'Momentum'
separate_velDisplay_MomentumLUT = GetColorTransferFunction('Momentum', velDisplay,
                                                           separate=True)
separate_velDisplay_MomentumLUT.RGBPoints = [0.6279259188570192, 0.231373, 0.298039,
                                             0.752941, 2.0787090822926446, 0.865003,
                                             0.865003, 0.865003, 3.52949224572827,
                                             0.705882, 0.0156863, 0.14902]
separate_velDisplay_MomentumLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
velDisplay.Representation = 'Surface'
velDisplay.ColorArrayName = ['POINTS', 'Momentum']
velDisplay.LookupTable = separate_velDisplay_MomentumLUT
velDisplay.SelectTCoordArray = 'None'
velDisplay.SelectNormalArray = 'None'
velDisplay.SelectTangentArray = 'None'
velDisplay.OSPRayScaleArray = 'Momentum'
velDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
velDisplay.SelectOrientationVectors = 'Momentum'
velDisplay.ScaleFactor = 3.3323394775390627
velDisplay.SelectScaleArray = 'Momentum'
velDisplay.GlyphType = 'Arrow'
velDisplay.GlyphTableIndexArray = 'Momentum'
velDisplay.GaussianRadius = 0.16661697387695312
velDisplay.SetScaleArray = ['POINTS', 'Momentum']
velDisplay.ScaleTransferFunction = 'PiecewiseFunction'
velDisplay.OpacityArray = ['POINTS', 'Momentum']
velDisplay.OpacityTransferFunction = 'PiecewiseFunction'
velDisplay.DataAxesGrid = 'GridAxesRepresentation'
velDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
velDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003, 1.0,
                                         0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
velDisplay.ScaleTransferFunction.Points = [-1.1737351132147464, 0.0, 0.5, 0.0,
                                           2.970480772097517, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
velDisplay.OpacityTransferFunction.Points = [-1.1737351132147464, 0.0, 0.5, 0.0,
                                             2.970480772097517, 1.0, 0.5, 0.0]

# set separate color map
velDisplay.UseSeparateColorMap = True

# show data from target_shape
target_shapeDisplay = Show(target_shape, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'RGBA'
rGBALUT = GetColorTransferFunction('RGBA')
rGBALUT.RGBPoints = [338.68421870527123, 0.231373, 0.298039, 0.752941,
                     338.71546140341684, 0.865003, 0.865003, 0.865003,
                     338.7467041015625, 0.705882, 0.0156863, 0.14902]
rGBALUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
target_shapeDisplay.Representation = 'Surface'
target_shapeDisplay.AmbientColor = [0.6666666666666666, 0.0, 0.0]
target_shapeDisplay.ColorArrayName = ['POINTS', '']
target_shapeDisplay.DiffuseColor = [0.6666666666666666, 0.0, 0.0]
target_shapeDisplay.LookupTable = rGBALUT
target_shapeDisplay.SelectTCoordArray = 'None'
target_shapeDisplay.SelectNormalArray = 'None'
target_shapeDisplay.SelectTangentArray = 'None'
target_shapeDisplay.OSPRayScaleArray = 'RGBA'
target_shapeDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
target_shapeDisplay.SelectOrientationVectors = 'None'
target_shapeDisplay.ScaleFactor = 3.711578369140625
target_shapeDisplay.SelectScaleArray = 'None'
target_shapeDisplay.GlyphType = 'Arrow'
target_shapeDisplay.GlyphTableIndexArray = 'RGBA'
target_shapeDisplay.GaussianRadius = 0.18557891845703126
target_shapeDisplay.SetScaleArray = ['POINTS', 'RGBA']
target_shapeDisplay.ScaleTransferFunction = 'PiecewiseFunction'
target_shapeDisplay.OpacityArray = ['POINTS', 'RGBA']
target_shapeDisplay.OpacityTransferFunction = 'PiecewiseFunction'
target_shapeDisplay.DataAxesGrid = 'GridAxesRepresentation'
target_shapeDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
target_shapeDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003,
                                                  1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
target_shapeDisplay.ScaleTransferFunction.Points = [184.0, 0.0, 0.5, 0.0, 184.03125,
                                                    1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
target_shapeDisplay.OpacityTransferFunction.Points = [184.0, 0.0, 0.5, 0.0, 184.03125,
                                                      1.0, 0.5, 0.0]

# show data from raw_data
raw_dataDisplay = Show(raw_data, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
raw_dataDisplay.Representation = 'Surface'
raw_dataDisplay.AmbientColor = [0.6666666666666666, 0.0, 0.0]
raw_dataDisplay.ColorArrayName = ['POINTS', '']
raw_dataDisplay.DiffuseColor = [0.6666666666666666, 0.0, 0.0]
raw_dataDisplay.LookupTable = rGBALUT
raw_dataDisplay.SelectTCoordArray = 'None'
raw_dataDisplay.SelectNormalArray = 'None'
raw_dataDisplay.SelectTangentArray = 'None'
raw_dataDisplay.OSPRayScaleArray = 'RGBA'
raw_dataDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
raw_dataDisplay.SelectOrientationVectors = 'None'
raw_dataDisplay.ScaleFactor = 3.520928955078125
raw_dataDisplay.SelectScaleArray = 'None'
raw_dataDisplay.GlyphType = 'Arrow'
raw_dataDisplay.GlyphTableIndexArray = 'RGBA'
raw_dataDisplay.GaussianRadius = 0.17604644775390627
raw_dataDisplay.SetScaleArray = ['POINTS', 'RGBA']
raw_dataDisplay.ScaleTransferFunction = 'PiecewiseFunction'
raw_dataDisplay.OpacityArray = ['POINTS', 'RGBA']
raw_dataDisplay.OpacityTransferFunction = 'PiecewiseFunction'
raw_dataDisplay.DataAxesGrid = 'GridAxesRepresentation'
raw_dataDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
raw_dataDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.00000000000003,
                                              1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
raw_dataDisplay.ScaleTransferFunction.Points = [184.0, 0.0, 0.5, 0.0, 184.03125, 1.0,
                                                0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
raw_dataDisplay.OpacityTransferFunction.Points = [184.0, 0.0, 0.5, 0.0, 184.03125, 1.0,
                                                  0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for rGBALUT in view renderView1
rGBALUTColorBar = GetScalarBar(rGBALUT, renderView1)
rGBALUTColorBar.WindowLocation = 'UpperLeftCorner'
rGBALUTColorBar.Title = 'RGBA'
rGBALUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
rGBALUTColorBar.Visibility = 0

# get color legend/bar for momentumLUT in view renderView1
momentumLUTColorBar = GetScalarBar(momentumLUT, renderView1)
momentumLUTColorBar.WindowLocation = 'UpperRightCorner'
momentumLUTColorBar.Title = 'Momentum'
momentumLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
momentumLUTColorBar.Visibility = 0

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction('Velocity')
velocityLUT.RGBPoints = [0.012635511405556531, 0.231373, 0.298039, 0.752941,
                         2.777629417969627, 0.865003, 0.865003, 0.865003,
                         5.5426233245336975, 0.705882, 0.0156863, 0.14902]
velocityLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Title = 'Velocity'
velocityLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
velocityLUTColorBar.Visibility = 0

# get color transfer function/color map for 'SignificativeDiffMeanMomentum'
significativeDiffMeanMomentumLUT = GetColorTransferFunction(
    'SignificativeDiffMeanMomentum')
significativeDiffMeanMomentumLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941,
                                              5.878906683738906e-39, 0.865003, 0.865003,
                                              0.865003, 1.1757813367477812e-38,
                                              0.705882, 0.0156863, 0.14902]
significativeDiffMeanMomentumLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for significativeDiffMeanMomentumLUT in view renderView1
significativeDiffMeanMomentumLUTColorBar = GetScalarBar(
    significativeDiffMeanMomentumLUT, renderView1)
significativeDiffMeanMomentumLUTColorBar.Title = 'SignificativeDiffMeanMomentum'
significativeDiffMeanMomentumLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
significativeDiffMeanMomentumLUTColorBar.Visibility = 0

# get color transfer function/color map for 'MeanForce'
meanForceLUT = GetColorTransferFunction('MeanForce')
meanForceLUT.RGBPoints = [0.037401204972628194, 0.231373, 0.298039, 0.752941,
                          3.8383138989924097, 0.865003, 0.865003, 0.865003,
                          7.639226593012191, 0.705882, 0.0156863, 0.14902]
meanForceLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for meanForceLUT in view renderView1
meanForceLUTColorBar = GetScalarBar(meanForceLUT, renderView1)
meanForceLUTColorBar.Title = 'MeanForce'
meanForceLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
meanForceLUTColorBar.Visibility = 0

# get color transfer function/color map for 'external_force'
external_forceLUT = GetColorTransferFunction('external_force')
external_forceLUT.RGBPoints = [0.5481802204128764, 0.231373, 0.298039, 0.752941,
                               1.8519282458079402, 0.865003, 0.865003, 0.865003,
                               3.1556762712030038, 0.705882, 0.0156863, 0.14902]
external_forceLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for external_forceLUT in view renderView1
external_forceLUTColorBar = GetScalarBar(external_forceLUT, renderView1)
external_forceLUTColorBar.Title = 'external_force'
external_forceLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
external_forceLUTColorBar.Visibility = 0

# get color legend/bar for separate_velDisplay_MomentumLUT in view renderView1
separate_velDisplay_MomentumLUTColorBar = GetScalarBar(separate_velDisplay_MomentumLUT,
                                                       renderView1)
separate_velDisplay_MomentumLUTColorBar.Title = 'Momentum'
separate_velDisplay_MomentumLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
separate_velDisplay_MomentumLUTColorBar.Visibility = 0

# hide data in view
Hide(vel_1, renderView1)

# hide data in view
Hide(momenta_1, renderView1)

# hide data in view
Hide(source, renderView1)

# hide data in view
Hide(regression, renderView1)

# hide data in view
Hide(momenta, renderView1)

# hide data in view
Hide(raw_data, renderView1)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction('Velocity')
velocityPWF.Points = [0.012635511405556531, 0.0, 0.5, 0.0, 5.5426233245336975, 1.0, 0.5,
                      0.0]
velocityPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'Momentum'
momentumPWF = GetOpacityTransferFunction('Momentum')
momentumPWF.Points = [0.006285316954890572, 0.0, 0.5, 0.0, 9.339936593471618, 1.0, 0.5,
                      0.0]
momentumPWF.ScalarRangeInitialized = 1

# get separate opacity transfer function/opacity map for 'Momentum'
separate_velDisplay_MomentumPWF = GetOpacityTransferFunction('Momentum', velDisplay,
                                                             separate=True)
separate_velDisplay_MomentumPWF.Points = [0.6279259188570192, 0.0, 0.5, 0.0,
                                          3.52949224572827, 1.0, 0.5, 0.0]
separate_velDisplay_MomentumPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'SignificativeDiffMeanMomentum'
significativeDiffMeanMomentumPWF = GetOpacityTransferFunction(
    'SignificativeDiffMeanMomentum')
significativeDiffMeanMomentumPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38,
                                           1.0, 0.5, 0.0]
significativeDiffMeanMomentumPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'RGBA'
rGBAPWF = GetOpacityTransferFunction('RGBA')
rGBAPWF.Points = [338.68421870527123, 0.0, 0.5, 0.0, 338.7467041015625, 1.0, 0.5, 0.0]
rGBAPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'external_force'
external_forcePWF = GetOpacityTransferFunction('external_force')
external_forcePWF.Points = [0.5481802204128764, 0.0, 0.5, 0.0, 3.1556762712030038, 1.0,
                            0.5, 0.0]
external_forcePWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'MeanForce'
meanForcePWF = GetOpacityTransferFunction('MeanForce')
meanForcePWF.Points = [0.037401204972628194, 0.0, 0.5, 0.0, 7.639226593012191, 1.0, 0.5,
                       0.0]
meanForcePWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(target_shape)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
