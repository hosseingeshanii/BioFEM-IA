"""pvpython pv_views.py <frames_in_dir> <png_out_dir> <field> <vmin> <vmax> <views>
Renders every in_<k>.vtk (legacy VTK, lengths in cm, point field <field>) for each view in
<views> (comma list of: side, base) to <png_out_dir>/<view>_<k>.png. The reference shape
(in_<first k>.vtk) is drawn as a grey wireframe. Camera fixed from the reference bounds;
long axis z, apex at +z, open base at -z. No scalar bar (the caller draws it).
Run by `biofem post animate`; not meant to be called directly.
"""
import glob
import os
import re
import sys

from paraview.simple import (CreateView, LegacyVTKReader, Show, Hide, Delete, ColorBy, Render,
                             SaveScreenshot, GetColorTransferFunction)

src, out, field, vmin, vmax, views = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6].split(",")
files = sorted(glob.glob(os.path.join(src, "in_*.vtk")), key=lambda f: int(re.findall(r"in_(\d+)\.vtk", f)[0]))
os.makedirs(out, exist_ok=True)

view = CreateView("RenderView")
view.ViewSize = [600, 640]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except Exception:
    pass

ref = LegacyVTKReader(FileNames=[files[0]])
rd = Show(ref, view)
rd.Representation = "Wireframe"
ColorBy(rd, None)
rd.AmbientColor = rd.DiffuseColor = [0.55, 0.55, 0.55]
rd.Opacity = 0.25
x0, x1, y0, y1, z0, z1 = ref.GetDataInformation().GetBounds()
c = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
R = max(x1 - x0, y1 - y0, z1 - z0)
CAM = {
    "side": dict(pos=[c[0] + 2.2 * R, c[1] - 2.2 * R, c[2] + 0.8 * R], up=[0, 0, 1], foc=c),
    "base": dict(pos=[c[0], c[1], z0 - 2.6 * R], up=[0, 1, 0], foc=[c[0], c[1], z0]),
}
lut = GetColorTransferFunction(field)
lut.ApplyPreset("Viridis (matplotlib)", True)
lut.AutomaticRescaleRangeMode = "Never"
lut.Discretize = 0
cur = None
for f in files:
    k = int(re.findall(r"in_(\d+)\.vtk", f)[0])
    if cur is not None:
        Hide(cur, view)
        Delete(cur)
    cur = LegacyVTKReader(FileNames=[f])
    cd = Show(cur, view)
    ColorBy(cd, ("POINTS", field))
    cd.SetScalarBarVisibility(view, False)
    lut.RescaleTransferFunction(vmin, vmax)
    for name in views:
        cam = CAM[name]
        view.CameraPosition = cam["pos"]
        view.CameraFocalPoint = cam["foc"]
        view.CameraViewUp = cam["up"]
        view.CameraViewAngle = 30
        Render(view)
        SaveScreenshot(os.path.join(out, f"{name}_{k:06d}.png"), view, ImageResolution=[600, 640])
