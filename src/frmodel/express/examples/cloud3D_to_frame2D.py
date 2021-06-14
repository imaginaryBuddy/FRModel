import numpy as np
import pandas as pd
from scipy import ndimage

from frmodel.base.D2 import Frame2D
from frmodel.base.D2.frame._frame_plot import Frame2DPlot

from frmodel.base.D3 import Cloud3D
Frame2DPlot.set_browser_plotting()
#%%
""" Parameters
SCALING:            The general scaling of the data.
                    This will heavily affect performance.
MEDIAN_FILTER_SIZE: The Median Filter Window size.
                    Try to not increase this too much.
CLOUD_SAMPLES:      The number of random points from the Cloud sample to 
                    calculate the integer interpolations
PLOTLY_REDUCTION:   The number of folds to reduce the number of points plotly
                    has to render.
                    Do not decrease this too low.
MARKER_SIZE:        The size of the marker in Plotly. This doesn't change
                    performance too much.
"""
SCALING = 0.1
MEDIAN_FILTER_SIZE = 30
CLOUD_SAMPLES = 100000
PLOTLY_REDUCTION = 2
MARKER_SIZE = 3
CONTRAST = 2

#%%

f = Frame2D.from_image_spec(
    "rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_Red.tif",
    "rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_Green.tif",
    "rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_Blue.tif",
    "rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_RedEdge.tif",
    "rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_NIR.tif",
    scale=SCALING
)
c = Cloud3D.from_las("rsc/las/chestnut_10May/terra_las/cloud.las")
#%%
# The geotiff path is just any geotiff file that has the metadata of the coords.
z = c.to_frame(geotiff_path="rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_Red.tif",
               shape=(f.shape[0], f.shape[1]),
               samples=CLOUD_SAMPLES).data[...,0]
#%%
z = ndimage.median_filter(z, MEDIAN_FILTER_SIZE)
#%%

g = f.get_chns(True, chns=[f.CHN.XY])
ar = np.zeros([*z.shape, 6])
ar[...,-1] = z
ar[...,0:5] = g[..., [g.CHN.RGB, g.CHN.XY]].data
ar[...,:3] = ((ar[...,:3] - np.nanmin(ar[...,:3])) / (np.nanmax(ar[...,:3] - np.nanmin(ar[...,:3])))) ** (1/CONTRAST) * 255

df = pd.DataFrame(ar[::PLOTLY_REDUCTION, ::PLOTLY_REDUCTION].reshape([-1, 6]), columns=['R', 'G', 'B', 'X', 'Y', 'Z'])
df = df.dropna()
#%%
import plotly.graph_objs as go

trace = go.Scatter3d(x=df.X,
                     y=df.Y,
                     z=df.Z,
                     mode='markers',
                     marker=dict(size=MARKER_SIZE,
                                 color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(df.R.values, df.G.values, df.B.values)],
                                 opacity=0.9,))

data = [trace]

layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0),
                   scene=dict(xaxis={'title': 'x'},
                                  yaxis={'title': 'y'},
                                  zaxis={'title': 'z'},
                                  aspectratio=dict(x=0.5522, y=1, z=0.3)))
fig = go.Figure(data=data, layout=layout)
fig.show()
