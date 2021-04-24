import numpy as np
import pandas as pd
from sompy.sompy import SOMFactory

dataset = pd.read_csv("en_climate_summaries_BC_01-2020.csv")
names = {
    "Long": "Longitude (West - , degrees)",
    "Lat": "Latitude (North + , degrees)",
    "Tm": "Mean Temperature (°C)",
    "Tx": "Highest Monthly Maximum Temperature (°C)",
    "Tn": "Lowest Monthly Minimum Temperature (°C)",
    "S": "Snowfall (cm)",
    "P": "Total Precipitation (mm)",
    "HDD": "Degree Days below 18 °C",
}
data = dataset[names.keys()]
data = data.apply(pd.to_numeric, errors="coerce")
data = data.dropna()

print(data.head())

# create the SOM network and train it. You can experiment with different normalizations and initializations
sm = SOMFactory().build(
    data.values,
    normalization="var",
    initialization="pca",
    component_names=list(names.values()),
)
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print(
    "Topographic error = %s; Quantization error = %s"
    % (topographic_error, quantization_error)
)

# component planes view
from sompy.visualization.mapview import View2D

view2D = View2D(10, 10, "rand data", text_size=12)
view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)

# U-matrix plot
from sompy.visualization.umatrix import UMatrixView

umat = UMatrixView(width=10, height=10, title="U-matrix")
umat.show(sm)