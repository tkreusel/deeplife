# models/egnn_v2.py — compatibility shim
# Renamed to models/se3flow_energy.py  (SE3FlowLayer, SE3FlowEnergyNet)
# This file exists only so that old checkpoints with model_type="flowmatch_v2_energy"
# still load correctly. Do not use directly — import from se3flow_energy instead.
from models.se3flow_energy import SE3FlowLayer as EGNNv2Layer                   # noqa: F401
from models.se3flow_energy import SE3FlowEnergyNet as EGNNv2EnergyScoreNetwork  # noqa: F401
