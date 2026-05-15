from smap_msl_dataset_api import SMAPMSLChannelDataset, working_channels, Quantizer
from pathlib import Path
ds = SMAPMSLChannelDataset('M-1', split='test', data_dir=Path(''))
print('channel M-1 test windows:', len(ds))
print('anomaly mask sum:', ds.anomaly_mask.sum())
print('telemetry scaled range:', ds.telemetry_scaled.min(), 'to', ds.telemetry_scaled.max())
q = Quantizer()
levels = q.quantize(ds.telemetry_scaled)
print('quantized levels range:', levels.min(), 'to', levels.max())
print('saturated at boundaries:', (levels == 0).sum(), '+', (levels == 255).sum())
print('working channel count:', len(working_channels(data_dir=Path(''))))

"""
channel M-1 test windows: 2027
anomaly mask sum: 1141
telemetry scaled range: -1.094110086481219 to 0.868885763940473
quantized levels range: 0 to 239
saturated at boundaries: 384 + 0
working channel count: 52
"""