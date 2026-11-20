import time, sys, json, numpy as np
from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
from prunemaxxing.geo.site import load_site
from prunemaxxing.planning.mission import MissionConfig, run_mission
mode = sys.argv[1] if len(sys.argv)>1 else 'dust'
nst = int(sys.argv[2]) if len(sys.argv)>2 else None
site=load_site(); twin=build_field_twin_from_dem(select_pilot_window(crop_target_field(load_dem('viladecans_full_dem_10m.TIF')),12.0))
t0=time.time()
res=run_mission(site,twin,MissionConfig(payload_mode=mode, max_stations=nst))
print('sim wall', round(time.time()-t0,1),'s')
print(json.dumps(res.kpis, indent=1, default=str))
for e in res.events:
    if e['kind'] in ('obstacle_detected','detour_spliced'): print(e)
import pickle; pickle.dump(res, open(f'outputs/dev/mission_{mode}.pkl','wb'))
