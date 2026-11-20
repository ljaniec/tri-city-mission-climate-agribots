import time, numpy as np
from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
from prunemaxxing.geo.site import load_site
from prunemaxxing.planning.route_planner import plan_global_route
from prunemaxxing.planning.path_geometry import min_turn_radius, curvature
import networkx as nx
t0=time.time()
site=load_site(); dem=select_pilot_window(crop_target_field(load_dem('viladecans_full_dem_10m.TIF')),12.0)
twin=build_field_twin_from_dem(dem)
plan=plan_global_route(site,twin)
print('components', nx.number_connected_components(plan.network))
print('plan time', round(time.time()-t0,2))
print('length', round(plan.length_m), {k:round(v) for k,v in plan.label_length().items()})
print('lanes', len(plan.lanes), 'stations', len(plan.stations), 'cells', len(plan.cells), 'unreach', len(plan.unreachable_cells))
k=np.abs(curvature(plan.path)); print('min R (99.5pct)', round(min_turn_radius(plan.path),2), 'max kappa', np.nanmax(k).round(3), 'n pts kappa>1/6', int((k>1/6.0).sum()))
print('cells per station', [len(s.cells) for s in plan.stations], 'max dist', round(max(c.dist_m for c in plan.cells),1))
print('total target t', round(sum(c.target_kg for c in plan.cells)/1000), 'area ha', round(sum(c.area_m2 for c in plan.cells)/1e4,2))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(12,12))
for p in site.parcels: ax.plot(*p.polygon.exterior.xy,'k-',lw=0.8)
for b in plan.bedded_areas.values(): ax.plot(*b.exterior.xy,'k:',lw=0.5)
for t in site.tracks: ax.plot(*t.line.xy,color='tan',lw=3)
for w in site.weeds: ax.fill(*w.polygon.exterior.xy,color='r',alpha=0.4)
for w in plan.inflated_weeds: ax.plot(*w.exterior.xy,'r--',lw=0.5)
for o in site.obstacles: ax.fill(*o.polygon.exterior.xy,color='m',alpha=0.6)
cols={'transit':'b','lane':'g','headland':'orange','weed_detour':'r'}
for lab,c in cols.items():
    m=plan.labels==lab; ax.scatter(plan.path[m,0],plan.path[m,1],s=0.5,c=c,label=lab)
hi=np.where(k>1/6.0)[0]; ax.scatter(plan.path[hi,0],plan.path[hi,1],s=20,facecolors='none',edgecolors='k')
st=np.array([s.xy for s in plan.stations]); ax.scatter(st[:,0],st[:,1],marker='s',c='k',s=12)
ax.plot(*plan.path[0],'k*',ms=15)
ax.set_aspect('equal'); ax.legend(); fig.savefig('outputs/dev/plan.png',dpi=80)
ds=np.diff(plan.s); print('min ds', ds.min(), 'n ds<0.05', int((ds<0.05).sum()))
for i in hi[:40]:
    print(i, plan.labels[i], (plan.path[i]-[site.origin_e,site.origin_n]).round(2), round(k[i],3), 'ds', round(ds[min(i,len(ds)-1)],3))
