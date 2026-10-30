"""Foursquare quality at d = 50 m with the calibrated z = 2.45 and caliper matching. Usage: python run_cal_fsq.py CITY"""
import sys
import fsq_loader, paths
from suite import Suite
city = sys.argv[1]
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city)
S = Suite(f"FSQ_{city}_z2.45_cal", rows, 50.0, "out_rev2", zmain=2.45, log=lambda s: None); S.quality(); S.save()
print("DONE", flush=True)
