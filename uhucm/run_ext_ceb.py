"""External baselines (E5) on Foursquare with the compound EB posteriors. Usage: python run_ext_ceb.py CITY
Needs PSOMUCP_DIR (clone of the PSO-MUCP repository) and FSQ_DIR."""
import os, sys, fsq_loader, ext_baselines, paths
c = sys.argv[1]
rows, u, NU, info = fsq_loader.load(paths.fsq(c), c, model="ceb"); UB, Zh = info.pop("UB"), info.pop("Zhat")
ext_baselines.run_all(rows, f"CEB_{c}", "out_suite_ceb", os.path.join(paths.PSOMUCP_DIR, "HUCPM"), UB=UB, Zhat=Zh)
