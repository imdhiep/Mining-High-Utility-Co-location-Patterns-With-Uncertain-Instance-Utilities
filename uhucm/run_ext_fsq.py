import os, sys, fsq_loader, ext_baselines, paths
c = sys.argv[1]
rows, u, NU, info = fsq_loader.load(paths.fsq(c), c)
ext_baselines.run_all(rows, f"FSQ_{c}", "out_baseline", os.path.join(paths.PSOMUCP_DIR, "HUCPM"))
