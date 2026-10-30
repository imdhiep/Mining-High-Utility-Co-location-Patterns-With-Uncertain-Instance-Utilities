"""Locations of external inputs, overridable by environment variables.
FSQ_DIR      folder with dataset_TSMC2014_NYC.txt and dataset_TSMC2014_TKY.txt (Foursquare TSMC2014)
PSOMUCP_DIR  clone of https://github.com/yuanshuaikang/PSO-MUCP (authors' code in HUCPM/, public data in DATA/)"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSQ_DIR = os.environ.get("FSQ_DIR", os.path.join(ROOT, "data", "dataset_tsmc2014"))
PSOMUCP_DIR = os.environ.get("PSOMUCP_DIR", os.path.join(ROOT, "external", "PSO-MUCP"))


def fsq(city):
    return os.path.join(FSQ_DIR, f"dataset_TSMC2014_{city}.txt")
