#!/usr/bin/env python
"""Aggregate strategy CSV logs and plot Pass Rate vs Test-Time Compute."""
import argparse, glob, os, re
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hub_utils import push_file

_LOG_RE=re.compile(r"(?P<strategy>.+)_(?P<model>qwen1_5b|qwen7b)_(?P<dataset>humaneval|mbpp)_(?P<split>.+)_limit(?P<limit>\d+)\.csv$")

def parse(path):
    m=_LOG_RE.search(os.path.basename(path))
    if not m:return None
    return {"strategy":m.group("strategy"),"model":m.group("model"),"dataset":m.group("dataset"),"split":m.group("split"),"limit":int(m.group("limit"))}

def summarize(path):
    df=pd.read_csv(path)
    if df.empty:return None
    if "flops_estimated_flops" not in df.columns: raise ValueError(f"{path} has no FLOP column")
    n=len(df); passed=df["passed"].astype(bool)
    total_flops=df["flops_estimated_flops"].astype(float).sum(); tokens=df.get("flops_total_tokens",pd.Series([0]*n)).astype(int).sum()
    return {"n_problems":n,"n_passed":int(passed.sum()),"pass_rate":float(passed.mean()),"total_flops":total_flops,"mean_test_time_compute_flops":total_flops/n,"total_tokens":int(tokens),"mean_tokens_per_problem":float(tokens/n)}

def plot(df,path):
    plt.figure(figsize=(10,7))
    for (model,dataset,split),g in df.groupby(["model","dataset","split"]):
        g=g.sort_values("mean_test_time_compute_flops"); label=f"{model} / {dataset} / {split}"
        plt.plot(g["mean_test_time_compute_flops"],100*g["pass_rate"],marker="o",label=label)
        for _,r in g.iterrows(): plt.annotate(r["strategy"],(r["mean_test_time_compute_flops"],100*r["pass_rate"]),fontsize=8,xytext=(4,4),textcoords="offset points")
    plt.xscale("log"); plt.xlabel("Test-Time Compute: mean estimated FLOPs per problem (log scale)"); plt.ylabel("Pass Rate (%)"); plt.title("Pass Rate vs. Test-Time Compute"); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=200,bbox_inches="tight"); plt.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--logs_dir",default="logs"); p.add_argument("--out_dir",default="logs/results"); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--no_push",action="store_true"); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    records=[]
    for path in glob.glob(os.path.join(a.logs_dir,"*.csv")):
        meta=parse(path)
        if meta:
            s=summarize(path)
            if s:records.append({**meta,**s})
    if not records:return
    df=pd.DataFrame(records).sort_values(["dataset","split","model","strategy"]); out=os.path.join(a.out_dir,"summary_table.csv"); df.to_csv(out,index=False); print(df.to_string(index=False)); print(f"Wrote {out}")
    plotpath=os.path.join(a.out_dir,"pass_rate_vs_test_time_compute.png"); plot(df,plotpath); print(f"Wrote {plotpath}")
    if not a.no_push:
        with open(a.config,encoding="utf-8") as f:cfg=yaml.safe_load(f)
        push_file(out,cfg,f"logs/results/{os.path.basename(out)}"); push_file(plotpath,cfg,f"logs/results/{os.path.basename(plotpath)}")
if __name__=="__main__":main()
