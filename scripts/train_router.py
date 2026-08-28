#!/usr/bin/env python
"""Train learned latent router with JSON progress checkpoints and safetensors weights."""
import argparse, os, sys, collections
import torch, torch.nn as nn, yaml
from safetensors.torch import save_file, load_file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_dataset
from src.flop_utils import FlopLedger
from src.model_utils import load_model_and_tokenizer
from src.strategies.best_of_n import run_best_of_n
from src.strategies.greedy import run_greedy
from src.strategies.router import LatentRouterNet, embed_problems
from src.strategies.tree_search import run_tree_search
from src.strategies.self_consistency import run_self_consistency
from src.checkpoint_utils import atomic_json_save, load_json_checkpoint
from src.hub_utils import push_file, download_file
_FUNCS = {
    "greedy": run_greedy,
    "best_of_n": run_best_of_n,
    "self_consistency": run_self_consistency,
    "tree_search": run_tree_search,
}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--model",required=True,choices=["qwen1_5b","qwen7b"]); p.add_argument("--dataset",default="mbpp",choices=["humaneval","mbpp"]); p.add_argument("--split",default="train"); p.add_argument("--limit",type=int,default=None); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--out",default="checkpoints/router.safetensors"); p.add_argument("--no_push",action="store_true"); a=p.parse_args()
 with open(a.config,encoding="utf-8") as f: cfg=yaml.safe_load(f)
 push_every_n = cfg["hub"].get("push_every_n_problems", 30)
 os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True); os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
 problems=load_dataset(a.dataset,split=a.split,limit=a.limit); tag=len(problems); progress_path=os.path.join(cfg["paths"]["checkpoints_dir"],f"router_labels_{a.model}_{a.dataset}_{a.split}_limit{tag}.json"); hub_progress=f"checkpoints/{os.path.basename(progress_path)}"
 if not os.path.exists(progress_path) and not a.no_push: download_file(cfg,hub_progress,progress_path)
 completed=load_json_checkpoint(progress_path).get("completed",{}) if os.path.exists(progress_path) else {}
 model,tokenizer,num_params=load_model_and_tokenizer(a.model,cfg); strategies=cfg["router"]["strategies_available"]; lam=cfg["router"]["cost_penalty_lambda"]
 
 for i, problem in enumerate(problems, 1):
    pid = problem["problem_id"]
    completed.setdefault(pid, {})

    for strategy in strategies:
        if strategy in completed[pid]:
            continue

        print(f"[{i}/{len(problems)}] {pid} -> {strategy}")

        l = FlopLedger()
        r = _FUNCS[strategy](
            model, tokenizer, num_params, [problem], cfg, l
        )[0]
        fr = l.as_dicts()[0]

        completed[pid][strategy] = {
            "passed": bool(r["passed"]),
            "flops": float(fr["estimated_flops"]),
        }

        # Save locally after every strategy
        atomic_json_save(
            {
                "version": 3,
                "completed": completed,
                "metadata": {
                    "model": a.model,
                    "dataset": a.dataset,
                    "split": a.split,
                    "limit": tag,
                },
            },
            progress_path,
        )

    # Push after every N fully completed problems
    if not a.no_push and (
        i % push_every_n == 0 or i == len(problems)
    ):
        print(f"Pushing progress checkpoint after {i} problems...")
        push_file(progress_path, cfg, hub_progress)
 all_flops=[completed[pb["problem_id"]][st]["flops"] for pb in problems for st in strategies]; max_flops=max(all_flops) if all_flops else 1.0; labels=[]
 for problem in problems:
  rec=completed[problem["problem_id"]]; best=max(strategies,key=lambda st:(1.0 if rec[st]["passed"] else 0.0)-lam*(rec[st]["flops"]/max_flops)); labels.append(strategies.index(best))
 counts=collections.Counter(labels); print("Router label distribution:",{strategies[k]:v for k,v in sorted(counts.items())})
 embeddings=embed_problems(problems,cfg); X=torch.tensor(embeddings,dtype=torch.float32); y=torch.tensor(labels,dtype=torch.long)
 net=LatentRouterNet(embeddings.shape[1],cfg["router"]["hidden_dim"],cfg["router"]["latent_dim"],len(strategies)); opt=torch.optim.Adam(net.parameters(),lr=cfg["router"]["lr"]); loss_fn=nn.CrossEntropyLoss()
 train_state=a.out+".train.json"; weight_resume=a.out+".train.safetensors"; start_epoch=0
 if not os.path.exists(train_state) and not a.no_push: download_file(cfg,f"checkpoints/{os.path.basename(train_state)}",train_state)
 if not os.path.exists(weight_resume) and not a.no_push: download_file(cfg,f"checkpoints/{os.path.basename(weight_resume)}",weight_resume)
 if os.path.exists(train_state) and os.path.exists(weight_resume):
  state=load_json_checkpoint(train_state); net.load_state_dict(load_file(weight_resume)); start_epoch=int(state.get("epoch",0)); print(f"Resuming router training from epoch {start_epoch}")
 net.train()
 for epoch in range(start_epoch,cfg["router"]["epochs"]):
  opt.zero_grad(); logits,_=net(X); loss=loss_fn(logits,y); loss.backward(); opt.step(); acc=(logits.argmax(-1)==y).float().mean().item(); print(f"epoch {epoch+1}/{cfg['router']['epochs']} loss={loss.item():.4f} train_acc={acc:.2%}")
  save_file(net.state_dict(),weight_resume); atomic_json_save({"version":3,"epoch":epoch+1,"loss":float(loss.item()),"metadata":{"model":a.model,"dataset":a.dataset,"split":a.split,"limit":tag}},train_state)
  if not a.no_push: push_file(weight_resume,cfg,f"checkpoints/{os.path.basename(weight_resume)}"); push_file(train_state,cfg,f"checkpoints/{os.path.basename(train_state)}")
 save_file(net.state_dict(),a.out); print(f"Saved trained learned latent router to {a.out}")
 if not a.no_push: push_file(a.out,cfg,f"checkpoints/{os.path.basename(a.out)}")
if __name__=="__main__": main()
