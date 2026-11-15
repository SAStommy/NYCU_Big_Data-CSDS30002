"""Run the assignment experiments. Start in the BigData2026-Assignment2 folder.

Examples (PowerShell):
  .venv-relgnn\Scripts\python.exe scripts\run_experiment.py --part baseline
  .venv-relgnn\Scripts\python.exe scripts\run_experiment.py --part q2
  .venv-relgnn\Scripts\python.exe scripts\run_experiment.py --part q3
  .venv-relgnn\Scripts\python.exe scripts\run_experiment.py --part q4
  .venv-relgnn\Scripts\python.exe scripts\run_experiment.py --part all
"""
import argparse
import os
import gc
from pathlib import Path
import random
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from native_relgnn import EVENTS, PROJECT, DEVICE, load_tables, labels, make_model, make_snapshot

# Assignment predictions and the fixed-week answer belong in submission/.
SUBMISSION_DIR = PROJECT / "submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Historical weeks are Taiwan local Sunday-to-Sunday. Parquet timestamps are UTC.
# Q2/Q3 use only historical weeks with labels in the current dataset.
TRAIN_WEEKS = [
    ("2026-10-03 16:00:00", "2026-10-10 16:00:00"),
    ("2026-10-10 16:00:00", "2026-10-17 16:00:00"),
    ("2026-10-17 16:00:00", "2026-10-24 16:00:00"),
]
VALIDATION_WEEKS = [
    (("2026-10-17 16:00:00", "2026-10-24 16:00:00"), TRAIN_WEEKS[:2]),
    (("2026-10-24 16:00:00", "2026-10-31 16:00:00"), TRAIN_WEEKS),
]
TEST_WEEK = ("2026-10-31 16:00:00", "2026-11-07 16:00:00")
# The assignment prediction starts 2026-11-22 00:00 Taiwan time.
# Convert that cutoff to UTC; do not use the future test labels locally.
SUBMISSION_CUTOFF = "2026-11-21 16:00:00"
FINAL_TRAIN_WEEKS = list(dict.fromkeys(
    TRAIN_WEEKS + [VALIDATION_WEEKS[-1][0], TEST_WEEK]
))
EPOCHS = 20 #由6加大至30 訓練次數
CHANNELS = 64 #由32加大至128 圖網路隱藏層維度
HEADS = 2 #由2加大至4 圖網路多頭注意力頭數
LEARNING_RATE = 0.001 #可以改
PN_RATIO = 12 #正負樣本比例由12.0改為30/50
SEED = 42

# Variant settings: structural features, temporal features, route attention.
VARIANTS = {
    "base": ("base", False),
    "q2_topology": ("topology", False),
    "q2_attention": ("base", True),
    "q2_ours": ("topology", True),
    "q3_windows": ("windows", False),
    "q3_decay_7d": ("decay7", False),
    "q3_decay_14d": ("decay14", False),
    "q3_decay_28d": ("decay28", False),
    "q3_ours": ("temporal", False),
    "combined": ("combined", True),
}
Q2_SELECTION = {"q2_topology":("topology",False),"q2_attention":("base",True),
               "q2_ours":("topology",True)}
Q3_SELECTION = {"q3_windows":"windows","q3_decay_7d":"decay7",
                "q3_decay_14d":"decay14","q3_decay_28d":"decay28",
                "q3_ours":"temporal"}

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

def release_gpu_memory():
    """Collect dead Python references and return unused CUDA cache blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_sample(tables, feature_mode, week):
    start, end = week
    snapshot = make_snapshot(tables, start, feature_mode)
    target = labels(tables, start, end)
    # The graph contains only rows visible before its cutoff.
    user_ids = snapshot["frames"]["Users"].user_id.astype(int).to_numpy()
    y = target.loc[user_ids, EVENTS].to_numpy(dtype=np.float32)
    cohort = snapshot["cohort_mask"].detach().cpu().numpy()
    return snapshot, torch.tensor(y), torch.tensor(cohort)

def fit_model(tables, feature_mode, route_attention, weeks):
    seed_everything()
    samples = [build_sample(tables, feature_mode, week) for week in weeks]
    model = make_model(samples[0][0], route_attention, CHANNELS, HEADS).to(DEVICE)
    # L2 loss is weight decay控制
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Reweight rare targets from the training windows only.
    y_train = np.concatenate([y.numpy()[mask.numpy()] for _, y, mask in samples], axis=0)
    positives = y_train.sum(axis=0)

    #正負比例由12.0改為30/50
    pos_weight = torch.tensor(np.clip((len(y_train)-positives)/(positives+1), 1.0, PN_RATIO), dtype=torch.float32,  device=DEVICE)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for snapshot, target, cohort in samples:
            optimizer.zero_grad(set_to_none=True)
            # Move training labels and masks to GPU.
            target = target.to(DEVICE)
            cohort = cohort.to(DEVICE)

            logits = model(snapshot)
            loss = loss_fn(logits[cohort], target[cohort])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
        print(f"    epoch {epoch+1:02d}/{EPOCHS}, loss={total/len(samples):.4f}")
    return model

@torch.no_grad()
def predict(model, snapshot):
    model.eval()
    return torch.sigmoid(model(snapshot)).cpu().numpy()

def tune_thresholds(prob, truth):
    thresholds = {}
    for col_i, event in enumerate(EVENTS):
        best_f1, best_t = -1.0, 0.5
        for threshold in np.arange(0.05, 0.951, 0.01):
            score = f1_score(truth[:, col_i], prob[:, col_i] >= threshold, zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, float(threshold)
        thresholds[event] = best_t
    return thresholds

def score_predictions(prob, truth, thresholds):
    pred = np.column_stack([prob[:, i] >= thresholds[event] for i, event in enumerate(EVENTS)]).astype(int)
    per_event = [f1_score(truth[:, i], pred[:, i], zero_division=0) for i in range(len(EVENTS))]
    macro = float(np.mean(per_event))
    micro = float(f1_score(truth.ravel(), pred.ravel(), zero_division=0))
    return pred, {"macro_f1":macro,"micro_f1":micro,"score":(macro+micro)/2,
                  **{f"f1_{event}":per_event[i] for i,event in enumerate(EVENTS)}}

def output_frame(tables, user_ids, y_pred):
    test_ids = pd.read_csv(PROJECT / "submission" / "test_input.csv", dtype={"UserID":"int64"})
    user_map = tables["Users"].set_index("github_user_id").user_id
    internal_ids = test_ids.UserID.map(user_map)
    lookup = pd.Series(np.arange(len(user_ids)), index=user_ids)
    positions = internal_ids.map(lookup)
    out = pd.DataFrame({"UserID":test_ids.UserID})
    for i,event in enumerate(EVENTS):
        values = np.zeros(len(out), dtype=int)
        valid = positions.notna()
        values[valid.to_numpy()] = y_pred[positions.loc[valid].astype(int).to_numpy(), i]
        out[event] = values
    return out

def score_fixed_week(tables, snapshot, probability, thresholds, week):
    """Score a historical holdout week without writing its prediction CSV."""
    test = pd.read_csv(PROJECT / "submission" / "test_input.csv", dtype={"UserID":"int64"})
    user_map = tables["Users"].set_index("github_user_id").user_id
    internal = test.UserID.map(user_map)
    user_ids = snapshot["frames"]["Users"].user_id.astype(int).to_numpy()
    user_positions = pd.Series(np.arange(len(user_ids)), index=user_ids)
    positions = internal.map(user_positions)
    valid = positions.notna().to_numpy()

    test_probability = np.zeros((len(test), len(EVENTS)), dtype=np.float32)
    test_probability[valid] = probability[positions.loc[valid].astype(int).to_numpy()]
    target = labels(tables, *week)
    test_truth = np.column_stack([
        internal.map(target[event]).fillna(0).to_numpy(dtype=np.float32)
        for event in EVENTS
    ])
    _, metrics = score_predictions(test_probability, test_truth, thresholds)
    return metrics

def save_ablation_plot(rows, question):
    """Compare validation score with the latest historical holdout score."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".matplotlib"))
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(rows).sort_values("score",ascending=False)
    fig,ax=plt.subplots(figsize=(10,5))
    positions=np.arange(len(frame))
    width=.38
    bars=ax.bar(positions-width/2,frame.score,width,label="Rolling validation",color="#2878b5")
    holdout=ax.bar(positions+width/2,frame.fixed_test_score,width,label="Fixed latest-week holdout",color="#e69535")
    ax.set_ylabel("Score = (macro-F1 + micro-F1) / 2")
    ax.set_ylim(0,max(0.05,float(max(frame.score.max(),frame.fixed_test_score.max()))*1.2))
    ax.set_xticks(positions,frame.variant,rotation=25,ha="right")
    ax.set_title(f"{question} ablation: validation and fixed historical holdout")
    ax.legend()
    ax.bar_label(bars,labels=[f"{v:.3f}" for v in frame.score],padding=2,fontsize=8)
    ax.bar_label(holdout,labels=[f"{v:.3f}" for v in frame.fixed_test_score],padding=2,fontsize=8)
    fig.tight_layout()
    path=PROJECT/f"{question.lower()}_ablation.png"
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path.name}")

def save_framework(best_q2, best_q3, combined_mode, route_attention):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT/".matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    decay_label = {"q3_decay_7d":"7-day half-life decay", "q3_decay_14d":"14-day half-life decay",
                   "q3_decay_28d":"28-day half-life decay", "q3_ours":"3/7/14-day windows + 14-day decay",
                   "q3_windows":"3/7/14-day windows"}.get(best_q3, "temporal features")
    window_label = "3/7/14-day windows" if best_q3 in {"q3_windows","q3_ours"} else "sliding windows off"
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set(xlim=(0, 13), ylim=(0, 5)); ax.axis("off")
    boxes = [
        (0.25,1.8,2.0,1.5,"Eight tables\nUsers · Repositories\nPRs · Issues · Reviews\nComments · Pushes"),
        (2.75,1.8,2.0,1.5,"Time cutoffs\nTrain → validate → test\nNo future rows"),
        (5.25,3.0,2.5,1.3,"RelGNN\nAtomic routes\nComposite message passing\n"+best_q2),
        (5.25,0.7,2.5,1.3,"Temporal features\n"+window_label+"\n"+decay_label+"\n"+best_q3),
        (8.35,1.8,2.0,1.5,"Validation\nTask thresholds\nAblation selection"),
        (10.85,1.8,1.9,1.5,"Six activities\nBinary predictions\nCSV + score"),
    ]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",facecolor="#eaf2fb",edgecolor="#2d628f",linewidth=1.5))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    arrows=[((2.25,2.55),(2.75,2.55)),((4.75,2.55),(5.25,3.6)),((4.75,2.55),(5.25,1.35)),
            ((7.75,3.6),(8.35,2.8)),((7.75,1.35),(8.35,2.3)),((10.35,2.55),(10.85,2.55))]
    for start,end in arrows: ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=14,color="#555",linewidth=1.4))
    ax.set_title("GitHub activity prediction with official RelGNN",fontsize=15,pad=12)
    ax.text(6.5,0.12,f"Combined features: {combined_mode} | Task route attention: {route_attention} | Test: Taiwan 2026-11-01 to 2026-11-08",ha="center",fontsize=8,color="#444")
    fig.tight_layout()
    fig.savefig(PROJECT/"framework_315554001.png",dpi=180,bbox_inches="tight")
    plt.close(fig)

def save_schema_graph():
    """Draw all eight relational tables, primary keys, foreign keys, and FK edges."""
    tables = {
      "Users":["PK user_id","github_user_id","login","is_bot","is_cohort","first_seen_at"],
      "Repositories":["PK repo_id","github_repo_id","org_id (code)","owner_login","first_seen_at"],
      "PullRequests":["PK pr_id","FK repo_id","FK author_id","FK merged_by_id","head_repo_id (code)","first_seen_at"],
      "Issues":["PK issue_id","FK repo_id","FK author_id","FK pull_request_id","FK assignee_id","first_seen_at"],
      "IssueComments":["PK comment_id","FK issue_id","FK repo_id","FK author_id","created_at"],
      "PRReviews":["PK review_id","FK pr_id","FK repo_id","FK reviewer_id","first_seen_at"],
      "PRReviewComments":["PK comment_id","FK pr_id","FK repo_id","FK author_id","FK pull_request_review_id","FK in_reply_to_id","created_at"],
      "Pushes":["PK push_id","FK user_id","FK repo_id","created_at"],
    }
    table_html=[]
    for name,cols in tables.items():
        rows="".join(f'<TR><TD PORT="{col.split()[1].split("(")[0] if len(col.split())>1 else col}" ALIGN="LEFT">{col}</TD></TR>' for col in cols)
        table_html.append(f'"{name}" [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5"><TR><TD BGCOLOR="#dbeafe"><B>{name}</B></TD></TR>{rows}</TABLE>>];')
    edges=[("PullRequests","repo_id","Repositories","repo_id"),("PullRequests","author_id","Users","user_id"),("PullRequests","merged_by_id","Users","user_id"),
      ("Issues","repo_id","Repositories","repo_id"),("Issues","author_id","Users","user_id"),("Issues","pull_request_id","PullRequests","pr_id"),("Issues","assignee_id","Users","user_id"),
      ("IssueComments","issue_id","Issues","issue_id"),("IssueComments","repo_id","Repositories","repo_id"),("IssueComments","author_id","Users","user_id"),
      ("PRReviews","pr_id","PullRequests","pr_id"),("PRReviews","repo_id","Repositories","repo_id"),("PRReviews","reviewer_id","Users","user_id"),
      ("PRReviewComments","pr_id","PullRequests","pr_id"),("PRReviewComments","repo_id","Repositories","repo_id"),("PRReviewComments","author_id","Users","user_id"),
      ("PRReviewComments","pull_request_review_id","PRReviews","review_id"),("PRReviewComments","in_reply_to_id","PRReviewComments","comment_id"),
      ("Pushes","user_id","Users","user_id"),("Pushes","repo_id","Repositories","repo_id")]
    edge_lines=[f'"{src}":"{fk}" -> "{dst}":"{pk}" [label="{fk}"];' for src,fk,dst,pk in edges]
    dot='digraph Schema {\nrankdir=LR; graph [label="Assignment 2 Schema Graph | row time: 2026-10-01 to 2026-11-07 UTC",labelloc=t,fontsize=18,ranksep=1.0,nodesep=.45]; node [shape=plain,fontname="Arial"]; edge [fontname="Arial",fontsize=8,color="#536b85",arrowsize=.7];\n'
    dot+='\n'.join(table_html)+'\n'+ '\n'.join(edge_lines)+'\n}'
    dot_path=PROJECT/"schema_graph_315554001.dot"
    png_path=PROJECT/"schema_graph_315554001.png"
    dot_path.write_text(dot,encoding="utf-8")
    graphviz=shutil.which("dot")

def fit_variant(tables, name, mode, route_attention, prediction_cutoff=None):
    seed_everything()
    print(f"\n[{name}] features={mode}, task-route-attention={route_attention}")
    val_probs, val_truths = [], []
    for val_week, earlier_weeks in VALIDATION_WEEKS:
        val_model = fit_model(tables, mode, route_attention, earlier_weeks)
        val_snapshot, val_y, val_mask = build_sample(tables, mode, val_week)
        val_prob_all = predict(val_model, val_snapshot)
        val_probs.append(val_prob_all[val_mask.numpy()])
        val_truths.append(val_y.numpy()[val_mask.numpy()])
        del val_model, val_snapshot, val_y, val_mask, val_prob_all
        release_gpu_memory()
    val_prob = np.concatenate(val_probs, axis=0)
    val_truth = np.concatenate(val_truths, axis=0)
    thresholds = tune_thresholds(val_prob, val_truth)
    _, validation_score = score_predictions(val_prob, val_truth, thresholds)

    result={"variant":name,"thresholds":thresholds,"validation":validation_score}
    # Hold out the newest complete week currently available (Taiwan 11/1–11/8)
    # to inspect predictive ability. Keep the labels in memory; do not save a
    # variant prediction CSV.
    proxy_snapshot=make_snapshot(tables,TEST_WEEK[0],mode)
    proxy_model=fit_model(tables,mode,route_attention,
                          TRAIN_WEEKS+[VALIDATION_WEEKS[-1][0]])
    proxy_probability=predict(proxy_model,proxy_snapshot)
    result["fixed_test"]=score_fixed_week(
        tables,proxy_snapshot,proxy_probability,thresholds,TEST_WEEK
    )
    del proxy_model, proxy_snapshot, proxy_probability
    release_gpu_memory()
    if prediction_cutoff is not None:
        # Final submission prediction has no local truth file: the 11/22–11/29
        # activity has not happened. Use only rows available before its cutoff.
        test_snapshot=make_snapshot(tables,prediction_cutoff,mode)
        final_model=fit_model(tables,mode,route_attention,FINAL_TRAIN_WEEKS)
        test_prob_all=predict(final_model,test_snapshot)
        test_user_ids=test_snapshot["frames"]["Users"].user_id.astype(int).to_numpy()
        test_pred=np.column_stack([
            test_prob_all[:,i] >= thresholds[event]
            for i,event in enumerate(EVENTS)
        ]).astype(int)
        #result["model"]=final_model
        result["prediction"]=output_frame(tables,test_user_ids,test_pred)
        del final_model,test_snapshot,test_prob_all,test_pred
        release_gpu_memory()
    return result

def run(parts, include_base=True):
    seed_everything()
    tables=load_tables()
    print("Read 8 tables:", {k:len(v) for k,v in tables.items()})
    #if "baseline" in parts: save_schema_graph()
    summaries=[]
    configs=[]
    if "baseline" in parts or (("q2" in parts or "q3" in parts) and include_base):
        configs.append("base")
    if "q2" in parts: configs += ["q2_topology","q2_attention","q2_ours"]
    if "q3" in parts: configs += ["q3_windows","q3_decay_7d","q3_decay_14d","q3_decay_28d","q3_ours"]
    if ("q2" in parts or "q3" in parts) and not include_base:
        baseline_path=PROJECT/"baseline_validation.csv"
        if not baseline_path.exists():
            raise FileNotFoundError(
                "Run the baseline cell first; baseline_validation.csv is needed for the ablation chart."
            )
        baseline_rows=pd.read_csv(baseline_path)
        baseline_rows=baseline_rows.loc[baseline_rows.variant=="base"]
        if baseline_rows.empty:
            raise ValueError("baseline_validation.csv does not contain the base result.")
        summaries.extend(baseline_rows.to_dict("records"))
    for config in configs:
        if isinstance(config,tuple): name,mode,attention=config
        else: name,(mode,attention)=config,VARIANTS[config]
        result=fit_variant(tables,name,mode,attention)
        summaries.append({"variant":name,**result["validation"],
                          **{f"fixed_test_{k}":v for k,v in result["fixed_test"].items()},
                          **{f"threshold_{k}":v for k,v in result["thresholds"].items()}})
        if name == "base":
            pd.DataFrame([summaries[-1]]).to_csv(PROJECT/"baseline_validation.csv",index=False)
        release_gpu_memory()
    if summaries:
        pd.DataFrame(summaries).to_csv(PROJECT/"validation_ablation.csv",index=False)
        if "q2" in parts:
            q2_rows=[row for row in summaries if row["variant"]=="base" or row["variant"].startswith("q2_")]
            pd.DataFrame(q2_rows).to_csv(PROJECT/"q2_ablation.csv",index=False)
            save_ablation_plot(q2_rows,"Q2")
            best_q2=max((r for r in q2_rows if r["variant"].startswith("q2_")),key=lambda r:r["score"])
            (PROJECT/"q2_selection.txt").write_text(f"{best_q2['variant']}\n",encoding="utf-8")
        if "q3" in parts:
            q3_rows=[row for row in summaries if row["variant"]=="base" or row["variant"].startswith("q3_")]
            pd.DataFrame(q3_rows).to_csv(PROJECT/"q3_ablation.csv",index=False)
            save_ablation_plot(q3_rows,"Q3")
            best_q3=max((r for r in q3_rows if r["variant"].startswith("q3_")),key=lambda r:r["score"])
            (PROJECT/"q3_selection.txt").write_text(f"{best_q3['variant']}\n",encoding="utf-8")
    if "q4" in parts:
        # Q4 combines the strongest Q2 and Q3 variants selected by their
        # independent validation ablations. --part all refreshes these first.
        q2_path=PROJECT/"q2_ablation.csv"
        q3_path=PROJECT/"q3_ablation.csv"
        if not q2_path.exists() or not q3_path.exists():
            raise FileNotFoundError("Run --part q2 and --part q3 first to create their ablation tables.")
        best_q2=pd.read_csv(q2_path).sort_values("score",ascending=False).iloc[0].variant
        best_q3=pd.read_csv(q3_path).sort_values("score",ascending=False).iloc[0].variant
        q2_mode,attention=Q2_SELECTION[best_q2]
        q3_mode=Q3_SELECTION[best_q3]
        pieces=[] if q2_mode=="base" else q2_mode.split("+")
        pieces += q3_mode.split("+") if q3_mode!="temporal" else ["windows","decay"]
        combined_mode="+".join(dict.fromkeys(pieces)) or "base"
        print(f"Q4 selections: {best_q2} + {best_q3} -> {combined_mode}, route_attention={attention}")
        result=fit_variant(tables,"combined",combined_mode,attention,
                           prediction_cutoff=SUBMISSION_CUTOFF)
        result["prediction"].to_csv(SUBMISSION_DIR/"output_315554001.csv",index=False)
        pd.DataFrame([{ "variant":"combined",**result["validation"],
                       **{f"fixed_test_{k}":v for k,v in result["fixed_test"].items()},
                       **{f"threshold_{k}":v for k,v in result["thresholds"].items()}}]).to_csv(
                           PROJECT/"q4_validation.csv",index=False)
        #save_framework(best_q2,best_q3,combined_mode,attention)
        (PROJECT/"q4_selection.txt").write_text(
            f"Q2={best_q2}\nQ3={best_q3}\ncombined_features={combined_mode}\nroute_attention={attention}\n",
            encoding="utf-8")
    print("Completed. Q2/Q3 validation scores are in the ablation tables and plots.")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",choices=["baseline","q2","q3","q4","all"],required=True)
    args=parser.parse_args()
    parts={"all":{"baseline","q2","q3","q4"}}.get(args.part,{args.part})
    run(parts)

if __name__=="__main__":
    main()
