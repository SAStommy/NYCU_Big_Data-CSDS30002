"""Full-batch GitHub activity models built on the supplied official RelGNN code.

Run from BigData2026-Assignment2. Paths and experiment settings live near the
top of run_experiment.py so they are easy to edit for coursework experiments.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[RelGNN] Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"[RelGNN] GPU: {torch.cuda.get_device_name(0)}")
from types import SimpleNamespace
from torch_frame import stype
from relbench.base import Database, Table
from relbench.modeling.graph import make_pkey_fkey_graph

PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / "data" / "db").exists():
    raise RuntimeError("Run scripts from the BigData2026-Assignment2 project root.")
# Edit this relative path if the RelGNN checkout is not beside this project.
RELGNN_ROOT = PROJECT.parent / "RelGNN"
EXAMPLES = RELGNN_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
from atomic_routes import get_atomic_routes
from relgnn_model import RelGNN_Model

# RelGNN's hetero wrapper interprets the atomic-route kind ('dim-dim',
# 'dim-fact-dim') as a node type in its generic warning. It is not a missing
# GitHub table node; keep the console focused on actionable warnings.
warnings.filterwarnings("ignore", message="There exist node types")

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]
PK = {"Users":"user_id", "Repositories":"repo_id", "PullRequests":"pr_id",
      "Issues":"issue_id", "IssueComments":"comment_id", "PRReviews":"review_id",
      "PRReviewComments":"comment_id", "Pushes":"push_id"}
ROW_TIME = {"Users":"first_seen_at", "Repositories":"first_seen_at",
            "PullRequests":"first_seen_at", "Issues":"first_seen_at",
            "IssueComments":"created_at", "PRReviews":"first_seen_at",
            "PRReviewComments":"created_at", "Pushes":"created_at"}
FK = {
 "PullRequests":[("repo_id","Repositories"),("author_id","Users")],
 "Issues":[("repo_id","Repositories"),("author_id","Users"),
            ("pull_request_id","PullRequests"),("assignee_id","Users")],
 "IssueComments":[("issue_id","Issues"),("repo_id","Repositories"),("author_id","Users")],
 "PRReviews":[("pr_id","PullRequests"),("repo_id","Repositories"),("reviewer_id","Users")],
 "PRReviewComments":[("pr_id","PullRequests"),("repo_id","Repositories"),("author_id","Users"),
                     ("pull_request_review_id","PRReviews"),("in_reply_to_id","PRReviewComments")],
 "Pushes":[("repo_id","Repositories"),("user_id","Users")],
}
EVENTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
EVENT_TABLE = {
 "open_pr":("PullRequests","created_at","author_id"),
 "open_issue":("Issues","created_at","author_id"),
 "issue_comment":("IssueComments","created_at","author_id"),
 "pr_review":("PRReviews","submitted_at","reviewer_id"),
 "pr_review_comment":("PRReviewComments","created_at","author_id"),
 "push":("Pushes","created_at","user_id"),
}
FEATURE_DIM = 50
BASE_END = 16
TOPO_START, TOPO_END = 16, 26
TIME_START, TIME_END = 26, 50

def load_tables(db_dir=None):
    db_dir = Path(db_dir) if db_dir else PROJECT / "data" / "db"
    return {name: pd.read_parquet(db_dir / f"{name}.parquet") for name in TABLES}

def _time(series):
    return pd.to_datetime(series, errors="coerce")

def labels(tables, start, end):
    """Binary labels for one exact UTC interval, aligned to the full Users PK."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    y = pd.DataFrame(0, index=tables["Users"].user_id.astype(int), columns=EVENTS, dtype=np.float32)
    for event, (table, time_col, user_col) in EVENT_TABLE.items():
        df = tables[table]
        ts = _time(df[time_col])
        mask = ts.ge(start) & ts.lt(end) & df[user_col].notna()
        if event == "open_issue":
            mask &= ~df.is_pull_request.fillna(False)
        actor_ids = set(df.loc[mask, user_col].astype(int))
        y.loc[y.index.intersection(actor_ids), event] = 1.0
    return y

def _to_numpy(values, index):
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce").fillna(0).to_numpy(dtype=np.float32)

def make_snapshot(tables, cutoff, feature_mode="base"):
    """Build cutoff-safe table nodes, FK edges, and numeric features."""
    cutoff = pd.Timestamp(cutoff)
    frames, old_to_new, times = {}, {}, {}
    for table in TABLES:
        source = tables[table]
        ts = _time(source[ROW_TIME[table]])
        frame = source.loc[ts.lt(cutoff)].copy().reset_index(drop=True)
        frames[table] = frame
        old_to_new[table] = pd.Series(np.arange(len(frame)), index=frame[PK[table]].astype(int))
        times[table] = _time(frame[ROW_TIME[table]])

    x = {table: np.zeros((len(frames[table]), FEATURE_DIM), dtype=np.float32) for table in TABLES}
    for table, df in frames.items():
        age = (cutoff - times[table]).dt.total_seconds().fillna(0).clip(lower=0) / 86400.0
        x[table][:, 0] = np.log1p(age.to_numpy(dtype=np.float32))
        x[table][:, 1] = 1.0
        if table == "Users":
            x[table][:, 2] = _to_numpy(df.is_bot.fillna(False).astype(int), df.index)
            x[table][:, 3] = _to_numpy(df.is_cohort.fillna(False).astype(int), df.index)
        elif table == "PullRequests":
            for slot, col in [(4,"draft"),(5,"is_cross_repo")]:
                x[table][:, slot] = _to_numpy(df[col].fillna(False).astype(int), df.index)
        elif table == "Issues":
            x[table][:, 6] = _to_numpy(df.is_pull_request.fillna(False).astype(int), df.index)
            x[table][:, 7] = np.log1p(_to_numpy(df.num_labels, df.index))
            x[table][:, 8] = _to_numpy(df.locked.fillna(False).astype(int), df.index)
        elif table == "IssueComments":
            x[table][:, 9] = _to_numpy(df.is_pull_request.fillna(False).astype(int), df.index)
        elif table == "PRReviews":
            state = df.state.fillna("").str.lower()
            for slot, value in [(10,"approved"),(11,"changes_requested"),(12,"commented")]:
                x[table][:, slot] = (state == value).to_numpy(dtype=np.float32)
        elif table == "PRReviewComments":
            x[table][:, 13] = np.log1p(_to_numpy(df.position.fillna(0), df.index).clip(0, 100000))
            x[table][:, 14] = (df.subject_type.fillna("").str.lower() == "file").to_numpy(dtype=np.float32)
        elif table == "Pushes":
            x[table][:, 14] = _to_numpy(df.is_new_ref.fillna(False).astype(int), df.index)
            x[table][:, 15] = (df.ref_type.fillna("").str.lower() == "tag").to_numpy(dtype=np.float32)

    # Q2 structural features: activity counts, repo diversity, and repeated
    # User-Repository co-occurrence. These are computed only before cutoff.
    feature_parts=set(feature_mode.split("+"))
    add_topology=bool(feature_parts & {"topology","proposed","combined"})
    add_windows=bool(feature_parts & {"windows","temporal","proposed","combined"})
    decay_modes={part for part in feature_parts if part.startswith("decay")}
    add_decay=bool(decay_modes | (feature_parts & {"temporal","proposed","combined"}))
    half_life=14
    for part in decay_modes:
        suffix=part.removeprefix("decay")
        if suffix.isdigit(): half_life=int(suffix)
    if add_topology:
        user_lookup = old_to_new["Users"]
        repo_lookup = old_to_new["Repositories"]
        user_table = frames["Users"]
        user_pos = user_table.user_id.astype(int).to_numpy()
        ucounts = np.zeros((len(user_table), 10), dtype=np.float32)
        pairs = [set() for _ in range(len(user_table))]
        specs = [("PullRequests","author_id","repo_id",0),("Issues","author_id","repo_id",1),
                 ("IssueComments","author_id","repo_id",2),("PRReviews","reviewer_id","repo_id",3),
                 ("PRReviewComments","author_id","repo_id",4),("Pushes","user_id","repo_id",5)]
        for table, user_col, repo_col, col in specs:
            df = frames[table]
            if user_col not in df: continue
            user_new = df[user_col].map(user_lookup)
            valid = user_new.notna()
            counts = user_new.loc[valid].astype(int).value_counts()
            ucounts[counts.index.to_numpy(), col] += np.log1p(counts.to_numpy(dtype=np.float32))
            for ui, ri in zip(user_new[valid].astype(int), df.loc[valid, repo_col]):
                if pd.notna(ri): pairs[ui].add(int(ri))
        for i, repos in enumerate(pairs):
            ucounts[i, 6] = np.log1p(len(repos))
            ucounts[i, 7] = len(repos) / max(1.0, float(np.expm1(ucounts[i, :6]).sum()))
        x["Users"][:, TOPO_START:TOPO_START+8] = ucounts[:, :8]
        # Repository side: event volume and distinct contributor count.
        repos = frames["Repositories"]
        rcounts = np.zeros((len(repos),2),dtype=np.float32)
        for table,user_col,repo_col,_ in specs:
            df=frames[table]
            if repo_col not in df: continue
            rr=df[repo_col].map(repo_lookup); valid=rr.notna()
            counts=rr.loc[valid].astype(int).value_counts()
            rcounts[counts.index.to_numpy(),0]+=counts.to_numpy(dtype=np.float32)
            actors=df.loc[valid,[user_col,repo_col]].dropna().drop_duplicates()
            for rid,n in actors.groupby(repo_col)[user_col].nunique().items():
                pos=repo_lookup.get(int(rid))
                if pd.notna(pos): rcounts[int(pos),1]=n
        x["Repositories"][:,TOPO_START+8]=np.log1p(rcounts[:,0])
        x["Repositories"][:,TOPO_START+9]=np.log1p(rcounts[:,1])

    # Q3 temporal features: 3/7/14-day activity counts and 14-day half-life decay.
    if add_windows or add_decay:
        user_lookup = old_to_new["Users"]
        activity_idx = {e:i for i,e in enumerate(EVENTS)}
        for event,(table,time_col,user_col) in EVENT_TABLE.items():
            df=frames[table]
            if user_col not in df: continue
            event_time=_time(df[time_col])
            user_pos=df[user_col].map(user_lookup)
            valid=event_time.lt(cutoff)&user_pos.notna()
            age=(cutoff-event_time).dt.total_seconds()/86400.0
            eidx=activity_idx[event]
            if add_windows:
                for j,days in enumerate((3,7,14)):
                    mask=valid & age.le(days)
                    counts=user_pos.loc[mask].astype(int).value_counts()
                    x["Users"][counts.index.to_numpy(),TIME_START+eidx*3+j]=np.log1p(counts.to_numpy(dtype=np.float32))
            if add_decay:
                mask=valid & age.le(56)
                if mask.any():
                    w=np.exp(-np.log(2)*age.loc[mask].to_numpy(dtype=np.float32)/float(half_life))
                    weighted=pd.DataFrame({"user":user_pos.loc[mask].astype(int).to_numpy(),"w":w}).groupby("user").w.sum()
                    x["Users"][weighted.index.to_numpy(dtype=int),TIME_START+18+eidx]=weighted.to_numpy(dtype=np.float32)

    # Build the graph with RelBench's official table/PK/FK graph builder.
    # The assignment's numeric features remain ordinary numerical columns.
    db_tables = {}
    stype_dict = {}
    feature_cols = [f"feature_{i:02d}" for i in range(FEATURE_DIM)]
    for table in TABLES:
        graph_df = pd.DataFrame(x[table], columns=feature_cols)
        graph_df["__id__"] = np.arange(len(graph_df), dtype=np.int64)
        graph_df["__time__"] = times[table].reset_index(drop=True)
        fkey_map = {}
        for fk_col, parent in FK.get(table, []):
            original = frames[table][fk_col]
            graph_df[fk_col] = original.map(old_to_new[parent]).astype("float64")
            fkey_map[fk_col] = parent
        db_tables[table] = Table(
            df=graph_df,
            fkey_col_to_pkey_table=fkey_map,
            pkey_col="__id__",
            time_col="__time__",
        )
        stype_dict[table] = {col: stype.numerical for col in feature_cols}

    graph, col_stats = make_pkey_fkey_graph(
        Database(db_tables), col_to_stype_dict=stype_dict
    )
    for table in TABLES:
        graph[table].num_nodes = len(frames[table])

    # Move the graph and its tensor attributes to GPU.
    graph = graph.to(DEVICE)

    routes = get_atomic_routes(graph.edge_types)
    # RelGNN_Model follows the same interface used by the official examples.
    # A full-batch task has one shared prediction cutoff, so batch indices are 0.
    batch = FullGraphBatch(graph, cutoff)
    return {"frames":frames,"graph":graph,"col_stats":col_stats,
            "batch":batch,"routes":routes,
            "user_index":old_to_new["Users"],"cohort_mask":torch.tensor(frames["Users"].is_cohort.fillna(False).to_numpy(),dtype=torch.bool, device=DEVICE,)}

class FullGraphBatch:
    """Minimal full-graph batch exposing the fields expected by RelGNN_Model."""
    def __init__(self, graph, cutoff):
        self.graph = graph
        self.tf_dict = {node: graph[node].tf for node in graph.node_types}
        self.time_dict = {node: graph[node].time for node in graph.node_types}
        self.batch_dict = {
            node: torch.zeros(graph[node].num_nodes, dtype=torch.long, device=DEVICE)
            for node in graph.node_types
        }
        self.edge_index_dict = graph.edge_index_dict
        self.user_store = SimpleNamespace(
            seed_time=torch.full(
                (graph["Users"].num_nodes,), int(pd.Timestamp(cutoff).value // 10**9),
                dtype=torch.long,device=DEVICE,
            )
        )

    def __getitem__(self, node_type):
        if node_type != "Users":
            return self.graph[node_type]
        return self.user_store

class OfficialRelGNN(RelGNN_Model):
    """Official RelGNN_Model with the assignment's six-label output layer."""
    def __init__(self,snapshot,channels=32,heads=2,layers=1):
        super().__init__(
            data=snapshot["graph"], col_stats_dict=snapshot["col_stats"],
            num_model_layers=layers, channels=channels, out_channels=len(EVENTS),
            aggr="sum", norm="batch_norm", atomic_routes=snapshot["routes"],
            num_heads=heads, simplified_MP=True,
        )
    def forward(self,snapshot):
        return super().forward(snapshot["batch"], "Users")

class RouteAttentionRelGNN(OfficialRelGNN):
    """Official RelGNN with a task-specific residual correction to route sums."""
    def __init__(self,snapshot,channels=32,heads=2):
        super().__init__(snapshot,channels,heads,layers=1)
        routes=snapshot["routes"]
        self.route_logits=nn.Parameter(torch.zeros(len(EVENTS),len(routes)))
        # Start exactly at the official sum-aggregation behavior. Training can
        # learn a per-task correction toward attention-weighted route messages.
        self.route_gates=nn.Parameter(torch.zeros(len(EVENTS)))
    def forward(self,snapshot):
        batch=snapshot["batch"]
        seed_time=batch["Users"].seed_time
        x=self.encoder(batch.tf_dict)
        rel_time=self.temporal_encoder(seed_time,batch.time_dict,batch.batch_dict)
        for node_type,value in rel_time.items():
            x[node_type]=x[node_type]+value
        conv_layer=self.gnn.convs[0]
        # Calculate each official RelGNN route convolution once. The official
        # Users representation sums these messages before norm + ReLU.
        user_route_ids=[]
        user_messages=[]
        for rid,edge_type in enumerate(snapshot["routes"]):
            conv=conv_layer.convs[edge_type]
            kind=edge_type[0]
            if kind=="dim-dim":
                src,rel,dst=edge_type[1:]
                out=conv((x[src],x[dst]),batch.edge_index_dict[(src,rel,dst)])
                if out is None: continue
                updates=[(dst,out)]
            else:
                src,rel,dst=edge_type[1:4]
                src2,rel2,dst2=edge_type[4:]
                out=conv((x[src2],x[src],x[dst]),
                         (batch.edge_index_dict[(src,rel,dst)],batch.edge_index_dict[(src2,rel2,dst2)]))
                if out is None: continue
                # Keep both updates even if src == dst, matching the official
                # HeteroConv behavior that aggregates both messages.
                updates=[(dst,out[0]),(src,out[1])]
            for node_type,message in updates:
                if node_type == "Users":
                    user_route_ids.append(rid)
                    user_messages.append(message)
        if not user_route_ids:
            raise RuntimeError("The current schema routes produced no messages to Users")
        route_ids=torch.tensor(
            user_route_ids,
            dtype=torch.long,device=self.route_logits.device,
        )
        official_sum=torch.stack(user_messages,dim=0).sum(dim=0)
        official_user_h=self.gnn.norms[0]["Users"](official_sum).relu()

        task_outputs=[]
        for task_id in range(len(EVENTS)):
            weights=torch.softmax(self.route_logits[task_id,route_ids],dim=0)
            attended_sum=torch.stack(
                [weight*message for weight,message in zip(weights,user_messages)],dim=0
            ).sum(dim=0)
            attended_user_h=self.gnn.norms[0]["Users"](attended_sum).relu()
            # A zero-initialized gate keeps the initial representation equal
            # to official RelGNN. Each activity learns its own correction.
            gate=torch.tanh(self.route_gates[task_id])
            user_h=official_user_h+gate*(attended_user_h-official_user_h)
            task_outputs.append(self.head(user_h)[:, task_id:task_id+1])
        return torch.cat(task_outputs,dim=1)

def make_model(snapshot,route_attention=False,channels=32,heads=2):
    model_cls=RouteAttentionRelGNN if route_attention else OfficialRelGNN
    return model_cls(snapshot,channels=channels,heads=heads).to(DEVICE)
