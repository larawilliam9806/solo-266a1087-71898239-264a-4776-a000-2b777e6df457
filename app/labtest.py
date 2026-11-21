"""示踪化验核对：一组化验单与账簿料流的联合一致性判定。

模型：示踪进料批 T 的全部克数（= 其称重）作为"该料"进入链路流动——
批号容量 = 称重克数，工序投入 = 产出守恒，无去向的批为汇。一张化验单
（被测批 b、实测 g 克）即硬约束 x_b = g：

- 被测批上游缺来源（既非进料也非任何工序产出）的单列为待核，不参与本次判定；
- 其余单须能被同一份符合批重和工序守恒的整数克分配同时解释。判定归约为
  "带下界的可行循环流"（下界变换 + 最大流），账簿无环且容量皆整数，
  有可行解即有一组整数克解；
- 相容：返回一份共同分配（逐批克数 + 逐工序投入/产出克数），可逐道核对；
- 冲突：返回一小组无法同时成立的单号——移除其中任一张，剩下的即相容
  （删除法求包含极小的不可行子集）。

本模块只读账簿状态，不修改任何数据。
"""
from collections import defaultdict

from .flow import INF, add_edge, max_flow


def load_group(conn):
    """读取当前化验单组（一份账只保留一组）；尚未提交返回 None。"""
    row = conn.execute("SELECT tracer_batch FROM lab_group WHERE id = 1").fetchone()
    if row is None:
        return None
    sheets = [
        {"sheet_no": r["sheet_no"], "batch": r["batch_no"], "grams": r["grams"]}
        for r in conn.execute(
            "SELECT sheet_no, batch_no, grams FROM lab_sheets ORDER BY rowid")
    ]
    return {"tracer_batch": row["tracer_batch"], "sheets": sheets}


def _missing_sources(state, batch):
    """batch 上游缺来源的批号（既非进料也非任何工序产出）。"""
    src_proc = state.source_process()
    missing, seen, stack = [], set(), [batch]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x in state.incoming:
            continue
        pid = src_proc.get(x)
        if pid is None:
            missing.append(x)
        else:
            stack.extend(state.processes[pid]["inputs"])
    return missing


def pending_sheets(state, sheets):
    """返回 (待核单号列表, 待补批号列表)：被测批上游缺来源的化验单。"""
    pend, missing = [], set()
    for s in sheets:
        m = _missing_sources(state, s["batch"])
        if m:
            pend.append(s["sheet_no"])
            missing.update(m)
    return pend, sorted(missing)


def _circulation(state, tracer, fixed):
    """求一份符合批重与工序守恒、且满足 fixed 硬约束的整数克分配。

    fixed: {批号: 该料克数}。返回 {批号: 克数}（覆盖账簿全部批号）；
    无可行分配返回 None。
    """
    weights = state.weights
    for b, g in fixed.items():
        if b not in weights or g > weights[b]:  # 实测超过批重，任何分配都解释不了
            return None

    node_id = {}

    def nid(key):
        if key not in node_id:
            node_id[key] = len(node_id) + 4  # 0=SS 1=TT 2=S 3=SNK
        return node_id[key]

    for b in weights:
        nid(("in", b))
        nid(("out", b))
    for pid in state.processes:
        nid(("proc", pid))
    g = [[] for _ in range(len(node_id) + 4)]
    SS, TT, S, SNK = 0, 1, 2, 3

    dest_proc = state.dest_process()
    demand = defaultdict(int)  # 下界变换：b(v) = 入边下界和 - 出边下界和
    handles = {}               # batch -> (u, 边下标, 容量上界)

    def edge(u, v, lo, hi):
        add_edge(g, u, v, hi - lo, 0)
        demand[v] += lo
        demand[u] -= lo
        return (u, len(g[u]) - 1, hi)

    # 示踪进料批的全部克数进入链路
    edge(S, nid(("in", tracer)), weights[tracer], weights[tracer])
    for b, w in weights.items():
        lo = fixed.get(b, 0)
        hi = fixed[b] if b in fixed else w
        handles[b] = edge(nid(("in", b)), nid(("out", b)), lo, hi)
    for pid, p in state.processes.items():
        pn = nid(("proc", pid))
        for b in p["inputs"]:
            edge(nid(("out", b)), pn, 0, INF)
        for b in p["outputs"]:
            edge(pn, nid(("in", b)), 0, INF)
    for b in weights:  # 无去向（未投料或已发货）的批为汇
        if b not in dest_proc:
            edge(nid(("out", b)), SNK, 0, INF)
    edge(SNK, S, 0, INF)  # 汇回流成循环流

    total = 0
    for v, d in demand.items():
        if d > 0:
            add_edge(g, SS, v, d, 0)
            total += d
        elif d < 0:
            add_edge(g, v, TT, -d, 0)
    if max_flow(g, SS, TT) != total:
        return None
    # 边上实际流量 = 上界 - 剩余容量（已含下界）
    return {b: hi - g[u][i].cap for b, (u, i, hi) in handles.items()}


def _feasible(state, tracer, sheets):
    """一组化验单（子集）能否被同一份分配解释；能则返回该分配，否则 None。"""
    fixed = {}
    for s in sheets:
        b, grams = s["batch"], s["grams"]
        if b in fixed and fixed[b] != grams:  # 同批两张单读数不一致
            return None
        fixed[b] = grams
    return _circulation(state, tracer, fixed)


def _cone(state, tracer):
    """示踪批沿去向可达的批号与工序（影响面）。"""
    dest_proc = state.dest_process()
    batches, procs, stack = set(), set(), [tracer]
    while stack:
        b = stack.pop()
        if b in batches:
            continue
        batches.add(b)
        pid = dest_proc.get(b)
        if pid is not None:
            procs.add(pid)
            stack.extend(state.processes[pid]["outputs"])
    return batches, procs


def _make_allocation(state, tracer, checked, grams):
    """把流量解整理成可逐道核对的共同分配（覆盖影响面与被测批）。"""
    cone_b, cone_p = _cone(state, tracer)
    src_proc = state.source_process()
    dest_proc = state.dest_process()
    batches, procs = set(cone_b), set(cone_p)
    for s in checked:
        b = s["batch"]
        batches.add(b)
        if b in src_proc:
            procs.add(src_proc[b])
        if b in dest_proc:
            procs.add(dest_proc[b])
    return {
        "batch_grams": {b: grams.get(b, 0) for b in sorted(batches)},
        "processes": [
            {
                "process_id": pid,
                "input_grams": {b: grams.get(b, 0) for b in state.processes[pid]["inputs"]},
                "output_grams": {b: grams.get(b, 0) for b in state.processes[pid]["outputs"]},
            }
            for pid in sorted(procs)
        ],
    }


def compute_conclusion(state, group):
    """对当前账簿状态核对整组化验单，返回结论（不修改任何数据）。

    group: {"tracer_batch": ..., "sheets": [{"sheet_no", "batch", "grams"}, ...]}
    """
    tracer = group["tracer_batch"]
    sheets = group["sheets"]
    pend_nos, missing = pending_sheets(state, sheets)
    pend_set = set(pend_nos)
    checked = [s for s in sheets if s["sheet_no"] not in pend_set]
    checked_nos = sorted(s["sheet_no"] for s in checked)

    alloc = _feasible(state, tracer, checked)
    if alloc is not None:
        return {
            "status": "consistent",
            "checked_sheets": checked_nos,
            "pending_sheets": sorted(pend_nos),
            "missing_batches": missing,
            "allocation": _make_allocation(state, tracer, checked, alloc),
            "conflict_sheets": [],
        }

    # 删除法求包含极小的不可行子集：移除其中任一张后，剩下的即相容
    core = list(checked)
    for s in checked:
        trial = [x for x in core if x["sheet_no"] != s["sheet_no"]]
        if len(trial) < len(core) and _feasible(state, tracer, trial) is None:
            core = trial
    return {
        "status": "conflict",
        "checked_sheets": checked_nos,
        "pending_sheets": sorted(pend_nos),
        "missing_batches": missing,
        "allocation": None,
        "conflict_sheets": sorted(s["sheet_no"] for s in core),
    }
