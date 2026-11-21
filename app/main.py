"""再生塑料发货追溯 API。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .db import connect, init_db
from .labtest import compute_conclusion, load_group
from .models import LabGroupRequest, RegisterRequest
from .registry import ConflictError, register_documents
from .state import load_state
from .trace import build_trace


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(
    title="再生塑料发货追溯 API",
    version="1.1.0",
    description="登记进料批、工序投入产出批与整批发货；按不合格进料批追溯各已发批的可能含量；"
                "示踪化验核对：整组化验单与账簿料流的联合一致性判定。",
    lifespan=lifespan,
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/documents", status_code=201)
def post_documents(req: RegisterRequest):
    """登记单据（进料/工序/发货）。同一请求内多张单据原子生效：
    批号或重量冲突、形成循环、保存失败时整次不生效。
    已提交化验单组时，回执附本次补齐来源的化验单号与整组新结论；
    化验核对不影响登记入账。"""
    try:
        return register_documents(req.documents)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/lab-tests")
def post_lab_tests(req: LabGroupRequest):
    """整组提交（或换交）示踪化验单：一份账只保留一组，整组原子替换。
    校验或写入失败时保留原单据与结论；核对结论随提交一并返回。"""
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        state = load_state(conn)
        tracer = req.tracer_batch
        if tracer not in state.weights:
            raise HTTPException(status_code=404, detail=f"批号 {tracer} 不存在")
        if tracer not in state.incoming:
            raise HTTPException(status_code=400, detail=f"批号 {tracer} 不是进料批")
        unknown = sorted({s.batch for s in req.sheets} - set(state.weights))
        if unknown:
            raise HTTPException(status_code=422,
                                detail=f"被测批号未登记：{', '.join(unknown)}")

        replaced = load_group(conn) is not None
        conn.execute("DELETE FROM lab_sheets")
        conn.execute("DELETE FROM lab_group")
        conn.execute("INSERT INTO lab_group(id, tracer_batch) VALUES (1, ?)", (tracer,))
        for s in req.sheets:
            conn.execute(
                "INSERT INTO lab_sheets(sheet_no, batch_no, grams) VALUES (?, ?, ?)",
                (s.sheet_no, s.batch, s.grams))
        group = {
            "tracer_batch": tracer,
            "sheets": [{"sheet_no": s.sheet_no, "batch": s.batch, "grams": s.grams}
                       for s in req.sheets],
        }
        conclusion = compute_conclusion(state, group)
        conn.commit()
        return {
            "applied": True,
            "replaced": replaced,
            "tracer_batch": tracer,
            "sheet_count": len(req.sheets),
            "conclusion": conclusion,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@app.get("/api/lab-tests")
def get_lab_tests():
    """查询当前化验单组与核对结论；化验单、料流与结论取自同一个已入账状态。"""
    conn = connect()
    try:
        conn.execute("BEGIN")
        state = load_state(conn)
        group = load_group(conn)
        if group is None:
            raise HTTPException(status_code=404, detail="尚未提交化验单组")
        conclusion = compute_conclusion(state, group)
        conn.commit()
        return {
            "tracer_batch": group["tracer_batch"],
            "tracer_weight": state.weights[group["tracer_batch"]],
            "sheets": group["sheets"],
            "conclusion": conclusion,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

@app.get("/api/trace/{batch_no}")
def get_trace(batch_no: str):
    """质检查询：输入不合格进料批号，返回各已发批可能含该料的
    最少/最多克数（各附一份符合整条链称重记录的克数分配）；
    上游有缺单的已发批返回待补批号。"""
    conn = connect()
    try:
        state = load_state(conn)
    finally:
        conn.close()
    if batch_no not in state.weights:
        raise HTTPException(status_code=404, detail=f"批号 {batch_no} 不存在")
    if batch_no not in state.incoming:
        raise HTTPException(status_code=400, detail=f"批号 {batch_no} 不是进料批")
    return {
        "defective_batch": batch_no,
        "defective_weight": state.weights[batch_no],
        "shipments": build_trace(state, batch_no),
    }


@app.get("/api/batches")
def list_batches():
    conn = connect()
    try:
        state = load_state(conn)
    finally:
        conn.close()
    return {"batches": [{"batch_no": b, "weight": w}
                        for b, w in sorted(state.weights.items())]}


@app.get("/api/batches/{batch_no}")
def get_batch(batch_no: str):
    conn = connect()
    try:
        state = load_state(conn)
    finally:
        conn.close()
    if batch_no not in state.weights:
        raise HTTPException(status_code=404, detail=f"批号 {batch_no} 不存在")
    src_proc = state.source_process()
    dest_proc = state.dest_process()
    if batch_no in state.incoming:
        source = {"type": "incoming"}
    elif batch_no in src_proc:
        source = {"type": "process", "process_id": src_proc[batch_no]}
    else:
        source = {"type": "missing"}
    if batch_no in dest_proc:
        destination = {"type": "process", "process_id": dest_proc[batch_no]}
    elif batch_no in state.shipments:
        destination = {"type": "shipment", "shipment_id": state.shipments[batch_no]}
    else:
        destination = {"type": "none"}
    return {
        "batch_no": batch_no,
        "weight": state.weights[batch_no],
        "source": source,
        "destination": destination,
    }


@app.get("/api/processes/{process_id}")
def get_process(process_id: int):
    conn = connect()
    try:
        state = load_state(conn)
    finally:
        conn.close()
    proc = state.processes.get(process_id)
    if proc is None:
        raise HTTPException(status_code=404, detail=f"工序 {process_id} 不存在")
    return {
        "process_id": process_id,
        "inputs": [{"batch_no": b, "weight": state.weights[b]} for b in proc["inputs"]],
        "outputs": [{"batch_no": b, "weight": state.weights[b]} for b in proc["outputs"]],
    }


@app.get("/api/shipments")
def list_shipments():
    conn = connect()
    try:
        state = load_state(conn)
    finally:
        conn.close()
    return {"shipments": [{"shipment_id": sid, "batch_no": b, "weight": state.weights[b]}
                          for b, sid in sorted(state.shipments.items())]}
