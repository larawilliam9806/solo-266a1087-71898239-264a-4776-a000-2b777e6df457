"""示踪化验核对 API 测试：整组提交/换交、待核待补、相容共同分配、
冲突极小单组、补单回执、无关批次登记、失败保留原组。"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from app.main import app
    with TestClient(app) as c:
        yield c


def register(client, docs, expect=201):
    resp = client.post("/api/documents", json={"documents": docs})
    assert resp.status_code == expect, resp.text
    return resp.json() if expect == 201 else resp


def incoming(batch, weight):
    return {"type": "incoming", "batch": batch, "weight": weight}


def process(inputs, outputs):
    return {"type": "process",
            "inputs": [{"batch": b, "weight": w} for b, w in inputs],
            "outputs": [{"batch": b, "weight": w} for b, w in outputs]}


def sheet(no, batch, grams):
    return {"sheet_no": no, "batch": batch, "grams": grams}


def submit(client, tracer, sheets, expect=200):
    resp = client.post("/api/lab-tests",
                       json={"tracer_batch": tracer, "sheets": sheets})
    assert resp.status_code == expect, resp.text
    return resp.json() if expect == 200 else resp


def get_group(client, expect=200):
    resp = client.get("/api/lab-tests")
    assert resp.status_code == expect, resp.text
    return resp.json() if expect == 200 else resp


def check_allocation(allocation, weights, tracer, checked):
    """共同分配必须：示踪批全部克数进入链路、不超各批称重、逐工序守恒、
    每张被核单的被测批克数等于实测值。"""
    bg = allocation["batch_grams"]
    assert bg[tracer] == weights[tracer]
    for b, g in bg.items():
        assert 0 <= g <= weights[b], (b, g)
    for no, batch, grams in checked:
        assert bg.get(batch, -1) == grams, (no, batch)
    for p in allocation["processes"]:
        assert sum(p["input_grams"].values()) == sum(p["output_grams"].values())
        for b, g in list(p["input_grams"].items()) + list(p["output_grams"].items()):
            assert 0 <= g <= weights[b], (b, g)


# 混炼分装：T(1000,示踪) + X(500) --P1--> B(900) + C(600)
MIX_DOCS = [
    incoming("T", 1000), incoming("X", 500),
    process([("T", 1000), ("X", 500)], [("B", 900), ("C", 600)]),
]
MIX_WEIGHTS = {"T": 1000, "X": 500, "B": 900, "C": 600}


def test_get_group_404_before_submit(client):
    assert get_group(client, expect=404).status_code == 404


def test_consistent_group_and_allocation(client):
    register(client, MIX_DOCS)
    body = submit(client, "T", [sheet("H1", "B", 900)])
    assert body["applied"] and body["replaced"] is False
    con = body["conclusion"]
    assert con["status"] == "consistent"
    assert con["checked_sheets"] == ["H1"]
    assert con["pending_sheets"] == [] and con["missing_batches"] == []
    assert con["conflict_sheets"] == []
    check_allocation(con["allocation"], MIX_WEIGHTS, "T", [("H1", "B", 900)])
    bg = con["allocation"]["batch_grams"]
    assert bg["C"] == 100  # 示踪料其余 100 克只能落在 C

    got = get_group(client)
    assert got["tracer_batch"] == "T" and got["tracer_weight"] == 1000
    assert got["sheets"] == [{"sheet_no": "H1", "batch": "B", "grams": 900}]
    assert got["conclusion"]["status"] == "consistent"


def test_overcount_conflict_minimal_core(client):
    register(client, MIX_DOCS)
    # 两张单各自对得上称重，合起来 1500 克超过进料 1000 克
    con = submit(client, "T", [sheet("H1", "B", 900),
                               sheet("H2", "C", 600)])["conclusion"]
    assert con["status"] == "conflict"
    assert con["allocation"] is None
    assert con["conflict_sheets"] == ["H1", "H2"]
    # 移除其中任一张，剩下的即相容
    con1 = submit(client, "T", [sheet("H2", "C", 600)])["conclusion"]
    assert con1["status"] == "consistent"
    check_allocation(con1["allocation"], MIX_WEIGHTS, "T", [("H2", "C", 600)])
    con2 = submit(client, "T", [sheet("H1", "B", 900)])["conclusion"]
    assert con2["status"] == "consistent"
    check_allocation(con2["allocation"], MIX_WEIGHTS, "T", [("H1", "B", 900)])


def test_measured_exceeds_weight_is_singleton_conflict(client):
    register(client, MIX_DOCS)
    con = submit(client, "T", [sheet("H1", "B", 901)])["conclusion"]
    assert con["status"] == "conflict"
    assert con["conflict_sheets"] == ["H1"]  # 单张即无法成立


def test_sheet_on_tracer_batch(client):
    register(client, MIX_DOCS)
    con = submit(client, "T", [sheet("H0", "T", 1000)])["conclusion"]
    assert con["status"] == "consistent"
    assert con["allocation"]["batch_grams"]["T"] == 1000
    con = submit(client, "T", [sheet("H0", "T", 999)])["conclusion"]
    assert con["status"] == "conflict" and con["conflict_sheets"] == ["H0"]


def test_sheet_on_unrelated_batch(client):
    register(client, MIX_DOCS + [incoming("Z", 50)])
    con = submit(client, "T", [sheet("HZ", "Z", 0)])["conclusion"]
    assert con["status"] == "consistent"
    assert con["allocation"]["batch_grams"]["Z"] == 0
    con = submit(client, "T", [sheet("HZ", "Z", 5)])["conclusion"]
    assert con["status"] == "conflict" and con["conflict_sheets"] == ["HZ"]


# 缺来源场景：M 被工序引用但来源未登记
#   T(1000) --P1--> B(600) + C(400)
#   C(400) + M(100) --P2--> S(500)
PENDING_DOCS = [
    incoming("T", 1000),
    process([("T", 1000)], [("B", 600), ("C", 400)]),
    process([("C", 400), ("M", 100)], [("S", 500)]),
]
PENDING_WEIGHTS = {"T": 1000, "B": 600, "C": 400, "M": 100, "S": 500}


def test_pending_then_supplement_turns_conflict(client):
    register(client, PENDING_DOCS)
    submit(client, "T", [sheet("S1", "B", 600), sheet("S2", "S", 500)])
    got = get_group(client)["conclusion"]
    assert got["pending_sheets"] == ["S2"]      # S 的上游 M 缺来源
    assert got["missing_batches"] == ["M"]
    assert got["checked_sheets"] == ["S1"]      # 其余单照常核对
    assert got["status"] == "consistent"

    # 合规补单：即使整组转为冲突也照常入账
    resp = register(client, [incoming("M", 100)])
    review = resp["lab_review"]
    assert review["resolved_sheets"] == ["S2"]  # 本次补齐来源的化验单
    con = review["conclusion"]                  # 整组新结论
    assert con["status"] == "conflict"
    assert con["pending_sheets"] == []
    assert con["conflict_sheets"] == ["S2"]     # S 至多含 400 克示踪料
    assert get_group(client)["conclusion"]["status"] == "conflict"


def test_supplement_resolving_to_consistent(client):
    register(client, PENDING_DOCS)
    submit(client, "T", [sheet("S2", "S", 400)])
    assert get_group(client)["conclusion"]["pending_sheets"] == ["S2"]
    resp = register(client, [incoming("M", 100)])
    review = resp["lab_review"]
    assert review["resolved_sheets"] == ["S2"]
    con = review["conclusion"]
    assert con["status"] == "consistent"
    check_allocation(con["allocation"], PENDING_WEIGHTS, "T", [("S2", "S", 400)])


def test_unrelated_registration_while_pending(client):
    register(client, PENDING_DOCS)
    submit(client, "T", [sheet("S2", "S", 400)])
    # 核对未完成时，无关批次的登记照常办完
    resp = register(client, [incoming("Z", 50), process([("Z", 50)], [("Z1", 50)])])
    assert resp["lab_review"]["resolved_sheets"] == []
    assert resp["lab_review"]["conclusion"]["pending_sheets"] == ["S2"]
    got = get_group(client)["conclusion"]
    assert got["pending_sheets"] == ["S2"] and got["missing_batches"] == ["M"]


def test_no_lab_review_without_group(client):
    resp = register(client, [incoming("T", 100)])
    assert "lab_review" not in resp


def test_resubmit_replaces_and_failed_resubmit_preserves(client):
    register(client, MIX_DOCS)
    submit(client, "T", [sheet("H1", "B", 900)])
    body = submit(client, "T", [sheet("H2", "C", 600)])
    assert body["replaced"] is True
    assert [s["sheet_no"] for s in get_group(client)["sheets"]] == ["H2"]

    # 换交失败：原单据与结论保留
    submit(client, "T", [sheet("H3", "NOPE", 1)], expect=422)   # 被测批未登记
    submit(client, "NOPE", [sheet("H3", "B", 1)], expect=404)   # 示踪批不存在
    submit(client, "B", [sheet("H3", "B", 1)], expect=400)      # 示踪批非进料
    got = get_group(client)
    assert [s["sheet_no"] for s in got["sheets"]] == ["H2"]
    assert got["conclusion"]["status"] == "consistent"


def test_submit_validation(client):
    register(client, MIX_DOCS)
    submit(client, "T", [], expect=422)                                  # 空组
    submit(client, "T", [sheet("H1", "B", 1), sheet("H1", "C", 2)],
         expect=422)                                                     # 单号重复
    submit(client, "T", [sheet("H1", "B", -1)], expect=422)              # 克数为负
    submit(client, "T", [sheet("H1", "B", 1.5)], expect=422)             # 非整数
    assert get_group(client, expect=404).status_code == 404              # 均未落库


def test_same_batch_two_sheets(client):
    register(client, MIX_DOCS)
    con = submit(client, "T", [sheet("H1", "B", 900),
                               sheet("H2", "B", 900)])["conclusion"]
    assert con["status"] == "consistent"          # 同批同读数可同时成立
    con = submit(client, "T", [sheet("H1", "B", 900),
                               sheet("H2", "B", 800)])["conclusion"]
    assert con["status"] == "conflict"            # 同批读数不一致
    assert con["conflict_sheets"] == ["H1", "H2"]


def test_recycle_chain_conflict_core(client):
    # 回炉场景：T(1000)+X(500)->B(900)+C(600)；C->S1(400)+R(200)；R+Y(300)->S2(500)
    register(client, [
        incoming("T", 1000), incoming("X", 500), incoming("Y", 300),
        process([("T", 1000), ("X", 500)], [("B", 900), ("C", 600)]),
        process([("C", 600)], [("S1", 400), ("R", 200)]),
        process([("R", 200), ("Y", 300)], [("S2", 500)]),
    ])
    weights = {"T": 1000, "X": 500, "Y": 300, "B": 900, "C": 600,
               "S1": 400, "R": 200, "S2": 500}
    con = submit(client, "T", [sheet("D1", "B", 900), sheet("D2", "S1", 400),
                               sheet("D3", "S2", 200)])["conclusion"]
    # B=900 时 C 只余 100，S2 却报 200（须 R=200>C 中全部示踪料）
    assert con["status"] == "conflict"
    assert con["conflict_sheets"] == ["D1", "D3"]
    # 移除核心中任一张后均相容
    con = submit(client, "T", [sheet("D1", "B", 900)])["conclusion"]
    assert con["status"] == "consistent"
    check_allocation(con["allocation"], weights, "T", [("D1", "B", 900)])
    con = submit(client, "T", [sheet("D3", "S2", 200)])["conclusion"]
    assert con["status"] == "consistent"
    check_allocation(con["allocation"], weights, "T", [("D3", "S2", 200)])
    # 其余两张也可同时成立（C 全部示踪料经 S1、R 入 S2）
    con = submit(client, "T", [sheet("D2", "S1", 400), sheet("D3", "S2", 200)])["conclusion"]
    assert con["status"] == "consistent"
    check_allocation(con["allocation"], weights, "T",
                     [("D2", "S1", 400), ("D3", "S2", 200)])


def test_conclusion_snapshot_consistent_with_documents(client):
    # 换交后登记新单据：查询看到的化验单、料流与结论来自同一已入账状态
    register(client, MIX_DOCS)
    submit(client, "T", [sheet("H1", "B", 900), sheet("H2", "C", 600)])
    assert get_group(client)["conclusion"]["status"] == "conflict"
    register(client, [incoming("Q", 10)])  # 无关登记不改变化验结论
    got = get_group(client)
    assert got["conclusion"]["status"] == "conflict"
    assert got["conclusion"]["conflict_sheets"] == ["H1", "H2"]
    assert [s["sheet_no"] for s in got["sheets"]] == ["H1", "H2"]
