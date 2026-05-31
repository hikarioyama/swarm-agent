# step37-harness — 設計判断と現状 (DESIGN)

Step-3.7-Flash を 2× RTX PRO 6000 で「数十〜数百エージェント並列」で効率的に動かす
ハーネスの、**現状**と**各判断の意図**をまとめる。数値はすべて実測由来
(`~/bench/step37-mtp/FLEET_OPTIMUM.md` + ライブ計測)。

---

## 0. 解くべき問題

- 単発の Step-3.7 は **~125 tok/s**。GPU を活かすには**数十体を同時 in-flight** にするしかない
  (1並列だと能力の ~90% を捨てる)。
- 本質的な難しさは**スループットでなく「多数の並列エージェントを効率的に統治すること」**。
  - ただのメイン-サブはメインがパンクする(context爆発・conc=1のdecodeが律速・N状態の同時把握)。
  - エージェントチーム模倣も違う(N²通信・共有state=並列の敵)。

---

## 1. 機材とモデルの実像(実測)

| 項目 | 値 | 備考 |
|---|---|---|
| GPU | 2× RTX PRO 6000 Blackwell, 96GB×2 = 192GB | TP=2 |
| モデル | StepFun Step-3.7-Flash, 198B MoE, **NVFP4 ~116GB** | 重みがVRAMの大半 |
| KVキャッシュ | **1,625,950 tokens** (fp8, ~41GB) | gpu-util 0.92 |
| 投機デコード | **MTP K=1**, acceptance ~0.79 | 全並列で +14〜44% |

**KVが「軽い」理由(重要)**: Step-3.7 は**ハイブリッドアテンション** — 45層中
**フルアテンションは12層だけ、残り33層は sliding-window**。よってフルKVを溜めるのは12層分で
**~24KB/token**。普通の45層フルGQAなら ~90KB/token(3.7倍重い)で ~460K tokens しか入らない。
→ **KVは非効率なのでなく、モデルが巨大(116GB)なだけ**。1.6M は妥当。増やすなら VRAM を空ける
(gpu-util↑ / ディスプレイを RX 9070 XT へオフロード)しかない。

---

## 2. 測定で確定した操業点 (FLEET_OPTIMUM.md)

- **worker(~8K context)の効率 in-flight 領域 = C32–C64**:
  C32=763 tok/s (23.8 tok/s/agent, ~6×single) / **C64=1225 tok/s (~9.8×single)**。
  C16→C64 まで Welch検定で有意上昇。C96+ は固定窓で ramp 汚染のため**範囲外**(B* は 64 で頭打ち扱い)。
- **worker context は ~8K が最適**: 16K は ~18%遅く KV も倍速で埋まり C32 で頭打ち、32K は ~C8 で崩壊。
  → **context を小さく保つほど効率が上がる**(成長したら compaction で抑える)。
- **MTP は K=1 が最良**(K1>K2>K3、高Kは位置別acceptance低下で利得なし)。
- サーバ設定は**現状が最適、変更不要**。

> 注: 計測は7回・Codexレビュー6巡を経た保守的な床。prefix-cache OFF(worker毎ユニークctx)・
> churn谷込みの「最悪ケース」なので、実fleet(共通prefix+cache)はもっと出る。
>
> **v0.2 で実エージェント経路を実測 → §6 の表**。合成の上記(httpx probe)を上回り、N=32 で 1247 tok/s。
> 長い実出力では **knee は ~C32**(C48 は flat)— 上記合成(C64まで上昇)より早い。律速は KV でなく計算/DecodeGate。

---

## 3. アーキテクチャ判断と意図

### 3.1 スティグマジー協調(board経由)— 指揮者を置かない
**意図**: 賢いメインを hot loop から降ろす。Amdahl — 監督が直列で10%触るだけで上限10倍。
- worker は互いにも中央にも喋らない。共有 **Board**(DAGキュー)で task claim → 結果書込 → 依存解錠。
- 協調は「メッセージ」でなく「キュー+依存状態」から創発 → メインの context爆発・直列監督が消える。
- 実装: `fleet/board.py`(状態 pending/ready/running/done/failed, 依存, 上流結果注入)。

### 3.2 知能ゼロの高速ディスパッチャ — N体を回すのは while ループ
**意図**: GPUを満たし続けるのは admission-control コード(LLM不要)であって、エージェントではない。
- 実装: `fleet/scheduler.py`。TARGET_INFLIGHT 体を常時稼働、ready taskを引いて process pool へ投入。

### 3.3 lean・prefix安定 worker — **最大レバー(検証済)**
**意図**: 全workerに39 tools+全MCPは巨大な冗長prefill。役割別に最小ツールだけ積む。
- 実測: デフォルト 39 tools = **14,113 tok**(同一スキーマ)を ~40体が**毎回**再prefill、fleet prefix-cache **hit 0%**
  (worker毎の context-files/memory が system block の整列を壊すため)。
- 修正(`AIAgent(enabled_toolsets=…, skip_context_files=True, skip_memory=True)`):
  coder[file,terminal,search]=3,328 tok(-77%)/ researcher[web,search]=398(-97%)/ reducer[]=最小。
  同役割が**byte一致の最小prefix**を共有 → vLLM の auto prefix-cache が worker#1 以降ヒット → **実効 -98%**。
- これが**最大の即効レバー**: config 変更だけ(HermesAgent無改変、AIAgentが全引数を既に受ける)。
- 実装済: `fleet/config.py` TOOL_PROFILES + `fleet/worker.py`。

### 3.4 ヘテロなロスター(KVポートフォリオ)— 全員同一contextにしない
**意図**: 役割ごとに必要なcontextが違う。KVは共有予算(1.6M tokens)。少数の大context役を予約し、
worker レーンを伸縮basin にする。**「全員同じctxが最適」は誤り**(ユーザ指摘どおり)。
- 単一context曲線(1K/8K/32K)は各レーンの**材料**、fleet最適はその**混合**。
- 実装: `fleet/config.py` ROSTER + `fleet/roster.py`(KV予算チェック)。詳細は §4。

### 3.5 使い捨て狭context worker — state は board に
**意図**: 短ctx = KV予算に収まる + スループット曲線の有利側。state を worker の context に溜めない。

### 3.6 admission を decode-batch に閉じる + duty oversubscription(設計、未実装)
**意図**: in-flight ≠ decoding。ツール待ち中の worker は GPU を使ってない。
- ~40体を**常時 decode** させるには enrolled = B*/duty で**過剰登録**(duty 0.4 なら ~120体登録で ~48 decode)。
- /metrics(num_requests_running, kv_cache_usage_perc, num_preemptions_total)で B* を閉ループ制御、
  KV>85%/preempt増 で AIMD バックオフ。
- **検証注意**: 機構は正しいが「+60-90%」は**過大判定**。実 duty は未測(light tools で ~0.88、
  browser/code 多用で大きく低下)。実 duty 測定が精度の鍵。

### 3.7 worker context は ~8K に束ねる
**意図**: §2 のとおり 16K/32K は効率が落ちる。多段で成長したら summarize/compaction で 8-16K に抑える。

---

## 4. 暫定ロスター(現状の構成)

KV予算 1,625,950 tokens に対する配分(`python -m fleet.roster` で再現):

| 役割 | context | 数 | tools | duty | 役目 |
|---|---|---|---|---|---|
| **director** | 128K | **1** | [todo] | 0.15 | 長期方針の舵取り。goal+plan+state 保持、**board経由**、hot loop非搭乗。ユーザの対話相手 |
| planner | 32K | 2 | [todo] | 0.5 | goal→DAG 分解。バースト |
| reducer | 16K | 6 | [] | 0.7 | 木で集約、根に近いほど成長 |
| **worker** | 8K | **48**(in-flight) | [file,terminal,search] | 0.4 | **大群**。使い捨て・lean・C32-64操業点 |
| router | 2K | 16 | [] | 0.2 | 分類/振り分け、ほぼ無料 |

- KV: in-flight **44%** / enrolled(duty過剰登録時)**96%** = 収まる(ギリギリ)。
- **director 1体 128K = KV予算の8%**。duty 0.15 でほぼ decode しない → KVを握るだけで GPU帯域は食わない。
- worker の "48" は in-flight目標。実 duty 次第で enrolled を 60〜130 に調整して「常時~48 decode」維持。

**意図の要点**: GPUに当たるのは各役割の `count×duty` の和(=同時decode)。worker48体が大半を占め
**全体で実測knee(~48-64)を狙う**。director は**48 workerを直接監視しない**(それだとパンク)、
board のサマリを見て方針更新する=スティグマジー。

---

## 5. 現状の実装ステータス

**実装済(動作確認済)**:
- `board.py` — DAGキュー、依存、上流結果注入(reducerが実結果を集約することを実証)
- `scheduler.py` — ProcessPoolExecutor の admission ループ(固定 inflight)
- `worker.py` — 使い捨て AIAgent worker + **役割別 lean toolset**(prefill ~93%減を実測)
- `config.py` — 実測操業点 + TOOL_PROFILES + ROSTER
- `roster.py` — KV予算チェッカ
- `cli.py` — `python -m fleet.cli tasks.jsonl --inflight N`
- `plugin/` — HermesAgent の `/fleet` コマンド(`~/.hermes/plugins/` へ symlink、本家リポ無傷)

**v0.2 で実装＋live検証済(ロードマップ全消化)**:
| # | 項目 | 実装 | 状態 |
|---|---|---|---|
| 1 | lean・prefix安定 worker | `compat.make_agent` 役割別 toolset | ✅ 済(prefill -77〜-98%) |
| 2 | decode-batch admission + AIMD | `compat.DecodeGate`(可変 semaphore) + `admission.AIMDController` | ✅ gate がサーバ`num_requests_running`を N に pin(occupancy 0.92-1.0)、AIMD は knee(~C32)に収束 |
| 3 | worker が decode_s/tool_s を返し実 duty 計測 | forwarder monkeypatch + `worker.run_task_local` | ✅ 実測(59/41/28 tok/s/agent @ N16/32/48) |
| 4 | レーン別 admission(KVウォーターフォール) | `DecodeGate` lane-priority acquire(`config.LANE_PRIORITY`) | ✅ 高lane優先で worker swarm に埋もれない |
| 5 | ~~単一プロセス asyncio worker~~ → **単一プロセス ThreadPool** | `engine.ThreadFleet`(ThreadPoolExecutor) | ✅ recon訂正: HermesAgent は **sync+threaded**(asyncio経路なし)。per-instance は heap のみ |
| - | prefix warm(各役割の#1を事前に温める) | `warm.warm_profiles` | ✅ role prefix hit-rate 48%→97% |
| - | parking(ツール待ち中のKV退避) | **不要(自動)** | ✅ stateless full-history resend のためツール実行中は KV ゼロ。recon が parking 機構を解消 |
| - | SQLite 永続 board | `board.SqliteBoard` + `open_board` | ✅ atomic claim + liveness-gated restart recovery |

> v0.2 で追加: `compat.py`(DecodeGate + 無改変 monkeypatch)・`metrics.py`(/metrics + duty)・`admission.py`(AIMD)・
> `engine.py`(ThreadFleet)・`warm.py`・`board.py`(SqliteBoard)・`BUILD_SPEC.md`(recon接地の実装契約)。
> 6エージェント敵対的レビューが CRITICAL/MAJOR を検出 → 全修正(gate phantom-ticket leak、enrolled clamp 反転、
> kv の label 横断 sum、AIMD saturation/EWMA、board write-guard + reset liveness、sandbox 隔離)。
> hermes-agent リポは無改変(全適応は `compat.py` の runtime monkeypatch)。

---

## 6. 解決済みの問い + 残課題(v0.2 live 実測)

**v0.2 実測で解決(real HermesAgent 経路を DecodeGate 経由で windowed 30s 測定、`results/operating_point.json`)**:

| N | tok/s(実測) | mean_running | occupancy | tok/s/agent | KV% | 合成基準(FLEET_OPTIMUM §4) |
|---:|---:|---:|---:|---:|---:|---|
| 16 | **947** | 16.0 | 1.00 | 59.2 | 11.3 | 652 / 0.97 / 11% |
| 32 | **1247** | 30.24 | 0.945 | 41.2 | 19.3 | 763 / 0.96 / 20% |
| 48 | **1246** | 44.12 | 0.919 | 28.2 | 27.9 | 922 / 0.96 / 30% |

- **実 throughput / operating point**: 実エージェント経路で **C32-64 域を達成・合成を上回る**(N=32 で 1247 tok/s = 合成 C64 級)。
  **新知見: 長い実出力では throughput knee は ~C32**(1247→1246 で 32→48 はほぼ flat)— 短出力の合成曲線(C64)より早い。
- **実 duty cycle**: per-agent decode を実測(59/41/28 tok/s/agent)。「未測」を解消。
- **KV モデル検証**: 実測 KV%(11/19/28)が予測(11/20/30)とほぼ一致 → **gate が KV を bound** する主張を実証。
  stateless full-history resend のため **ツール実行中は KV ゼロ = parking 自動**(enrolled ≫ KV-resident)。
- **AIMD 収束(criterion D)**: gate 12 から additive 成長し knee(~C32-46)で sawtooth、KV<30% で安全、deadlock/thrash なし。

**残課題**:
- **C96+**: 固定窓 ramp 汚染で範囲外のまま(N比例 warm-up が必要)。DecodeGate MAX は 96。
- **skip_context_files/skip_memory の品質影響**: board が state を持つので安全な想定だが未定量。
- **gate-bypass の既知ギャップ**: iteration-limit summary 生成(`chat_completion_helpers.py:1433/1476`)は forwarder を
  経由せず ungated(review #9、defer)。no-tool single-turn では発火せず、稀・短い1生成のみ。
- **KVを増やす価値**: 測定域(C32-64)では KV は 28% までで律速でない(律速は DecodeGate/計算)。gpu-util↑/表示オフロードは別途。

---

## 7. 由来 (provenance)

- 操業点・スループット曲線・MTP・context regime: `~/bench/step37-mtp/FLEET_OPTIMUM.md`
  + 計測infra(`steady_probe.py`, `analyze_steady.py`, `fleet_sweep.py`)。Codexレビュー6巡。
- 効率5次元の分析(lean=最大レバー検証、duty controller、KVレーン等): efficiency workflow(9エージェント)。
- このハーネス: `~/projects/step37-harness/`(独立 git)。プラグインは `~/.hermes/plugins/` へ symlink。
