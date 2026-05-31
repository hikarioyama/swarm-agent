# hermes-swarm — 構想 (CONCEPT)

**スワーム前提で起動する、HermesAgent の fork。** 立ち上げた瞬間からスワームで、目標を
1つ投げれば planner が分解し、当たり前のように数十エージェントが同時に走り、GPU を飽和
させた最大スループットでタスクを片付ける。

これは **構想ドキュメント**（実装前）。すべての数値・事実は step37-harness で実測・接地
済み（§3, §4, §10）。前身の step37-harness は「HermesAgent 無改造で外から包む」設計だった
が、本構想は **fork して swarm をネイティブ化**する（ユーザ決定）。

---

## 0. 一行で

> `hermes-swarm "このリポを監査して脆弱性を全部出して"` → planner が 12 サブタスクに分解 →
> 40 体が同時 decode で実行 → reducer が統合レポートを返す。**指揮者なし・スワームが既定。**

---

## 1. ゴール体験（何が「当たり前」になるか）

- **起動 = スワーム。** 単一エージェントの対話ではなく、立ち上げ時に board + 数十の使い捨て
  worker レーン + decode-gate が既に立っている。「並列化する/しない」の選択は無い。常時スワーム。
- **目標駆動。** 高レベルの目標を1つ渡すと、planner エージェントが task DAG に自動分解し、
  worker スワームが並列実行、reducer が木で集約して最終回答を返す。DAG を手書きしない。
- **最大スループット既定。** 同時 decode 数は実測 knee（~C32、§4）を AIMD が自動追従。ユーザは
  チューニング不要で「数十体が回って速い」状態がデフォルト。
- **使い捨て・狭 context。** 各 worker は1サブタスクを処理して死ぬ。state は board に残る。
  context は ~8K に束ね、成長したら compaction で抑える（throughput 有利側を維持）。

前身ハーネスとの差：ハーネスは「`python -m fleet.cli tasks.jsonl` に DAG を手で渡す実行基盤」
だった。hermes-swarm は「**起動即スワーム + 目標を投げるだけ**」の本体に格上げする。

---

## 2. なぜ fork か（決定と根拠）

| 選択肢 | 評価 |
|---|---|
| 外部ハーネス（無改造 monkeypatch） | ← 前身。git 安全だが「外から包む」だけで swarm がネイティブにならない。hot-path の一部（iteration-summary 生成）に手が届かず gate を回避される（§7）。|
| hermes 本体を直接改造 | 完全ネイティブだが daily driver（Step3.7 hermes、本番）に upstream pull 衝突・破壊リスク。|
| **fork（採用）** | 本家と完全分離。daily hermes 無傷。swarm 専用に改造し放題。upstream は手動 merge。|

**fork だから初めて可能になること（§5.4）**：monkeypatch では危険で諦めた領域に正面から手を
入れられる — decode-gate を runtime のネイティブ admission 層に、session/sandbox 隔離を本実装に、
planner/reducer を first-class なスワーム role に、Step3.7 の冗長 reasoning 制御を既定に。

---

## 3. 接地済みの事実（recon — 設計を縛る5本柱）

5 エージェント recon で HermesAgent 内部を確定（`BUILD_SPEC.md` に line-cite 完全版）。これらは
swarm 設計の前提であり、fork でも変わらない物理：

1. **sync + threaded（asyncio ではない）。** LLM 呼び出しは同期 `openai.OpenAI`。各ターンは OS
   スレッドをブロックするが socket I/O 中に GIL を解放 → 1プロセス内で数十生成が重なる。
   → swarm worker は **スレッド**（コルーチンではない）。100-200 体 = heap のみ（subprocess 不要）。
2. **stateless full-history resend。** 毎ターン全履歴を再送、サーバ側セッション無し。サーバは
   **生成中のリクエストのぶんしか KV を保持しない** → ツール実行中の worker は KV ゼロ。
   → **「parking」は自動**。enrolled ≫ KV-resident が成立。KV 律速 ≈ 同時 decode 数。
3. **ツールは in-process、MCP は per-agent で湧かない。** per-instance コスト ≈ heap のみ。
4. **LLM 呼び出しの単一 chokepoint** = `AIAgent._interruptible_(streaming_)api_call`。ここを押さえれば
   全生成を gate + 計測できる（streaming は eager に消費しきって返る）。
5. **thread-per-worker の必須条件**（TS1-7）：worker ごとに unique session_id（衝突で sandbox/cwd
   共有事故）/ sandbox 隔離（terminal/file ツールは task_id を "default" に畳む）/ tool_delay=0 /
   非対話化 / cache prewarm / `_last_resolved_tool_names` の扱い。**fork では monkeypatch でなく本実装で解決。**

---

## 4. 実測済みの操業点（「最大スループット」の数値根拠）

step37-harness のエンジンを **実 HermesAgent 経路** で live 実測（DecodeGate 経由、windowed 30s、
`results/operating_point.json`）：

| 同時 in-flight N | tok/s | occupancy | tok/s/agent | KV% | 合成基準(FLEET_OPTIMUM §4) |
|---:|---:|---:|---:|---:|---|
| 16 | 947  | 1.00  | 59.2 | 11.3 | 652 / 11% |
| 32 | **1247** | 0.945 | 41.2 | 19.3 | 763 / 20% |
| 48 | 1246 | 0.919 | 28.2 | 27.9 | 922 / 30% |

- **gate がサーバ `num_requests_running` を N に pin**（occupancy 0.92-1.0）= スワームの同時 decode 数を
  正確に制御できることを実証。
- **実エージェント経路で合成 C32-64 域を達成・上回る**（N=32 で 1247 tok/s）。
- **長い実出力では throughput knee は ~C32**（32→48 でほぼ flat）。短出力合成の C64 より早い。
  → swarm の既定 decode 目標は **~32** に置くのが妥当（AIMD で追従）。
- **KV モデル検証**：実測 KV%（11/19/28）が予測（11/20/30）と一致 → §3-2「gate が KV を bound」を実証。
- **AIMD 収束**：gate 12 から knee(~C32-46)に sawtooth 収束、KV<30%、thrash/deadlock なし。
- 単発 Step3.7 ≈ 125 tok/s。スワームで **~10× の throughput** を「既定で」引き出す、が hermes-swarm の意義。

---

## 5. アーキテクチャ構想（swarm-native）

### 5.1 起動即スワーム（boot）
プロセス起動時に常設で立てる：
- **Board**（DAG キュー、状態 pending/ready/running/done、依存、上流結果注入）。stigmergy の基盤。
  永続は SQLite（restart 耐性、atomic claim、liveness-gated recovery — 実装＆検証済み）。
- **DecodeGate**（可変 semaphore、lane-priority）＝同時生成数 = サーバ KV を pin。
- **ThreadFleet**（単一プロセス bounded ThreadPool）＝enrolled worker を oversubscribe（既定
  oversub×3、enroll≤256）。GPU を満たし続けるのは LLM 不要の admission ループ。
- **AIMD controller**＝/metrics（running, kv, preemptions, gen_tokens, waiting）から gate を knee に追従。
- **prefix-warm**＝各 role の system+tools prefix を事前に温め、worker#1 もキャッシュヒット（48→97% 実測）。

### 5.2 目標駆動 front door（採用フロー）
```
goal(1行) ─▶ planner エージェント ─▶ task DAG (board へ add) ─▶ worker スワーム並列実行
                                                                      │
                          reducer（木で fan-in、根に近いほど大 context）◀┘ ─▶ 最終回答
```
- **planner（director レーン）**：goal を読み、依存付き task list（id/prompt/deps/lane）を生成して
  board に投入。バースト的・高 value。これが前身ハーネスに無かった「自動分解」の本体。
- **worker（大群）**：lean toolset、使い捨て、C32 操業点。
- **reducer**：上流結果を実集約（step37-harness で reducer が実結果を統合することは実証済み）。
- **router**：分類/振り分け、ほぼ無料。
- 協調は board の「キュー＋依存状態」から創発（メッセージ無し）。指揮者を hot loop に置かない
  （Amdahl：監督が直列で触ると上限が縛られる）。

### 5.3 decode-gate admission（ネイティブ化）
recon §3-2 の帰結を runtime の admission 層として本実装：worker が生成に入る前に gate を取得、
ツール実行中は permit を持たない。これで「常時 ~32 decode」を duty に依らず正確に維持（前身では
forwarder monkeypatch、fork ではエージェント runtime の一部に）。lane-priority で planner/reducer が
worker swarm に埋もれない（KV ウォーターフォール）。

### 5.4 fork だからできること（外部 monkeypatch では不可能だった点）
- **gate-bypass の根絶**：iteration-limit summary 生成（§7）も含め、全 LLM 呼び出しを1つの admission
  層に通す。monkeypatch では client/socket lifecycle（TS6）を壊す恐れで諦めた。
- **session/sandbox 隔離を本実装**：task_id を "default" に畳む挙動を直し、worker ごとに真に独立した
  cwd/bash/env（外部では register_task_env_overrides を後付け）。
- **planner/reducer を first-class role に**：role 別の system prompt・context 予算・toolset を
  ネイティブ定義（外部では AIAgent 引数で疑似的に）。
- **Step3.7 冗長 reasoning の既定制御**：`reasoning_effort` と出力境界を role ごとに設定（worker は
  簡潔、planner は熟考）。外部では env-gated な後付けだった。
- **SessionDB 直列化の解消**：共有 sqlite conn+lock（TS4）を per-thread WAL/pool に作り替え、120 体の
  永続書き込みボトルネックを除去。
- **単一プロセス効率の最大化**：model client・tool registry・prewarm を swarm 全体で共有する前提で再設計。

---

## 6. 再利用する資産（step37-harness、検証済み）

fork に移植できる実装（すべて live 動作確認済み）：
- `compat.DecodeGate` … 可変・lane-priority・interrupt-safe semaphore（敵対的レビューで phantom-ticket
  leak を修正済み）。
- `engine.ThreadFleet` … oversubscribe + enrolled clamp（ENROLL_MAX 外側 cap 修正済み）+ daemon metrics sampler。
- `admission.AIMDController` … saturation を gate in_flight から判定、EWMA + dwell + KV/waiting backoff。
- `metrics.py` … /metrics scrape（kv は label 横断 MAX、sum しない修正済み）+ duty/throughput 導出。
- `board.SqliteBoard` … atomic claim、liveness-gated restart recovery、busy-retry、WAL checkpoint。
- `warm.warm_profiles` … role prefix 事前温め。
- 計測 infra … `scripts/throughput_probe.py`（操業点再現）、`scripts/aimd_probe.py`（収束）、
  `scripts/gen_tasks.py`（~8K タスク生成、tokenizer 校正）。

→ hermes-swarm は **エンジンを再実装するのでなく、これらを fork の runtime に組み込み、planner front
door と swarm-native boot を足す**のが最短。

---

## 7. fork で潰す既知の穴（前身の残課題）

- **gate-bypass（review #9）**：`chat_completion_helpers.py:1433/1476` の iteration-summary 生成が
  forwarder を経由せず ungated。前身では monkeypatch 危険で defer。**fork なら admission 層に統合して解消。**
- **iteration-limit / thinking-only ループ**：Step3.7 は冗長 reasoner（>10K tok 生成）。出力 cap だけだと
  continuation/thinking-only 再 prefill で詰まる。**fork で role 別 reasoning_effort と境界を既定化。**
- **sandbox 非隔離**：tool 使用 worker の cwd/bash 共有事故。**本実装で task 毎隔離。**
- **SessionDB 直列化**：120 体の永続が共有 conn で詰まる。**per-thread 化。**

---

## 8. リスクと未解決

- **upstream 乖離**：fork は手動 merge。hermes 本家の重い変更に追従コストが乗る（hot-path を触るほど痛い）。
  → 触る範囲を admission 層 + role 定義 + boot に局所化し、本家コアは極力そのまま使う設計に。
- **planner の品質**：goal → 良い DAG への分解は planner の腕次第。分解が浅い/循環すると swarm が空回り
  （deadlock 検出は engine にあるが、planner の自己検証ループが要る）。
- **C96+**：固定窓 ramp 汚染で未測のまま。gate MAX は当面 96。N比例 warm-up が要る。
- **品質 vs 速度**：skip_memory/skip_context/lean toolset が worker 出力品質を落とすかは未定量。
- **daily driver との GPU 競合**：swarm 起動中は Step3.7 サーバを飽和させる。本番 hermes と同時運用の
  ポリシー（時間帯/別ポート/予約 gate）が要る。

---

## 9. フェーズ（構想レベルのロードマップ）

1. **fork & 移植**：hermes-agent を fork、step37-harness のエンジン6点（§6）を runtime に統合、
   `hermes-swarm` 起動 = boot（board+gate+fleet+aimd 常設）。
2. **目標駆動 front door**：planner role（goal→DAG）+ reducer 集約 + 最終回答。`hermes-swarm "<goal>"`。
3. **admission ネイティブ化**：decode-gate を runtime 層に、gate-bypass 根絶（§7）、sandbox/session 本隔離。
4. **role-native**：planner/worker/reducer/router の system prompt・context 予算・reasoning_effort を既定定義。
5. **運用**：daily hermes との競合ポリシー、観測（live 操業点ダッシュボード）、restart 耐性。
6. **検証**：§4 の操業点を fork 上で再実測、planner 分解品質、end-to-end の goal→レポート。

---

## 10. 由来（provenance — 実データの所在）

- 接地（HermesAgent 内部）：`~/projects/step37-harness/BUILD_SPEC.md`（recon 5 エージェント、line-cite）。
- 実測操業点：`~/projects/step37-harness/results/operating_point.json` + `scripts/throughput_probe.py` /
  `aimd_probe.py`。合成基準：`~/bench/step37-mtp/FLEET_OPTIMUM.md`。
- 検証済みエンジン：`~/projects/step37-harness/fleet/`（compat/engine/admission/metrics/board/warm）。
- 設計と意図：`~/projects/step37-harness/DESIGN.md`（§5 実装状況 / §6 実測）。
- 敵対的レビューと修正：DESIGN.md §5 末尾（CRITICAL/MAJOR 6 観点）。
