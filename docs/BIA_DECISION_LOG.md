# Quyết định #13/#14 — Hướng xử lý BIA (Section 3.3)
### Trạng thái: HOÀN TẤT — quyết định đã chốt, mục 3 đóng theo (a), mục 3b đã verify qua 5 seed. Bản LaTeX sẵn sàng đưa vào `.tex` bất cứ lúc nào (vẫn ưu tiên hoãn — code trước, paper sau, theo lựa chọn đã chọn)

---

## 1. Quyết định đã chốt

**Không implement `class BiA`/Eq.4-6 gradient-based injection như bản thảo gốc mô tả.**

Lý do: bài báo mô tả 1 cơ chế (gradient-based injection-point optimization) chưa từng tồn tại trong code. Thay vào đó, code thật dùng 2 cơ chế khác, đã chạy và có bằng chứng hiệu quả:

| Cơ chế | Vị trí code | Vai trò | % dữ liệu SQLi |
|---|---|---|---|
| **Mechanism 1 — Template poisoning** | `balance_dataset()` trong `normalization.py` | Thay toàn bộ `content` benign bằng payload tấn công | 88,20% (`sqli_pool` + `sqli_pool_extra`) |
| **Mechanism 2 — Keyword-noise insertion** | `add_train_noise_augmentation.py` | Chèn nhiễu vào giữa từ khoá SQL để chống comment-splitting | 11,35% (`sqli_pool_noise`) |

→ Bài sẽ mô tả đúng 2 cơ chế này thay cho Eq.4-6 cũ. **Chưa viết vào `.tex` — xem mục 4.**

---

## 2. Dữ liệu đã xác minh (nguồn: `docs/EXPERIMENT_LOG_semantic_edge_investigation.md`, commit `4c1fa49`)

**Phân bố nguồn thật trong `augmented_web_attack.csv` (43.595 dòng):**
- SQLi (18.595): `sqli_pool`=53,78% · `sqli_pool_extra`=34,42% · `sqli_pool_noise`=11,35% · `csic_original`=0,45%
- XSS (10.000): `NA`=100% — **không có mechanism-2 nào áp dụng cho XSS**
- Benign (15.000): `NA`=66,67% · `benign_short_synthetic`=33,33%

**Mechanism 1 — không có kiểm tra tính hợp lệ HTTP:** xác nhận bằng đọc trực tiếp `balance_dataset()` (dòng 247-267) — cả 2 vòng lặp SQLi/XSS chỉ ghi đè `content` rồi append thẳng, không regex/length/format check nào.

**Mechanism 2 — thuật toán rời rạc đầy đủ (8 bước, đã liệt kê trong EXPERIMENT_LOG):**
- 2 luồng random độc lập, cùng `seed=42` (1 cho `DataFrame.sample`, 1 cho `random.Random` quyết định nhiễu)
- 29 từ khoá SQL cố định
- `NOISE_FRACTION=0.35`
- Vị trí chèn: uniform nội bộ từ khoá (`randint(1, len-1)`)
- 5 biến thể whitespace filler cố định
- Tỉ lệ skip thật đo được: 129/2240 = 5,76% (dòng không khớp từ khoá nào)

## 3. Bất đối xứng SQLi/XSS — ĐÃ ĐÓNG theo hướng (a): mở rộng, có bằng chứng

**Đã viết `scripts/add_xss_context_augmentation.py` (Mechanism 2b)** — không rập khuôn theo Mechanism 2 của SQLi (chèn nhiễu *vào trong* từ khoá), vì bước feasibility-check xác nhận: comment-splitting kiểu SQL không có tương đương hợp lệ trong HTML (chèn ký tự vào giữa tên thuộc tính sẽ làm trình duyệt không còn nhận đúng thuộc tính). Thay vào đó, Mechanism 2b chèn **thuộc tính HTML benign xen giữa cặp từ khoá ngữ nghĩa** (`img`↔`onerror`, `script`↔`src`, `svg`↔`onload`) — giữ nguyên từng từ khoá, chỉ tăng khoảng cách token, đã xác nhận vẫn là payload hợp lệ chạy được trên trình duyệt thật (HTML5 cho phép số lượng thuộc tính tuỳ ý, thứ tự tuỳ ý).

**Kết quả (186/8549 dòng XSS đủ điều kiện, lấy mẫu 35% → 64 dòng mới, `source='xss_pool_context'`):** test split F1=1,0 (không đổi), held-out 9-cell giữ 8/9 (không hồi quy) — **nhưng KHÔNG xác nhận được giả thuyết ban đầu** rằng gap vượt `window=15` sẽ làm model đoán sai. Xem mục 3b để biết lý do thật và phát hiện quan trọng hơn đã lộ ra từ đó.

Dù giả thuyết gốc sai, mục tiêu ban đầu của mục 3 (đối xứng phương pháp luận SQLi/XSS trong mô tả BIA) **đã đạt được** — Mechanism 2 giờ có 2 nhánh áp dụng cho cả 2 lớp tấn công, không còn bất đối xứng 100%/0% như trước.

---

## 3b. Phát hiện quan trọng hơn mục tiêu ban đầu — "Unified failure/success rule"

Thí nghiệm mục 3 (đẩy gap vượt `window=15`, đo cả gap=12 trong-range và gap=34 ngoài-range) cho kết quả: **cả 2 đều đúng 10/10, xác nhận VỮNG QUA CẢ 5 SEED** (42-46, dùng đúng 5 checkpoint đã tạo ra `final_stats_5seed.csv`/`heldout_percell_5seed.csv`), dù xác nhận bằng code rằng ở gap≥16, `E_sem` = 0 cạnh thật (100% biến mất). Softmax margin tối thiểu đo được trên toàn bộ 100 dòng×seed là 0,9999998 — không phải kết quả biên may mắn. So sánh trực tiếp với `data_uri_base64` (cell XSS khác cùng nhóm held-out, chỉ 4/5 seed, std=0,447): gap=12/34 **ổn định hơn hẳn**, không cùng mức dao động (`results/gap_window_5seed.csv`, commit `0a9ccdb`). Điều này **tinh chỉnh lại** (không phủ định) kết luận cũ ở §11(b) của EXPERIMENT_LOG:

| | Từ khoá còn nguyên | Từ khoá bị phá |
|---|---|---|
| **E_sem edge còn tồn tại** | (bình thường, đúng) | không xảy ra được |
| **E_sem edge mất** (do gap/noise) | **attribute-spacing: ĐÚNG** — nhờ node feature + E_seq/E_skip cục bộ vẫn còn tín hiệu từ vựng dự phòng | **comment-splitting: SAI** — không còn tín hiệu nào để cứu |

**Kết luận chính xác hơn:** E_sem là lớp phòng thủ THỨ HAI — vai trò của nó chỉ lộ rõ khi lớp phòng thủ THỨ NHẤT (tín hiệu từ vựng thô, node feature) *cũng* đã bị vô hiệu hoá đồng thời (như comment-splitting làm với cả từ khoá lẫn cạnh). Chỉ mất cạnh mà giữ nguyên từ khoá (attribute-spacing) thì vẫn còn đường dự phòng, model vẫn đúng. Đây là bản refine **yếu hơn nhưng đúng hơn** claim "window=15 là ngưỡng cứng vô hiệu hoá classification" — đã cập nhật đúng vào EXPERIMENT_LOG, không viết quá tay so với bằng chứng.

**Trade-off nếu tăng `window`:** rủi ro chính là tăng false-positive kết nối ngữ nghĩa (nối nhầm 2 token không liên quan về mặt tấn công), không phải chi phí tính toán (E_sem vốn đã là loại cạnh ít nhất trong 3 loại).

---

## 4. Bản nháp LaTeX (ĐÃ VIẾT XONG, TẠM GIỮ — chỉ đưa vào `.tex` sau khi mục 3 được quyết định và các việc code ưu tiên khác đã xong)

```latex
\subsection{Behavioral Injection Augmentation (BIA)}

Unlike prior formulations that describe payload injection as a continuous
optimization problem, our implementation of BIA consists of two discrete,
verifiable mechanisms applied at dataset construction time.

\textbf{Mechanism 1 -- Template poisoning.} For a target class $c \in
\{\text{SQLi}, \text{XSS}\}$, a benign template $t$ is sampled without
replacement from the benign pool, and its \texttt{content} field is
replaced in full by a payload $p$ drawn from a class-specific payload
pool:
\begin{equation}
s'_i = \text{template}(t_i) \;\text{with}\; \texttt{content} \leftarrow p_i,
\quad p_i \sim \mathcal{P}_c
\end{equation}
where $\mathcal{P}_{\text{SQLi}}$ is a curated pool of 41 payloads spanning
five attack families (Boolean-, UNION-, Time-, Error-based, and stacked
queries; Table~\ref{tab:sqli_pool}), and $\mathcal{P}_{\text{XSS}}$ is
drawn from an external XSS corpus. This mechanism accounts for 88.2\% of
synthetic SQLi rows and 100\% of XSS rows in the final dataset. No
HTTP-structural-validity check is performed after substitution; the
resulting request retains the surrounding template's field names but not
its original value, which we discuss as a limitation in
Section~\ref{sec:limitations}.

\textbf{Mechanism 2 -- Keyword-noise insertion.} To improve robustness
against comment-splitting obfuscation (e.g. \texttt{UNI/**/ON}), a
disjoint 35\% subset of Mechanism-1 SQLi rows is further perturbed by
inserting whitespace-family noise inside SQL keywords:
\begin{equation}
\text{noise}(k) = k[:j] \oplus \delta \oplus k[j:], \quad
j \sim \mathcal{U}\{1, |k|-1\}, \quad \delta \sim \mathcal{U}\{\delta_1,
\ldots, \delta_5\}
\end{equation}
where $k$ is drawn from a fixed 29-keyword list, $j$ is a uniformly random
interior split position, and $\delta$ is one of five whitespace-style
filler variants. Rows whose sampled keyword does not literally occur in
the payload are left unperturbed (5.76\% of the eligible subset). This
mechanism accounts for the remaining 11.35\% of synthetic SQLi rows. Unlike
SQL, HTML syntax does not permit an equivalent keyword-fragmentation
strategy: inserting characters inside an attribute name (e.g.
\texttt{on\textbackslash x00error=}) is not tolerated by HTML parsers and
does not preserve the semantic keyword, so no XSS analogue of
Mechanism~2 exists in this form.

\textbf{Mechanism 2b -- Context-spacing insertion (XSS).} Instead, for a
disjoint 35\% subset of Mechanism-1 XSS rows whose payload contains one of
three semantic keyword pairs (\texttt{img}/\texttt{onerror},
\texttt{script}/\texttt{src}, \texttt{svg}/\texttt{onload}), we insert one
or two benign HTML attributes (e.g. \texttt{data-*}, \texttt{class},
\texttt{id} with randomized values) between the two keywords, verified via
the tokenizer to keep the resulting token gap below the semantic-edge
window (Section~\ref{sec:esem_window}). This preserves both keywords
intact -- unlike Mechanism~2's noise, which fragments them -- and remains
valid, browser-executable HTML5. This mechanism accounts for 64 additional
XSS rows (0.7\% of synthetic XSS).

Both mechanisms use two independently seeded random streams
(\texttt{seed=42}): one for row/payload sampling (Mechanism 1) and one for
noise-position or attribute-insertion sampling (Mechanisms 2/2b), ensuring
the augmentation is deterministic and reproducible from the seed alone.

\subsubsection{The role of $E_{sem}$ under obfuscation}
\label{sec:esem_window}

Semantic edges are constructed within a fixed token window
($\text{window}=15$); pairs of semantically related tokens separated by
more tokens than this receive no $E_{sem}$ edge, independent of training
data. We probed this boundary directly by varying the token distance
between a keyword pair via Mechanism~2b's attribute insertion, comparing a
within-window case (gap$=12$) against an out-of-window case (gap$=34$,
zero $E_{sem}$ edges by construction). Contrary to our initial hypothesis
that exceeding the window would degrade classification, both cases were
classified correctly across all five seeds tested (accuracy $1.0$,
$n{=}10$ per group per seed; minimum softmax margin $0.9999998$ across all
100 seed-row combinations) -- markedly more stable than
\texttt{data\_uri\_base64}, a nominally "always-correct" XSS held-out
cell that we separately found correct on only 4/5 seeds (accuracy
$0.8 \pm 0.447$). We attribute the gap-window result to a compensating
pathway: Mechanism~2b preserves both keyword tokens intact, so node-level
lexical features together with local $E_{seq}$/$E_{skip}$ connectivity
remain sufficient even when $E_{sem}$ is entirely absent. This contrasts
with comment-splitting (Mechanism~2), which fragments the keyword tokens
themselves \emph{and} removes the $E_{sem}$ edge simultaneously -- leaving
no compensating signal, and failing consistently across all 5 seeds tested
(Section~\ref{sec:ablation}), unlike the gap-window case. We therefore
refine our earlier claim: $E_{sem}$ functions as a second line of defense,
whose contribution becomes visible only when the lexical signal it depends
on is \emph{also} disrupted, not whenever the edge itself is structurally
absent.
```

**Việc kèm theo khi đưa đoạn này vào `.tex` (chưa làm, chỉ ghi chú):**
1. Xoá/thay `\cite{vitorino2022adaptative}` ở Introduction dòng 157 ("gradient-based" không còn đúng).
2. Sửa Contribution #2 trong Introduction — bỏ cụm "gradient-based optimization".
3. Sửa lại câu kết luận §11(b) trong bảng ablation (EXPERIMENT_LOG) cho khớp "unified rule" ở mục 3b — thêm điều kiện, không để nguyên phát biểu vô điều kiện cũ.
4. **Không còn cần** câu "bất đối xứng SQLi/XSS" trong Limitations (mục 3 đã đóng) — thay bằng câu ngắn hơn về phát hiện 3b (E_sem là lớp phòng thủ thứ hai, có điều kiện, đã verify qua 5 seed) nếu muốn nhấn thêm trong Limitations.
5. ~~Re-run gap=12/34 qua 5 seed~~ — **ĐÃ XONG**: 5/5 ổn định cả 2 nhóm, ổn định hơn hẳn `data_uri_base64` (4/5). Xem `results/gap_window_5seed.csv`, commit `0a9ccdb`.

---

## 5. Việc tiếp theo (theo đúng thứ tự ưu tiên đã chọn: code trước, paper sau)

Mục 3 đã đóng — không còn quyết định nào treo cho phần BIA. Quay lại các mục Tầng 2/Tầng 3 còn lại trong `LOGWAT_Checklist_Theo_Muc_Do_San_Sang.md`:
- #11 Latency đo lại, #12 mean±std qua ≥5 seed (Tầng 2).
- #15 Transformer baseline (Tầng 3, rủi ro cao nhất còn lại).

**Lưu ý khi cập nhật Checklist chính:** mục "D — Limitations" ở Tầng 1 cần điều chỉnh nội dung (bỏ ý "bất đối xứng SQLi/XSS", thay bằng ý "E_sem là lớp phòng thủ thứ hai, có điều kiện" từ mục 3b) — chưa cập nhật file Checklist, cần làm ở lượt sau nếu muốn đồng bộ.
