Bản thay thế Eq.4-6 (Section 3.3, tiếng Anh — sẵn sàng đưa vào .tex)
latex
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
mechanism accounts for the remaining 11.35\% of synthetic SQLi rows and,
at the time of writing, has not been extended to XSS payloads, whose
obfuscation techniques (encoding- and attribute-based rather than
keyword-splitting) require a different perturbation strategy; we treat
this asymmetry as an open direction rather than a claimed capability
(Section~\ref{sec:limitations}).

Both mechanisms use two independently seeded random streams
(\texttt{seed=42}): one for row/payload sampling (Mechanism 1) and one for
noise-position sampling (Mechanism 2), ensuring the augmentation is
deterministic and reproducible from the seed alone.