# ae_signal — social media adverse-event signal intelligence (proof of concept)

Reads a file of social media posts, finds adverse events (AEs) reported after vaccination,
codes them to MedDRA with a panel of independent Claude agents, scores each event with a
confidence weighting, and produces standard data tables including
**Proportional Reporting Ratios (PRR)** for the target product.

- **Target product:** Abrysvo (Pfizer RSV vaccine, RSVpreF)
- **Purpose:** signal intelligence only. This is not an ICSR (individual case safety report) intake system.
- **Dictionary:** MedDRA 26.1 ASCII files (licensed, so they are not included in this repo)
- **Status:** experimental. Not validated for GxP use.

## How it works

```
posts file ──► prefilter ──► CODERS (n=3, parallel, each with a search_meddra tool)
               (product         │  extract events, choose LLTs from dictionary search results
               dictionary)      ▼
                           consensus: group by (post, suspect product, PT), coder agreement
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
          VERIFIERS (n=2)              CONFIDENCE RATERS (n=2)
          genuine AE? right product?   calibrated p(AE), p(product),
          right MedDRA term? missed?   p(temporal), p(coding)
                  └─────────────┬─────────────┘
                                ▼
                 confidence = p_ae × p_product × p_coding
                                ▼
          coded_events.csv ─► PRR by PT and by SOC (Evans criteria)
```

### Design choices

- **Agents never invent MedDRA codes.** Coders can only pick LLTs returned by the
  `search_meddra` tool, which does fuzzy matching against your MedDRA files. Every returned
  code is checked against the dictionary again, and the coder is told to fix any that fail.
- **Independent panels.** Each agent in a role runs without seeing the others. Verifiers and
  raters also work blind to each other, so their agreement is real evidence and not an echo.
- **Recall first.** An event proposed by only one of three coders is kept and passed to
  verification. Its low coder agreement lowers its confidence rather than removing it.
- **Confidence components are kept separately** in `coded_events.csv`, so you can re-weight
  them without new API calls:
  - `p_ae` = mean(verifier "genuine AE" vote share, rater p_adverse_event)
  - `p_product` = mean(verifier attribution vote share, rater p_product_attribution)
  - `p_coding` = mean(verifier coding vote share, rater p_coding_accuracy, coder agreement)
  - `confidence` = p_ae × p_product × p_coding
  - `p_temporal` is reported but not multiplied in, because most posts give no timing.
- **Version pinning.** Each event row records the MedDRA version it was coded with.
- **Resumable.** Each processed post is appended to `audit.jsonl`, including the raw output of
  every agent. Re-running the same command skips posts that are already done. The audit file
  is also your audit trail.
- **Refusal fallback** is turned on (`fallbacks: "default"`), so a safety-classifier decline is
  retried on a fallback model instead of losing the post.

### PRR

A "report" is one (post, product) pair with at least one event at or above `--min-confidence`.
For each PT (Preferred Term) and SOC (System Organ Class), the 2×2 table compares Abrysvo
reports with comparator-product reports:

| | term X | other terms |
|---|---|---|
| Abrysvo | a | b |
| comparators | c | d |

- PRR = [a/(a+b)] / [c/(c+d)]
- 95% CI uses the log method; χ² uses Yates correction.
- **Signal** = PRR ≥ 2 **and** χ² ≥ 4 **and** a ≥ 3 (Evans 2001).
- Zero cells get a +0.5 correction, flagged in `zero_cell_corrected`.
- `prr_confidence_weighted` recomputes the PRR with each report counted by its confidence
  instead of 1.

**The comparator matters.** PRR only measures disproportionality relative to a background. By
default the background is every other vaccine mentioned in the same file (Arexvy, mRESVIA,
flu, COVID, Shingrix, …). Use `--comparators Arexvy mRESVIA` for a within-class comparison.
For PRR to be meaningful, the input file must include comparator posts, not just Abrysvo posts.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# Pilot first: 50 posts
python -m ae_signal run --posts posts.csv --meddra-dir /path/to/MedDRA_26_1/MedAscii --out outputs --limit 50

# Full run (resumable; re-run the same command after an interruption)
python -m ae_signal run --posts posts.csv --meddra-dir /path/to/MedAscii --out outputs \
    --coders 3 --verifiers 2 --raters 2 --concurrency 8

# Re-score PRR at a different confidence threshold or comparator set (no API calls)
python -m ae_signal prr --out outputs --min-confidence 0.7 --comparators Arexvy mRESVIA

# Offline tests (synthetic MedDRA fixture with fake codes, fake agents)
python -m pytest
```

**Input:** CSV, JSONL or XLSX with a `text` column. `post_id`, `platform`, `created_at` and
any other columns are optional and are carried through to `posts.csv`.

**MedDRA directory:** the folder containing `llt.asc` and `mdhier.asc` (and optionally
`meddra_release.asc`, which is used to confirm the version).

### Outputs (in `--out`)

| File | Contents |
|---|---|
| `posts.csv` | One row per post: status (`ok` / `no_product_mention` / `no_events` / `error`), post type, products matched |
| `coded_events.csv` | One row per (post, suspect product, PT): verbatims, LLT→PT→HLT→HLGT→SOC, event kind, experiencer, seriousness, every agent vote, confidence components |
| `event_summary.csv` | Counts and mean confidence by product × SOC × PT |
| `prr_pt.csv`, `prr_soc.csv` | 2×2 counts, PRR, 95% CI, χ², weighted PRR, signal flag |
| `audit.jsonl` | Raw agent outputs per post, with timestamp and MedDRA version |

## Cost and scale

Each post that passes the prefilter costs 3 coder conversations (usually 2–4 turns each), plus
2 verifier calls and 2 rater calls, all on `claude-opus-5-5`. My rough estimate is
**$0.25–0.40 per post**, which is **$5–8k for 20,000 posts** if all of them mention a vaccine.
Measure the real figure on a 100-post pilot before committing to the full run. Ways to reduce
it, in order:

1. Tighten the prefilter. Posts with no vaccine mention never reach an agent.
2. Reduce the panel, e.g. 2 coders, 1 verifier, 1 rater.
3. Use a smaller model for coders. `RoleConfig.model` can be set per role.
4. Move verifier and rater calls to the Message Batches API, which is 50% cheaper but asynchronous.

## Known limitations and next steps

- **Validation:** build a gold set of about 300 posts double-coded by PV coders. Then measure
  recall and precision per stage and calibrate the confidence scores against it.
- **Synonym clusters:** coders that choose different PTs for the same words (e.g. *Pyrexia*
  vs *Body temperature increased*) produce two candidate events. Grouping by HLT or SMQ, or
  adding a reconciliation step, would merge these.
- **Missed events** that verifiers report are stored in `audit.jsonl` but not yet re-coded.
- **Deduplication** of reposts and cross-posts is not done yet. Duplicates inflate `a`.
- **Lay-language mapping** is handled entirely by the coder's own query reformulation. A
  curated consumer-health-vocabulary list would make retrieval more consistent.
- **Platform posts are untrusted input.** Prompts tell agents to treat them as data only.
  Agents have no tools besides the read-only MedDRA search.
