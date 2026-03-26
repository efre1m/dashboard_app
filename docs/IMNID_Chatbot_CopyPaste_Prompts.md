# IMNID Chatbot — Copy/Paste Prompt Examples

Use these exactly as written (or edit the region/facility and dates).  
Tip: Start by typing `maternal` or `newborn` so the chatbot knows which program you mean.

---

## Maternal (Mothers) — Examples

### 1) C‑Section Rate (%)
- `plot c-section rate for tigray from jan 1 2026 to mar 23 2026`
- `show c-section rate for amhara last year`
- `compare c-section rate for tigray vs amhara from jan 1 2026 to mar 23 2026`
- `compare c-section rate for ambo university hospital vs ambo general hospital from jan 1 2026 to mar 23 2026`
- `show c-section rate for tigray by facility last year`
- Follow‑up after a chart: `what about stillbirth rate`

Typos test:
- `can i see the c section rte for tigry from jan 1 2026 to mar 23 2026`

### 2) Postpartum Hemorrhage (PPH) Rate (%)
- `plot pph rate for oromia this year`
- `compare pph rate for tigray vs amhara last quarter`
- `compare pph rate for ambo university hospital vs ambo general hospital last quarter`
- `plot pph rate for amhara per facility this year`
- `define pph rate`
- Follow‑up after a chart: `same dates but for tigray`
- Follow‑up after a chart: `show as line chart`

### 3) Admitted Mothers
- `plot admitted mothers for tigray this year`
- `plot admitted mothers for tigray from jan 1 2026 to mar 23 2026`
- `compare admitted mothers for tigray vs oromia last year`
- `compare admitted mothers for ambo university hospital vs ambo general hospital last year`
- `show admitted mothers for tigray by facility from jan 1 2026 to mar 23 2026`
- Follow‑up after a chart: `what about maternal coverage rate`
- Follow‑up after a chart: `show as bar chart`

---

## Newborn (Babies) — Examples

### 1) Admitted Newborns
- `plot admitted newborns for tigray from jan 1 2026 to mar 23 2026`
- `plot admitted newborns for amhara last month`
- `compare admitted newborns for tigray vs amhara last quarter`
- `compare admitted newborns for ambo university hospital vs ambo general hospital last quarter`
- `plot admitted newborns for tigray by facility from jan 1 2026 to mar 23 2026`

Follow‑up memory test (after the chart above):
- `what about admitted mothers`
  - Expected: same region + same dates, but for Admitted Mothers.

### 2) KMC Coverage by Birth Weight
- `plot kmc coverage by birth weight for tigray this year`
- `compare kmc coverage by birth weight for tigray vs amhara last quarter`
- `compare kmc coverage by birth weight for ambo university hospital vs ambo general hospital last quarter`
- `plot kmc coverage by birth weight for tigray per facility last quarter`
- `plot kmc coverage by birth weight for ambo university hospital from jan 1 2026 to mar 23 2026`

Typos tests:
- `show kmc covergae by birht wieght for ambo unversity hospitel`
- `show kmc coverage by birth weight for ambo unversity hospital`

If you see “Which Ambo do you mean?”:
- reply `1` (or the number shown), or type the full facility name.

### 3) CPAP for RDS (or CPAP Coverage by Birth Weight)
- `plot cpap for rds for tigray last quarter`
- `compare cpap for rds for tigray vs amhara last year`
- `compare cpap for rds for ambo university hospital vs ambo general hospital last quarter`
- `plot cpap for rds for tigray by facility last quarter`
- `plot cpap coverage by birth weight for tigray this year`

---

## Discovery (when you don’t know what exists)
- `list indicators`
- `list regions`
- `list facilities in tigray`
- `list facilities in amhara`

---

## Follow‑ups you can use any time (after you already got a chart)
- `what about <another indicator>`
- `compare with <another region/facility>`
- `same dates but for <another region/facility>`
- `by facility` (break down results by facilities in the same region)
- `show as bar chart`
- `show as line chart`

---

## Comparing (quick patterns)
- Region vs Region: `compare <indicator> for tigray vs amhara last quarter`
- Facility vs Facility: `compare <indicator> for ambo university hospital vs ambo general hospital last quarter`
- Facilities inside a region: `plot <indicator> for tigray by facility this year`

## If you hit common issues

### “Facility not found”
Try:
- `list facilities in <region>`
- Use region instead: `plot <indicator> for <region>`

### “No data found”
Try:
- `all time`
- A wider range: `from jan 1 2025 to dec 31 2025`
- A region instead of a single facility

### AI temporarily unavailable
Try:
- send the same message again after a minute
- simplify the question (indicator + region + dates)

---

## Reset
- `clear chat`
