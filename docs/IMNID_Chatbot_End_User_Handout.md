# IMNID Dashboard Chatbot — End User Handout

## What it is
The IMNID Dashboard Chatbot lets you ask questions in normal language and instantly shows the matching dashboard chart or table (Maternal or Newborn).

## What it can do
- Show charts (trend over time) for an indicator
- Filter by **region** or **facility**
- Compare **regions** or **facilities** (use words like `compare`, `vs`, `versus`)
- Compare facilities **within a region** (use `by facility` / `per facility` / `breakdown by facility`)
- Remember your last filters for **follow‑up questions**
- List available **indicators**, **regions**, and **facilities**

## What AI it uses
- Uses **Gemini AI** (model: **gemini-2.5-flash-lite**) to understand your message and translate it into the correct dashboard request.

## How to ask (2 simple steps) 
1) Say what you want to see:
   - **Action** (plot/compare/list) **What** (indicator) + **Where** (region/facility) + **When** (date range)
2) Ask follow‑ups without repeating everything (it remembers your last place and dates).

## Quick examples to try
- `plot c-section rate for tigray from jan 1 2026 to mar 23 2026`
- `plot admitted newborns for tigray from jan 1 2026 to mar 23 2026`
- After any chart, type: `what about admitted mothers`

## Comparing regions and facilities
You can compare **two places** in one message. The chatbot understands comparison words like:
`compare`, `vs`, `versus`, `against`, `difference`, `benchmark`.

### A) Compare one region vs another region
- `compare c-section rate for tigray vs amhara from jan 1 2026 to mar 23 2026`
- `compare pph rate for oromia vs amhara last quarter`

### B) Compare one facility vs another facility
- `compare admitted mothers for ambo university hospital vs ambo general hospital this year`
- `compare kmc coverage by birth weight for ambo university hospital vs adigrat hospital from jan 1 2026 to mar 23 2026`

### C) Compare facilities inside one region (by facility / per facility)
Use these phrases when you want a **breakdown within a region**:
`by facility`, `per facility`, `breakdown by facility`, `compare facilities`.

Examples:
- `plot admitted newborns for tigray by facility from jan 1 2026 to mar 23 2026`
- `show c-section rate for amhara per facility last year`

Tip: If you don’t know the exact facility name, ask first: `list facilities in <region>`.

## Follow‑up questions (it remembers context)
After a chart, you can ask:
- `what about <another indicator>`
- `compare with <another region/facility>`
- `same dates but for <another facility/region>`
- `by facility` (to break down results by facilities in the same region)

## Typos are OK
You can type imperfect text like `tigry`, `hospitel`, `covergae` and it will usually still find the closest match.
If it’s unsure, it will ask you to choose.

## Common messages and what to do
- **“Which X do you mean?”** → reply with the number (e.g., `1`) or type the full name.
- **“Facility not found”** → try spelling again or ask: `list facilities in <region>`.
- **“No data found”** → widen the time range (try `all time`) or try a region instead of one facility.
- **“AI temporarily unavailable”** → try again after a minute (sometimes the AI service hits quota/network limits).

## Helpful commands
- `list indicators`
- `list regions`
- `list facilities in tigray`
- `clear chat`
