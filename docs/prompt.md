# Feature Request: Blood Culture Dashboard (NEST360)

## Overview

Create a new tab named **Blood Culture** within the **Newborn Dashboard**.

The Blood Culture tab should be positioned immediately after the existing **Infection** tab and should follow the same design language, filtering behavior, chart styling, responsiveness, legends, and user interactions currently used throughout the Newborn Dashboard.

The purpose of this dashboard is to support NEST360 neonatal sepsis surveillance by monitoring blood culture performance, contamination rates, and microbiological trends over time.

All indicators must respect the global dashboard filters, including time period, facility, country, region, and any other existing dashboard filters.

---

# Data Sources

## Blood Culture Result

### Variable
`GwrkagnbTet`

### Question
Blood culture for suspected sepsis

### Values

| Code | Value |
|--------|--------|
| 0 | Not Done |
| 1 | Done - Culture Negative |
| 2 | Done - Culture Positive |
| 3 | Done but Unknown Result |

---

## Culture Positive Organism (Blood)

### Variable
`aCHOclZEx6o`

### Question
Culture Positive Organism (Blood)

### Values

| Code | Organism |
|--------|--------|
| 1 | Staphylococcus aureus |
| 2 | Klebsiella spp. |
| 3 | Pseudomonas spp. |
| 4 | Escherichia coli |
| 5 | Acinetobacter spp. |
| 6 | Group B Streptococcus |
| 7 | Other gram-negatives |
| 8 | Other Streptococcus |
| 9 | Other Fungal spp. |
| 99 | Other (specify below) |
| 11 | Not indicated |

---

## Other Organism Specification

### Variable
`HvJ8H9tun4u`

### Question
If other - full species/genus of microorganism

This field should only be used when:

```text
aCHOclZEx6o = 99
```

The organism name entered in this field should replace "Other" in all charts, tables, tooltips, and calculations.

---

# Dashboard Layout

The Blood Culture tab should contain four visualizations displayed in the following order:

1. Positive Blood Culture Rate Among All Cultures Done
2. Percent Probable Contaminants Among Positive Blood Cultures
3. Distribution of Microorganisms Identified in Positive Blood Cultures
4. Monthly Trend of Microorganisms Identified in Positive Blood Cultures

---

# Indicator 1

## Positive Blood Culture Rate Among All Cultures Done

### NEST360 Definition

Measures the proportion of blood cultures performed that yielded a positive culture result.

### Chart Type

Coverage Run Chart

### Numerator

Number of babies with:

```text
GwrkagnbTet = 2
```

(Done – Culture Positive)

### Denominator

Number of babies with a blood culture performed:

```text
GwrkagnbTet = 1
OR
GwrkagnbTet = 2
OR
GwrkagnbTet = 3
```

Exclude:

```text
GwrkagnbTet = 0
```

(Not Done)

### Formula

```text
Positive Blood Culture Rate (%) =
(Number of Positive Blood Cultures ÷ Number of Blood Cultures Performed) × 100
```

### Hover Tooltip

Display:

- Indicator Name
- Time Period
- Numerator
- Denominator
- Percentage

Example:

```text
Positive Blood Culture Rate

Period: Jan 2026

Numerator: 24
Denominator: 87

Rate: 27.6%
```

---

# Indicator 2

## Percent Probable Contaminants Among Positive Blood Cultures

### NEST360 Definition

Measures the proportion of positive blood cultures that are likely contaminants rather than clinically significant bloodstream infections.

### Chart Type

Coverage Run Chart

### Probable Contaminant Definition

For the initial implementation, classify the following organisms as probable contaminants:

- Staphylococcus aureus
- Other Streptococcus

The contaminant mapping should be configurable and easy to update in future releases without changing indicator logic.

### Numerator

Number of positive blood cultures where the identified organism is classified as a probable contaminant.

### Denominator

Number of positive blood cultures:

```text
GwrkagnbTet = 2
```

### Formula

```text
Probable Contaminant Rate (%) =
(Number of Probable Contaminants ÷ Number of Positive Blood Cultures) × 100
```

### Hover Tooltip

Display:

- Indicator Name
- Time Period
- Numerator
- Denominator
- Percentage

Example:

```text
Probable Contaminants

Period: Jan 2026

Numerator: 4
Denominator: 24

Rate: 16.7%
```

---

# Indicator 3

## Distribution of Microorganisms Identified in Positive Blood Cultures

### NEST360 Definition

Shows the proportional distribution of microorganisms identified among all positive blood cultures.

### Chart Type

Horizontal Percentage Bar Chart

This chart should rank microorganisms from most frequent to least frequent.

### Inclusion Criteria

Only include:

```text
GwrkagnbTet = 2
```

### Organism Determination

Primary organism source:

```text
aCHOclZEx6o
```

If:

```text
aCHOclZEx6o = 99
```

then use:

```text
HvJ8H9tun4u
```

as the organism name.

Exclude:

```text
aCHOclZEx6o = 11
```

(Not indicated)

### Numerator

Number of positive cultures containing a specific microorganism.

### Denominator

Total microorganisms identified in positive blood cultures.

### Formula

```text
Microorganism Percentage (%) =
(Number of Cultures with Organism ÷ Total Positive Organisms Identified) × 100
```

### Display

For each microorganism show:

- Organism Name
- Count
- Percentage

Sorted descending by frequency.

### Hover Tooltip

Display:

- Organism Name
- Count
- Total Positive Organisms
- Percentage

Example:

```text
Klebsiella spp.

Count: 42
Total Positive Organisms: 120

Percentage: 35.0%
```

---

# Indicator 4

## Monthly Trend of Microorganisms Identified in Positive Blood Cultures

### NEST360 Definition

Tracks how the prevalence of specific microorganisms changes over time.

### Chart Type

Multi-Series Run Chart

### Inclusion Criteria

Only include:

```text
GwrkagnbTet = 2
```

### Organism Determination

Use:

```text
aCHOclZEx6o
```

and where:

```text
aCHOclZEx6o = 99
```

display:

```text
HvJ8H9tun4u
```

Exclude:

```text
aCHOclZEx6o = 11
```

### Numerator

Number of positive cultures containing a specific microorganism during the selected time period.

### Denominator

Total positive organisms identified during the same time period.

### Formula

For each microorganism:

```text
Microorganism Trend (%) =
(Number of Positive Cultures with Organism During Period ÷ Total Positive Organisms During Period) × 100
```

### Chart Configuration

X-Axis:
- Time Period

Y-Axis:
- Percentage (%)

Series:
- One line per microorganism

### Default Behavior

Display the five most common microorganisms by default.

Provide a microorganism selector allowing users to add or remove organisms from the chart.

### Hover Tooltip

Display:

- Organism Name
- Time Period
- Organism Count
- Total Positive Organisms
- Percentage

Example:

```text
Klebsiella spp.

Period: Jan 2026

Count: 12
Total Positive Organisms: 24

Percentage: 50.0%
```

---

# General Requirements

- All percentages should be displayed with one decimal place.
- All charts must update dynamically when dashboard filters change.
- All charts must support export functionality consistent with existing dashboard components.
- All charts must support hover tooltips and legends consistent with the rest of the Newborn Dashboard.
- Records with "Not indicated" organism values must not contribute to organism distribution or microorganism trend calculations.
- Organisms entered through the "Other" free-text field must be displayed using the actual organism name entered by the user.
- All calculations must follow NEST360 indicator definitions exactly.
- Ensure visual consistency with existing NEST360 Coverage Run Charts and organism surveillance visualizations used throughout the application.