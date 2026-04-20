# final.csv — CFST Experimental Database

Concrete-Filled Steel Tube (CFST) unified experimental dataset, consolidating test results from **619 published references** into a single flat table with **8 566 specimen records**.

The dataset covers three cross-section families — square/rectangular, circular, and round-ended — and includes both axially and eccentrically loaded specimens. All geometric parameters are mapped to a common set of variables so that **different section shapes can be described by the same 16 columns**.

---

## Overview

| Item | Value |
|------|-------|
| Total specimens | 8 566 |
| Source papers | 619 |
| Group A (square / rectangular) | 4 029 |
| Group B (circular) | 4 485 |
| Group C (round-ended) | 52 |
| Axial loading (e = 0) | 6 420 |
| Eccentric loading (e > 0) | 2 140 (single-end or double-end) |
| Recycled-aggregate specimens (R > 0) | 1 012 |
| Concrete strength fco | 6.68 – 193.3 MPa |
| Steel yield strength fy | 115 – 1 233 MPa |
| Experimental capacity Nexp | 14.4 – 61 980 kN |

---

## Column Definitions (17 columns)

| # | Header | Unit | Description |
|---|--------|------|-------------|
| 1 | **Ref.No.** | — | Full bibliographic citation identifying the source paper |
| 2 | **fco (MPa)** | MPa | Reported concrete compressive strength, as-is from source |
| 3 | **type** | — | Physical specimen used to measure fco (e.g. `Cube 150`, `Cylinder 100x200`, `Prism 150x150x300`). Records the original test specimen geometry, not a derived symbol |
| 4 | **Specimen** | — | Unique specimen label / ID as reported in the source paper |
| 5 | **fy (MPa)** | MPa | Steel tube yield strength |
| 6 | **fcy150 (MPa)** | MPa | Standardized 150 mm x 300 mm cylinder compressive strength, converted from fco according to the specimen type. See *fcy150 Conversion* section below |
| 7 | **R (%)** | % | Recycled coarse-aggregate replacement ratio. `0` = natural aggregate or not reported |
| 8 | **b (mm)** | mm | Outer width of the section (see *Geometry Unification*) |
| 9 | **h (mm)** | mm | Outer depth of the section (see *Geometry Unification*) |
| 10 | **t (mm)** | mm | Steel tube wall thickness |
| 11 | **r0 (mm)** | mm | Outer corner / arc radius (see *Geometry Unification*) |
| 12 | **L (mm)** | mm | Effective calculation length of the member |
| 13 | **e1 (mm)** | mm | Loading eccentricity at the upper end. `0` for axial specimens |
| 14 | **e2 (mm)** | mm | Loading eccentricity at the lower end. `0` for axial specimens |
| 15 | **Nexp (kN)** | kN | Experimental ultimate load capacity |
| 16 | **Group** | — | Section shape group: `A` / `B` / `C` (see below) |
| 17 | **Notes** | — | Supplementary remarks (e.g. specimen anomalies, conversion assumptions) |

---

## Section Groups and Geometry Unification

The dataset uses four geometric parameters — **b, h, t, r0** — to describe all section shapes in a unified manner. The key insight is that square, rectangular, circular, and round-ended sections can all be parameterized by the same variables, with `r0` (outer corner radius) acting as the shape discriminator.

```
┌─── b ───┐
│ ┌─r0    │
│ │       │ h       Typical column cross-section
│ └───────│         with rounded corners
└─────────┘
     t (wall thickness)
```

### Group A — Square / Rectangular

- **b** = outer width, **h** = outer depth
- **r0 = 0** by default (sharp corners); nonzero only when corner radii are explicitly reported
- Includes square sections (b = h) and rectangular sections (b ≠ h)
- 4 029 specimens, b ∈ [50, 1001] mm

### Group B — Circular

- **b = h = D** (outer diameter)
- **r0 = D / 2** (the entire section is a single arc)
- 4 485 specimens, D ∈ [25.4, 1100.4] mm

### Group C — Round-ended

- **b** = outer width (the longer dimension)
- **h** = outer depth (the shorter dimension, including the semicircular ends)
- **r0 = h / 2** (the short-side ends are semicircles)
- Not to be confused with elliptical sections (which are excluded)
- 52 specimens, b ∈ [120, 806] mm, h ∈ [50, 264] mm

This parameterization allows a single set of columns to represent all three families. In downstream modelling, `r0` naturally encodes the shape: `r0 = 0` → sharp rectangle, `r0 = D/2` (and `b = h`) → circle, `0 < r0 = h/2 < b/2` → round-ended.

---

## fcy150 Conversion

All reported concrete strengths (fco) are converted to a unified baseline: the **standard 150 mm x 300 mm cylinder strength** (fcy150). The conversion depends on the specimen type recorded in the `type` column and follows GB/T 50081-2019 (for cubes) and Yi et al. (2006) (for square prisms). Detailed conversion formulas are documented in `references/fcy150_conversion.md` within the skill directory.

---

## Eccentricity Convention

- **Axial compression**: `e1 = e2 = 0`
- **Single-end eccentricity**: one of e1, e2 is nonzero
- **Double-end eccentricity**: both e1 and e2 are nonzero (may differ in magnitude or sign depending on the loading scheme)

---

## File Format

- Encoding: **UTF-8 with BOM**
- Delimiter: comma
- Quoting: fields containing commas or quotes are double-quoted (RFC 4180)
- Numeric values are stored as-is from the source — no rounding or normalization is applied except for the computed fcy150 column
