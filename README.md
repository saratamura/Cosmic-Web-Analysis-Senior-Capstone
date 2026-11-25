# Mapping the Cosmic Web
### An Unsupervised Machine Learning Approach to Large-Scale Structure Identification

**Senior physics capstone | Sara Tamura | Gettysburg College | Fall 2026**

This project studies the three-dimensional distribution of galaxies observed in the Sloan Digital Sky Survey (SDSS) and Six-Degree Field Galaxy Survey (6dFGS). It integrates saved density-based clustering results, principal component analysis (PCA) morphology candidates, literature cluster reference positions, and exploratory underdense regions into an interactive 3D cosmic web atlas and all-sky RA/Dec view.

## Research at a glance

| Metric | Result |
|---|---:|
| Galaxies in saved assignments | 476,955 |
| Characterized DBSCAN groups | 1,531 |
| Proximity-match rows | 8 |
| Exploratory empty-sphere candidates | 44 |
| Morphology categories | 4 |

The numbers above are from the **complete capstone analysis**. The visualization displays sampled points for performance.

## Approach

- Group galaxies spatially using density-based clustering (DBSCAN; previously computed assignments).
- Characterize geometric candidates using PCA: **filament-like**, **sheet-like**, **compact/round**, and **complex/elongated**.
- Compare group locations to approximate literature references for Coma, Leo, Hercules, and Abell 3558.
- Visualize candidate empty spheres as an exploratory indicator of underdensity.
- Display both a three-dimensional spatial view and a Mollweide right-ascension/declination sky view using Plotly.

## Run the visualization

```bash
python -m pip install -r requirements.txt
python src/build_atlas.py
```

The script expects these four files inside `data/`:

```text
Full_Survey_Galaxy_Assignments.csv
Full_Survey_Structure_Catalog.csv
Experiment7_Known_Structure_Matches.csv
Experiment8_Void_Candidates.csv
```

It writes `results/Cosmic_Web_Atlas.html` and `results/Atlas_Summary.csv`. If working with the original full dataset, use the complete saved assignments from the research package; a small subset is not equivalent to the reported complete analysis. Because some artifacts are too large for this initial connected upload, the full data and HTML atlas are **not yet included in this repository**. Plotly's sky view may require internet access for geographic assets.

## Scope and limitations

This repository reconstructs the **visualization from previously saved experiment outputs**. It does **not** reproduce DBSCAN, PCA, or validation experiments from raw data because the original upstream analysis scripts are not included here. Morphology labels are geometric candidates, not confirmed physical classifications. Spatial proximity to a reference cluster does not prove membership, and empty spheres are **not validated cosmic voids**. Survey footprint and selection effects affect apparent structure; this is not a complete map of the universe.

## Data provenance and sharing

Underlying observations were derived from SDSS and 6dFGS. Exact survey releases, data citations, and redistribution terms should be checked before making this repository public. No software license is assigned yet.

## Related research

[PyCLEA](https://github.com/saratamura/PyCLEA) — an interactive educational galaxy redshift laboratory developed during an X-SIG research fellowship.
