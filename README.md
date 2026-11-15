# SPHEREx Time-Lapse Explorer | NASA Space Apps Challenge 2026

**Challenge:** Planet X and SPHEREx  
**Live Capabilities:** Real-time NASA/IPAC IRSA SIAv2 Querying • Multi-Band Infrared Time-Lapse • Difference Imaging • ESA Gaia DR3 Cross-Match

## Overview
**SPHEREx Time-Lapse Explorer** is an open-source web application designed to democratize access to multi-epoch infrared observations from NASA's SPHEREx space telescope. By turning raw FITS spectral cutouts into chronological time-lapses and thermal difference maps, anyone can visually spot moving objects (asteroids, brown dwarfs, variable stars, or Planet X candidates) across 102 near-infrared bands (0.75–5.0 μm).

## Key Features
- **One-Click NASA Validation Targets:** Pre-configured coordinates for *Barnard's Star*, *WISE 0855*, *Epsilon Indi B*, *RW Aurigae*, and *R Draconis*.
- **Custom Celestial Search:** Query any sky region by Right Ascension (RA), Declination (Dec), Field of View, and Infrared Band.
- **Interactive Time-Lapse Player:** Adjustable frame rate (FPS), manual frame scrubber, and instant *Start vs. End Blink Comparator*.
- **Difference Imaging Engine:** Statistical Z-score normalization and pixel-by-pixel subtraction (`Final - Initial`) highlighting proper motion as red/blue dipoles.
- **ESA Gaia DR3 Astrometric Overlay:** Live ADQL query to identify known catalog sources and proper motion (`mas/yr`) within the field of view.
- **Shareable URL Parameters:** Direct deep-linking (`/?ra=...&dec=...&size=...`) for collaborative citizen science.

## Installation & Quick Start
```bash
git clone https://github.com/dixw/spherex-timelapse-explorer
cd spherex-timelapse-explorer

AI Disclosure: Large Language Model assistance was used during development to accelerate FastAPI/Astropy code structuring, debugging, and English documentation drafting. All astronomical images, FITS cutouts, and catalog measurements are 100% authentic scientific data from NASA/IPAC IRSA and ESA Gaia DR3.

python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
uvicorn backend:app --reload
```
Open `http://localhost:8000` in your browser.
