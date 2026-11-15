import os
import io
import json
import hashlib
import urllib3
import requests
import csv
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.votable import parse_single_table
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from reproject import reproject_interp
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="SPHEREx Time-Lapse Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/frames", exist_ok=True)
os.makedirs("static/cache", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

PRESETS = [
    {"id": "barnard", "name": "Barnard's Star", "ra": 269.45208, "dec": 4.69336, "desc": "Estrella con mayor movimiento propio conocido en el cielo."},
    {"id": "wise0855", "name": "WISE 0855", "ra": 133.79513, "dec": -7.24515, "desc": "Enana marrón ultrafría de rápido desplazamiento."},
    {"id": "eps_indi", "name": "Epsilon Indi B", "ra": 330.84023, "dec": -56.78598, "desc": "Sistema binario de enanas marrones cercano."},
    {"id": "rw_aur", "name": "RW Aurigae", "ra": 76.95653, "dec": 30.40144, "desc": "Estrella joven con variaciones irregulares de brillo."},
    {"id": "r_dra", "name": "R Draconis", "ra": 248.13375, "dec": 66.75556, "desc": "Estrella variable pulsante de largo período."}
]

@app.get("/")
def serve_home():
    return FileResponse("index.html")

@app.get("/api/presets")
def get_presets():
    return {"presets": PRESETS}

@app.get("/api/rgb")
def get_false_color_rgb(
    ra: float = Query(..., description="Ascensión Recta en grados"),
    dec: float = Query(..., description="Declinación en grados"),
    size: float = Query(0.05, description="Tamaño del recorte en grados")
):
    """Genera una composición científica RGB en Falso Color combinando 3 bandas infrarrojas."""
    host = "irsa.ipac.caltech.edu"
    sia_url = "https://" + host + "/SIA"
    params = {
        "COLLECTION": "spherex_dr1",
        "POS": f"circle {ra} {dec} {size}",
        "RESPONSEFORMAT": "VOTABLE"
    }

    #try:
    #   res = requests.get(sia_url, params=params, timeout=30, verify=False)
    #    res.raise_for_status()
    #    votable = parse_single_table(io.BytesIO(res.content))
    #    table = votable.to_table()
    #except Exception as e:
    #    raise HTTPException(status_code=502, detail=f"Error conectando con NASA IRSA: {str(e)}")
    c_key = hashlib.md5(f"sia_{ra}_{dec}_{size}".encode()).hexdigest()[:12]
    votable = fetch_irsa_votable(sia_url, params, c_key)
    table = votable.to_table()

    if len(table) == 0:
        raise HTTPException(status_code=404, detail="No hay datos suficientes para RGB en esta zona.")

    col_map = {field.name: table.colnames[i] for i, field in enumerate(votable.fields)}

    candidatos = []
    for i in range(len(table)):
        dtype = str(table[col_map["dataproduct_type"]][i]).strip()
        if "image" not in dtype.lower():
            continue
        em_min = float(table[col_map["em_min"]][i]) * 1e6
        em_max = float(table[col_map["em_max"]][i]) * 1e6
        wl = (em_min + em_max) / 2.0
        img_url = str(table[col_map["access_url"]][i]).strip()
        band_name = str(table[col_map["energy_bandpassname"]][i]).strip()
        candidatos.append({"wl": wl, "url": img_url, "band": f"{band_name} ({wl:.2f} μm)"})

    if len(candidatos) < 3:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 3 bandas distintas para componer RGB.")

    # Ordenar de menor longitud de onda (Azul) a mayor longitud de onda (Rojo)
    candidatos.sort(key=lambda x: x["wl"])
    tercio = max(1, len(candidatos) // 3)
    grupos = {
        "blue": candidatos[:tercio],
        "green": candidatos[tercio:2*tercio] or candidatos,
        "red": list(reversed(candidatos[2*tercio:] or candidatos))
    }

    def obtener_canal_valido(lista_candidatos):
        for cand in lista_candidatos[:4]:
            sep = "&" if "?" in cand["url"] else "?"
            cutout_url = f"{cand['url']}{sep}center={ra},{dec}&size={size}"
            try:
                r = requests.get(cutout_url, timeout=45, verify=False)
                with fits.open(io.BytesIO(r.content)) as hdul:
                    for ext in hdul:
                        if ext.data is not None and ext.data.ndim == 2:
                            mat = np.nan_to_num(ext.data, nan=np.nanmedian(ext.data))
                            return mat, cand["band"]
            except Exception:
                continue
        return None, None

    m_b, band_b = obtener_canal_valido(grupos["blue"])
    m_g, band_g = obtener_canal_valido(grupos["green"])
    m_r, band_r = obtener_canal_valido(grupos["red"])

    if m_b is None or m_g is None or m_r is None:
        raise HTTPException(status_code=500, detail="No se pudieron descargar los 3 canales FITS válidos.")

    # Igualar dimensiones mínimas comunes entre los 3 canales (alto y ancho)
    min_h = min(m_b.shape[0], m_g.shape[0], m_r.shape[0])
    min_w = min(m_b.shape[1], m_g.shape[1], m_r.shape[1])

    def normalizar_canal(mat):
        sub = mat[:min_h, :min_w]
        vmin, vmax = ZScaleInterval().get_limits(sub)
        norm = np.clip((sub - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        return np.arcsinh(norm * 3.0) / np.arcsinh(3.0)

    rgb_cube = np.dstack([
        normalizar_canal(m_r),  # Rojo = Infrarrojo lejano
        normalizar_canal(m_g),  # Verde = Infrarrojo medio
        normalizar_canal(m_b)   # Azul = Infrarrojo cercano
    ])

    rgb_hash = hashlib.md5(f"rgb_{ra}_{dec}_{size}".encode()).hexdigest()[:12]
    rgb_filename = f"rgb_{rgb_hash}.png"
    rgb_path = os.path.join("static/frames", rgb_filename)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.imshow(rgb_cube, origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(rgb_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return {
        "rgb_image_url": f"/static/frames/{rgb_filename}",
        "channels": {
            "red": band_r,
            "green": band_g,
            "blue": band_b
        }
    }
def fetch_irsa_votable(sia_url: str, params: dict, cache_key: str):
    """Consulta IRSA verificando que la respuesta sea un VOTABLE real antes de guardarlo en caché."""
    cache_path = os.path.join("static/cache", f"{cache_key}.xml")
    raw_bytes = None

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached_data = f.read()
        if b"VOTABLE" in cached_data[:1000]:
            raw_bytes = cached_data
        else:
            os.remove(cache_path)

    if raw_bytes is None:
        for intento in range(3):
            try:
                res = requests.get(sia_url, params=params, timeout=45, verify=False)
                res.raise_for_status()
                if b"VOTABLE" not in res.content[:1000]:
                    raise ValueError("IRSA devolvió HTML en lugar de VOTABLE")
                raw_bytes = res.content
                with open(cache_path, "wb") as f:
                    f.write(raw_bytes)
                break
            except Exception as e:
                if intento == 2:
                    raise HTTPException(
                        status_code=502,
                        detail=f"NASA IRSA no devolvió datos válidos tras 3 intentos: {str(e)}"
                    )

    return parse_single_table(io.BytesIO(raw_bytes))

#def fetch_real_catalog_sources(ra: float, dec: float, radius: float):
    """Consulta en cascada real y rápida: 1) ESA Gaia DR3 (8s max) -> 2) NASA IRSA AllWISE/2MASS (CSV)."""
    cache_key = hashlib.md5(f"fast_real_{ra:.5f}_{dec:.5f}_{radius:.4f}".encode()).hexdigest()[:12]
    cache_path = os.path.join("static/cache", f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            if len(cached) > 0:
                return cached

    # NIVEL 1: ESA Gaia DR3 oficial (límite estricto de 8 segundos para no colgar la interfaz)
    try:
        esa_url = "https://esa.int"
        q_esa = (
            f"SELECT TOP 25 source_id, ra, dec, phot_g_mean_mag, pmra, pmdec "
            f"FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) "
            f"ORDER BY phot_g_mean_mag ASC"
        )
        r = requests.post(
            esa_url,
            data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q_esa},
            timeout=(5, 8),
            verify=False
        )
        r.raise_for_status()
        if "source_id" in r.text:
            reader = csv.DictReader(io.StringIO(r.text))
            sources = []
            for row in reader:
                if not row.get("ra") or not row.get("dec"):
                    continue
                pm_r = float(row["pmra"]) if row.get("pmra") and row["pmra"].strip() not in ("", "null", "NaN") else 0.0
                pm_d = float(row["pmdec"]) if row.get("pmdec") and row["pmdec"].strip() not in ("", "null", "NaN") else 0.0
                mag_v = round(float(row["phot_g_mean_mag"]), 2) if row.get("phot_g_mean_mag") and row["phot_g_mean_mag"].strip() not in ("", "null", "NaN") else 0.0
                sources.append({
                    "catalog": "ESA Gaia DR3",
                    "epoch": 2016.0,
                    "source_id": str(row["source_id"]).strip(),
                    "ra": round(float(row["ra"]), 5),
                    "dec": round(float(row["dec"]), 5),
                    "mag": mag_v,
                    "pm_total": round(float(np.hypot(pm_r, pm_d)), 1),
                    "pmra": pm_r,
                    "pmdec": pm_d
                })
            if len(sources) > 0:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(sources, f)
                return sources
    except Exception as e:
        print("ESA Gaia bloqueado/timeout (8s), conmutando a NASA IRSA TAP:", type(e).name)
# NIVEL 2: NASA/IPAC IRSA TAP Oficial (Catálogo AllWISE con movimiento propio en CSV)
    try:
        irsa_tap = "https://caltech.edu"
        q_irsa = (
            f"SELECT TOP 25 designation, ra, dec, w1mpro, pmra, pmdec "
            f"FROM allwise_p3as_psd "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1 "
            f"ORDER BY w1mpro ASC"
        )
        r = requests.post(
            irsa_tap,
            data={"QUERY": q_irsa, "FORMAT": "CSV"},
            timeout=(8, 15),
            verify=False
        )
        r.raise_for_status()
        if "designation" in r.text:
            reader = csv.DictReader(io.StringIO(r.text))
            sources = []
            for row in reader:
                if not row.get("ra") or not row.get("dec"):
                    continue
                pm_r = float(row["pmra"]) if row.get("pmra") and row["pmra"].strip() not in ("", "null", "NaN") else 0.0
                pm_d = float(row["pmdec"]) if row.get("pmdec") and row["pmdec"].strip() not in ("", "null", "NaN") else 0.0
                mag_v = round(float(row["w1mpro"]), 2) if row.get("w1mpro") and row["w1mpro"].strip() not in ("", "null", "NaN") else 0.0
                sources.append({
                    "catalog": "NASA IRSA AllWISE",
                    "epoch": 2010.5,
                    "source_id": str(row["designation"]).strip(),
                    "ra": round(float(row["ra"]), 5),
                    "dec": round(float(row["dec"]), 5),
                    "mag": mag_v,
                    "pm_total": round(float(np.hypot(pm_r, pm_d)), 1),
                    "pmra": pm_r,
                    "pmdec": pm_d
                })
            if len(sources) > 0:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(sources, f)
                return sources
    except Exception as e:
        print("Error en NASA IRSA TAP:", e)

    return []
def fetch_real_catalog_sources(ra: float, dec: float, radius: float):
    """Consulta oficial en tiempo real a NASA/IPAC IRSA SCS (Catálogo AllWISE con movimiento propio)."""
    cache_key = hashlib.md5(f"irsa_scs_allwise_{ra:.5f}_{dec:.5f}_{radius:.4f}".encode()).hexdigest()[:12]
    cache_path = os.path.join("static/cache", f"{cache_key}.xml")
    raw_bytes = None

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            raw_bytes = f.read()

    if raw_bytes is None:
        scs_url = "https://caltech.edu"
        params = {
            "table": "allwise_p3as_psd",
            "RA": ra,
            "DEC": dec,
            "SR": max(radius, 0.05),
            "format": "votable"
        }
        for intento in range(3):
            try:
                res = requests.get(scs_url, params=params, timeout=30, verify=False)
                res.raise_for_status()
                raw_bytes = res.content
                with open(cache_path, "wb") as f:
                    f.write(raw_bytes)
                break
            except Exception as e:
                if intento == 2:
                    print("Error conectando con NASA IRSA SCS:", e)
                    return []

    try:
        votable = parse_single_table(io.BytesIO(raw_bytes))
        table = votable.to_table()
        col_map = {field.name.lower(): table.colnames[i] for i, field in enumerate(votable.fields)}

        col_id = col_map.get("designation") or table.colnames[0]
        col_ra = col_map.get("ra")
        col_dec = col_map.get("dec")
        col_mag = col_map.get("w1mpro")
        col_pmra = col_map.get("pmra")
        col_pmdec = col_map.get("pmdec")

        sources = []
        for i in range(len(table)):
            pmra = table[col_pmra][i] if col_pmra else np.ma.masked
            pmdec = table[col_pmdec][i] if col_pmdec else np.ma.masked
            pm_r = float(pmra) if not np.ma.is_masked(pmra) and not np.isnan(float(pmra)) else 0.0
            pm_d = float(pmdec) if not np.ma.is_masked(pmdec) and not np.isnan(float(pmdec)) else 0.0
            mag_v = round(float(table[col_mag][i]), 2) if col_mag and not np.ma.is_masked(table[col_mag][i]) else 0.0

            sources.append({
                "catalog": "NASA IRSA AllWISE",
                "epoch": 2010.5,
                "source_id": f"WISE {str(table[col_id][i]).strip()}",
                "ra": round(float(table[col_ra][i]), 5),
                "dec": round(float(table[col_dec][i]), 5),
                "mag": mag_v,
                "pm_total": round(float(np.hypot(pm_r, pm_d)), 1),
                "pmra": pm_r,
                "pmdec": pm_d
            })

        # Ordenar de más brillante a más tenue y devolver las 25 principales
        sources.sort(key=lambda x: x["mag"])
        return sources[:25]
    except Exception as e:
        print("Error procesando VOTable de NASA IRSA SCS:", e)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return []

@app.get("/api/gaia")
def get_gaia_sources(ra: float = Query(...), dec: float = Query(...), size: float = Query(0.05)):
    sources = fetch_real_catalog_sources(ra, dec, max(size, 0.05))
    return {"count": len(sources), "sources": sources}

@app.get("/api/overlay")
def get_gaia_overlay(
    ra: float = Query(...),
    dec: float = Query(...),
    size: float = Query(0.05)
):
    """Proyecta fuentes reales de Gaia/AllWISE propagadas a la fecha exacta MJD del archivo FITS."""
    sia_url = "https://caltech.edu"
    params = {
        "COLLECTION": "spherex_dr1",
        "POS": f"circle {ra} {dec} {size}",
        "RESPONSEFORMAT": "VOTABLE"
    }

    c_key = hashlib.md5(f"sia_{ra}_{dec}_{size}".encode()).hexdigest()[:12]
    votable = fetch_irsa_votable(sia_url, params, c_key)
    table = votable.to_table()
    col_map = {field.name: table.colnames[i] for i, field in enumerate(votable.fields)}

    data, wcs_obj, obs_mjd = None, None, None
    for i in range(len(table)):
        dtype = str(table[col_map["dataproduct_type"]][i]).strip()
        if "image" not in dtype.lower():
            continue
        img_url = str(table[col_map["access_url"]][i]).strip()
        obs_mjd = float(table[col_map["t_min"]][i])
        sep = "&" if "?" in img_url else "?"
        cutout_url = f"{img_url}{sep}center={ra},{dec}&size={size}"
        try:
            r = requests.get(cutout_url, timeout=45, verify=False)
            with fits.open(io.BytesIO(r.content)) as hdul:
                for ext in hdul:
                    if ext.data is not None and ext.data.ndim == 2:
                        data = np.nan_to_num(ext.data, nan=np.nanmedian(ext.data))
                        wcs_obj = WCS(ext.header).celestial
                        break
            if data is not None:
                break
        except Exception:
            continue

    if data is None or wcs_obj is None:
        raise HTTPException(status_code=404, detail="No se encontró imagen FITS con WCS válido.")

    obs_year = Time(obs_mjd, format="mjd").jyear if obs_mjd else 2025.5
    sources = fetch_real_catalog_sources(ra, dec, size * 1.2)

    vmin, vmax = ZScaleInterval().get_limits(data)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(data, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)

    h, w = data.shape
    marcados = 0
    for s in sources:
        c_cat = SkyCoord(
            ra=s["ra"] * u.deg,
            dec=s["dec"] * u.deg,
            pm_ra_cosdec=s["pmra"] * u.mas / u.yr,
            pm_dec=s["pmdec"] * u.mas / u.yr,
            obstime=Time(s["epoch"], format="jyear")
        )
        c_obs = c_cat.apply_space_motion(new_obstime=Time(obs_year, format="jyear"))
        px, py = wcs_obj.world_to_pixel(c_obs)
        px, py = float(px), float(py)

        if 0 <= px < w and 0 <= py < h:
            marcados += 1
            circ = plt.Circle((px, py), radius=max(1.8, w * 0.065), color="#00f2fe", fill=False, linewidth=2.2)
            ax.add_patch(circ)

            pm_tot = s.get("pm_total", 0.0)
            if pm_tot > 20.0:
                norm_dx = (-s["pmra"] / pm_tot) * (w * 0.14)
                norm_dy = (s["pmdec"] / pm_tot) * (h * 0.14)
                ax.arrow(px, py, norm_dx, norm_dy, color="#facc15", head_width=max(1.1, w * 0.04), linewidth=2.2, zorder=10)

    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.axis("off")
    plt.tight_layout(pad=0)

    ov_hash = hashlib.md5(f"overlay_v2_{ra}_{dec}_{size}".encode()).hexdigest()[:12]
    ov_filename = f"overlay_{ov_hash}.png"
    ov_path = os.path.join("static/frames", ov_filename)
    plt.savefig(ov_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return {
        "overlay_image_url": f"/static/frames/{ov_filename}",
        "marked_count": marcados,
        "obs_epoch": round(obs_year, 2)
    }



@app.get("/api/timelapse")
def get_timelapse(
    ra: float = Query(..., description="Ascensión Recta en grados"),
    dec: float = Query(..., description="Declinación en grados"),
    size: float = Query(0.05, description="Tamaño del recorte en grados"),
    band: str = Query("ALL", description="Filtro de banda infrarroja o ALL"),
    max_frames: int = Query(10, description="Máximo de fotogramas a procesar")
):
    host = "irsa.ipac.caltech.edu"
    sia_url = "https://" + host + "/SIA"
    params = {
        "COLLECTION": "spherex_dr1",
        "POS": f"circle {ra} {dec} {size}",
        "RESPONSEFORMAT": "VOTABLE"
    }

    #try:
    #    res = requests.get(sia_url, params=params, timeout=30, verify=False)
    #    res.raise_for_status()
    #    votable = parse_single_table(io.BytesIO(res.content))
    #    table = votable.to_table()
    #except Exception as e:
    #    raise HTTPException(status_code=502, detail=f"Error conectando con NASA IRSA: {str(e)}")
    c_key = hashlib.md5(f"sia_{ra}_{dec}_{size}".encode()).hexdigest()[:12]
    votable = fetch_irsa_votable(sia_url, params, c_key)
    table = votable.to_table()

    if len(table) == 0:
        return {"frames": [], "bands": [], "total_found": 0, "diff_image_url": None}

    col_map = {field.name: table.colnames[i] for i, field in enumerate(votable.fields)}

    registros = []
    bandas_disponibles = set()

    for i in range(len(table)):
        dtype = str(table[col_map["dataproduct_type"]][i]).strip()
        if "image" not in dtype.lower():
            continue

        band_name = str(table[col_map["energy_bandpassname"]][i]).strip()
        em_min = float(table[col_map["em_min"]][i]) * 1e6
        em_max = float(table[col_map["em_max"]][i]) * 1e6
        band_label = f"{band_name} ({em_min:.2f}-{em_max:.2f} μm)"
        bandas_disponibles.add(band_label)

        if band != "ALL" and band != band_label:
            continue

        t_min = float(table[col_map["t_min"]][i])
        fecha_iso = Time(t_min, format="mjd").iso[:19]
        img_url = str(table[col_map["access_url"]][i]).strip()
        obs_id = str(table[col_map["obs_id"]][i]).strip()

        registros.append({
            "obs_id": obs_id,
            "t_min": t_min,
            "date": fecha_iso,
            "band": band_label,
            "wavelength_um": round((em_min + em_max) / 2, 3),
            "access_url": img_url
        })

    registros.sort(key=lambda x: x["t_min"])

    if len(registros) > max_frames:
        indices = np.linspace(0, len(registros) - 1, max_frames, dtype=int)
        seleccionados = [registros[idx] for idx in indices]
    else:
        seleccionados = registros

    frames_listos = []
    matrices_2d = []
    light_curve = []
    png_paths_for_gif = []

    for item in seleccionados:
        hash_id = hashlib.md5(f"{item['obs_id']}_{ra}_{dec}_{size}".encode()).hexdigest()[:12]
        png_filename = f"frame_{hash_id}.png"
        png_path = os.path.join("static/frames", png_filename)

        sep = "&" if "?" in item["access_url"] else "?"
        cutout_url = f"{item['access_url']}{sep}center={ra},{dec}&size={size}"

        try:
            r = requests.get(cutout_url, timeout=45, verify=False)
            with fits.open(io.BytesIO(r.content)) as hdul:
                data = None
                hdr = None
                for ext in hdul:
                    if ext.data is not None and ext.data.ndim == 2:
                        data = ext.data
                        hdr = ext.header
                        break
                if data is None:
                    continue

                data = np.nan_to_num(data, nan=np.nanmedian(data))
                matrices_2d.append((data, hdr))
                # Fotometría central (flujo relativo en una ventana central de 5x5 píxeles)
                cy, cx = data.shape[0] // 2, data.shape[1] // 2
                aperture = data[max(0, cy-2):cy+3, max(0, cx-2):cx+3]
                flux_val = round(float(np.nanmean(aperture)), 4)
                if not os.path.exists(png_path):
                    vmin, vmax = ZScaleInterval().get_limits(data)
                    plt.figure(figsize=(5, 5), dpi=100)
                    plt.imshow(data, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
                    plt.close()
        except Exception as e:
            print("Error leyendo FITS:", e)
            continue

        frames_listos.append({
            "obs_id": item["obs_id"],
            "date": item["date"],
            "mjd": round(item["t_min"], 4),
            "band": item["band"],
            "wavelength_um": item["wavelength_um"],
            "image_url": f"/static/frames/{png_filename}"
        })
        png_paths_for_gif.append(png_path)
        light_curve.append({
             "date": item["date"],
              "mjd": round(item["t_min"], 4),
             "flux": flux_val,
            "band": item["band"]
        })
    # Generar Imagen de Diferencia Multi-Objeto con Vectores de Trayectoria
    diff_url = None
    diff_candidates = []
    if len(matrices_2d) >= 2:
        data_ini, hdr_ini = next(iter(matrices_2d))
        data_fin, hdr_fin = matrices_2d[-1]
        min_h, min_w = data_ini.shape

        if min_h >= 6 and min_w >= 6:
            # Alinear astrométricamente la imagen final sobre la rejilla celeste de la inicial
            data_fin_aligned, _ = reproject_interp((data_fin, hdr_fin), hdr_ini)
            data_fin_aligned = np.nan_to_num(data_fin_aligned, nan=np.nanmedian(data_ini))

            m_ini = (data_ini - np.median(data_ini)) / (np.std(data_ini) + 1e-6)
            m_fin = (data_fin_aligned - np.median(data_fin_aligned)) / (np.std(data_fin_aligned) + 1e-6)
            diff_matrix = m_fin - m_ini

            # Buscar píxeles con caída fuerte (azul < percentil 8) y subida fuerte (rojo > percentil 92)
            low_thresh = np.percentile(diff_matrix, 8)
            high_thresh = np.percentile(diff_matrix, 92)

            min_coords = np.argwhere(diff_matrix <= low_thresh)
            max_coords = np.argwhere(diff_matrix >= high_thresh)

            min_coords = sorted(min_coords, key=lambda c: float(diff_matrix[tuple(c)]))
            used_max = set()
            used_min = []

            for y0, x0 in min_coords:
                if not (1 <= y0 < min_h - 1 and 1 <= x0 < min_w - 1):
                    continue
                # Evitar marcar dos veces el mismo grupo azul (mínimo 3 px de separación)
                if any(np.hypot(x0 - ux, y0 - uy) < 3.0 for uy, ux in used_min):
                    continue

                best_j = None
                best_dist = 999.0
                for j, (y1, x1) in enumerate(max_coords):
                    if j in used_max or not (1 <= y1 < min_h - 1 and 1 <= x1 < min_w - 1):
                        continue
                    d = float(np.hypot(x1 - x0, y1 - y0))
                    if 1.0 <= d <= 5.5 and d < best_dist:
                        best_dist = d
                        best_j = j

                if best_j is not None:
                    used_max.add(best_j)
                    used_min.append((y0, x0))
                    y1, x1 = max_coords[best_j]
                    diff_candidates.append({
                        "id": len(diff_candidates) + 1,
                        "start_px": [int(x0), int(y0)],
                        "end_px": [int(x1), int(y1)],
                        "shift_px": round(best_dist, 2),
                        "shift_arcsec": round(best_dist * 6.2, 2)
                    })
                    if len(diff_candidates) >= 5:
                        break

            # Respaldo: si no hubo pares cercanos, usar el dipolo global principal
            #if len(diff_candidates) == 0:
            #    interior = diff_matrix[1:-1, 1:-1]
            #    y0, x0 = np.unravel_index(np.argmin(interior), interior.shape)
            #    y1, x1 = np.unravel_index(np.argmax(interior), interior.shape)
            #    y0 += 1; x0 += 1; y1 += 1; x1 += 1
            #    d = float(np.hypot(x1 - x0, y1 - y0))
            #    diff_candidates.append({
            #        "id": 1,
            #        "start_px": [int(x0), int(y0)],
            #        "end_px": [int(x1), int(y1)],
            #        "shift_px": round(d, 2),
            #        "shift_arcsec": round(d * 6.2, 2)
            #    })

            #diff_hash = hashlib.md5(f"diff_multi_{ra}_{dec}_{size}_{band}_{len(matrices_2d)}".encode()).hexdigest()[:12]
            diff_hash = hashlib.md5(f"diff_v6_{ra}_{dec}_{size}_{band}_{len(matrices_2d)}".encode()).hexdigest()[:12]
            diff_filename = f"diff_{diff_hash}.png"
            diff_path = os.path.join("static/frames", diff_filename)

            limit = np.percentile(np.abs(diff_matrix), 98) + 1e-6
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            ax.imshow(diff_matrix, cmap="coolwarm", origin="lower", vmin=-limit, vmax=limit)

            for cand in diff_candidates:
                x0, y0 = cand["start_px"]
                x1, y1 = cand["end_px"]
                cid = cand["id"]

                ax.scatter([x0], [y0], s=150, facecolors="none", edgecolors="#00f2fe", linewidths=2.0, zorder=10)
                ax.scatter([x1], [y1], s=150, facecolors="none", edgecolors="#facc15", linewidths=2.0, zorder=10)

                if cand["shift_px"] > 0.5:
                    ax.annotate(
                        "", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2.0, shrinkA=5, shrinkB=5),
                        zorder=11
                    )
                ax.text(x1, min(min_h - 1, y1 + 1.0), f"#{cid}", color="#facc15", fontsize=8.5, fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0b0f19", alpha=0.85, edgecolor="none"), zorder=12)

            ax.set_xlim(-0.5, min_w - 0.5)
            ax.set_ylim(-0.5, min_h - 0.5)
            ax.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(diff_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            diff_url = f"/static/frames/{diff_filename}"

# Generar GIF animado descargable con Pillow
    gif_url = None
    if len(png_paths_for_gif) >= 2:
        gif_hash = hashlib.md5(f"gif_{ra}_{dec}_{size}_{band}_{len(png_paths_for_gif)}".encode()).hexdigest()[:12]
        gif_filename = f"timelapse_{gif_hash}.gif"
        gif_path = os.path.join("static/frames", gif_filename)
        try:
            pil_frames = [Image.open(p).convert("RGB") for p in png_paths_for_gif if os.path.exists(p)]
            if len(pil_frames) >= 2:
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=450,
                    loop=0
                )
                gif_url = f"/static/frames/{gif_filename}"
        except Exception:
            gif_url = None       
    return {
        "ra": ra,
        "dec": dec,
        "total_found": len(registros),
        "bands": sorted(list(bandas_disponibles)),
        "frames": frames_listos,
        "diff_image_url": diff_url,
        "diff_candidates": diff_candidates,
        "light_curve": light_curve,
        "gif_url": gif_url
    }
