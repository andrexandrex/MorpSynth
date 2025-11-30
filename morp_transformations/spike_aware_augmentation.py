# -------------------------------------------------------------------------
# 0. Imports   (place near the top of spike_aware_v1.py or your notebook)
# -------------------------------------------------------------------------
import numpy as np, cv2, math, random
import matplotlib.pyplot as plt
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import disk, erosion
from skimage.draw import polygon
import torch

label_to_rgb = {
    0: [0, 0, 0],        # Sea Surface
    1: [0, 255, 255],    # Oil Spill
    2: [255, 0, 0],      # Look-alike
    3: [153, 76, 0],     # Ship
    4: [0, 153, 0]       # Land
}

def translate_1d_to_rgb(mask_1d):
    """
    Converts a 1D mask (H, W) with values {0,1,...,4} into an RGB image.
    """
    h, w = mask_1d.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, rgb in label_to_rgb.items():
        mask_rgb[mask_1d == label] = rgb
    return mask_rgb

def jitter_edges_soft(mask_np, sigma=0.8, max_px=2, p=0.35, seed=None):
    """
    Light, normal-directed jitter on the boundary only.
    - sigma: std dev (px) of the displacement magnitude
    - max_px: cap on absolute displacement (px)
    - p: probability to jitter a given boundary point
    """
    import numpy as np, cv2
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    mask = (mask_np > 0).astype(np.uint8)
    H, W = mask.shape

    # boundary polyline(s)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)

    # signed distance → gradient gives (approx) outward normals
    # (positive inside, negative outside)
    dist_in  = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    sdf = dist_in - dist_out
    gy, gx = np.gradient(sdf.astype(np.float32))
    nrm = np.stack([gy, gx], axis=-1)
    nrm_norm = np.linalg.norm(nrm, axis=-1) + 1e-6
    nrm[..., 0] /= nrm_norm
    nrm[..., 1] /= nrm_norm

    for cnt in cnts:
        pts = cnt[:, 0, :]           # (N,2) as (x,y)
        if len(pts) < 3:
            continue
        pts = pts.astype(np.float32)
        # select a subset to jitter
        keep = rng.random(len(pts)) > p
        offs = rng.normal(0.0, sigma, size=(len(pts),))   # scalar along normal
        offs = np.clip(offs, -max_px, max_px)

        # sample normal at each point
        y = np.clip(pts[:, 1].round().astype(int), 0, H-1)
        x = np.clip(pts[:, 0].round().astype(int), 0, W-1)
        nxy = nrm[y, x]                     # (N,2) as (ny, nx)

        shift = np.stack([nxy[:, 1], nxy[:, 0]], axis=-1) * offs[:, None]  # (dx,dy)
        new_pts = np.where(keep[:, None], pts, pts + shift).round().astype(int)
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, W-1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, H-1)

        cv2.fillPoly(out, [new_pts.reshape(-1, 1, 2)], 1)

    return out.astype(np.uint8)
# -------------------------------------------------------------------------
# 1. Utility: tight rotation without pre‑computed bbox
# -------------------------------------------------------------------------
import random, math
import numpy as np
import cv2
def roughen_mask(binary, iters=(1,3), ksize=(3,7)):
    it = np.random.randint(iters[0], iters[1]+1)
    k  = np.random.randint(ksize[0], ksize[1]+1)
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # random order: erode then dilate or vice versa
    if np.random.rand() < 0.5:
        out = cv2.erode(binary, kernel, iterations=it)
        out = cv2.dilate(out, kernel, iterations=it)
    else:
        out = cv2.dilate(binary, kernel, iterations=it)
        out = cv2.erode(out, kernel, iterations=it)
    return (out > 0).astype(np.uint8)
# -----------------------------------------------------------------------------
# 1. rotate_region_tight → allow any angle in [0,360)
# -----------------------------------------------------------------------------
def rotate_region_no_crop_with_info(region_mask, angle=None):
    """
    Rota region_mask (uint8 0/1) en un canvas L×L de forma que no se pierda
    ningún píxel. Devuelve:
      - rot   : la máscara rotada (L×L, uint8 0/1)
      - angle : ángulo usado (grados CCW) (float)
      - M     : matriz 2×3 de cv2.getRotationMatrix2D
      - (cx, cy): centro del canvas (enteros)
      - (y0, x0, h, w, top, left, L): parámetros internos para mapear coordenadas
    """
    if angle is None:
        angle = random.uniform(0, 360)

    ys, xs = np.nonzero(region_mask)
    if len(ys) == 0:
        # Región vacía corner‐case
        return region_mask.copy(), angle, None, (0,0), (0,0,0,0,0,0,0)

    # 1A) tight‐crop para saber h,w,y0,x0
    y0, x0 = int(ys.min()), int(xs.min())
    y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
    crop = region_mask[y0:y1, x0:x1].astype(np.uint8)
    h, w = crop.shape

    # 1B) creamos canvas L×L
    L = int(math.ceil(math.hypot(h, w)))
    canvas = np.zeros((L, L), dtype=np.uint8)

    # 1C) colocamos crop centrado en canvas
    cy, cx = L // 2, L // 2
    top = cy - (h // 2)
    left = cx - (w // 2)
    canvas[top: top+h, left: left+w] = crop

    # 1D) rotamos con cv2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (L, L),
                         flags=cv2.INTER_NEAREST,
                         borderValue=0)

    return rot, angle, M, (cx, cy), (y0, x0, h, w, top, left, L)


# ------------------------------------------------------------------------------------------------
# 2) Función auxiliar para dividir un array 1D de índices en “runs” contiguos
# ------------------------------------------------------------------------------------------------
def _split_runs(idxs):
    """
    Dado un vector ordenado “idxs” (por ejemplo [2,3,4, 10,11]),
    devuelve [array([2,3,4]), array([10,11])].
    """
    if len(idxs) == 0:
        return []
    gaps = np.where(np.diff(idxs) != 1)[0] + 1
    return np.split(idxs, gaps)


# ------------------------------------------------------------------------------------------------
# 3) Función que, sobre una máscara YA ROTADA (L×L), añade bulges en los flancos rectos
# ------------------------------------------------------------------------------------------------
def _add_bulge_flat_sides_on_rotated(rotated_mask,
                                     alpha_deg=60,
                                     scale=1.2,
                                     max_radius=30):
    """
    Dada `rotated_mask` (uint8 0/1, tamaño L×L), busca todos los tramos
    de píxeles=1 contiguos que tocan cualquiera de los 4 bordes
    (fila=0, fila=L-1, col=0, col=L-1). Para cada tramo, calcula su punto medio
    (row_mid,col_mid) y la normal hacia “afuera” (arriba/abajo/izq/der), y llama
    a mirror_profile_expand en ese punto para crear un “mini-bulge”. 
    Devuelve una nueva máscara uint8 0/1 L×L con todos los píxeles añadidos.
    """
    H, W = rotated_mask.shape
    result = rotated_mask.copy().astype(np.uint8)

    # Importamos mirror_profile_expand (asegúrate de que esté disponible)
    from spike_aware_v1 import mirror_profile_expand

    # 3A) Lado izquierdo (col = 0)
    filas = np.where(rotated_mask[:, 0] == 1)[0]
    for run in _split_runs(filas):
        mid_r = int(round(run.mean()))
        mid_c = 0
        n_row, n_col = 0.0, -1.0  # normal hacia la izquierda
        bulge = mirror_profile_expand(
            rotated_mask,
            mid_r, mid_c,
            n_row, n_col,
            alpha=np.deg2rad(alpha_deg),
            n_rays=60,
            scale=scale,
            jitter=1,
            max_radius=max_radius,
            debug=False
        ).astype(bool)
        result |= bulge.astype(np.uint8)

    # 3B) Lado derecho (col = W-1)
    filas = np.where(rotated_mask[:, W - 1] == 1)[0]
    for run in _split_runs(filas):
        mid_r = int(round(run.mean()))
        mid_c = W - 1
        n_row, n_col = 0.0, +1.0  # normal hacia la derecha
        bulge = mirror_profile_expand(
            rotated_mask,
            mid_r, mid_c,
            n_row, n_col,
            alpha=np.deg2rad(alpha_deg),
            n_rays=60,
            scale=scale,
            jitter=1,
            max_radius=max_radius,
            debug=False
        ).astype(bool)
        result |= bulge.astype(np.uint8)

    # 3C) Lado superior (fila = 0)
    cols = np.where(rotated_mask[0, :] == 1)[0]
    for run in _split_runs(cols):
        mid_c = int(round(run.mean()))
        mid_r = 0
        n_row, n_col = -1.0, 0.0  # normal hacia arriba
        bulge = mirror_profile_expand(
            rotated_mask,
            mid_r, mid_c,
            n_row, n_col,
            alpha=np.deg2rad(alpha_deg),
            n_rays=60,
            scale=scale,
            jitter=1,
            max_radius=max_radius,
            debug=False
        ).astype(bool)
        result |= bulge.astype(np.uint8)

    # 3D) Lado inferior (fila = H-1)
    cols = np.where(rotated_mask[H - 1, :] == 1)[0]
    for run in _split_runs(cols):
        mid_c = int(round(run.mean()))
        mid_r = H - 1
        n_row, n_col = +1.0, 0.0  # normal hacia abajo
        bulge = mirror_profile_expand(
            rotated_mask,
            mid_r, mid_c,
            n_row, n_col,
            alpha=np.deg2rad(alpha_deg),
            n_rays=60,
            scale=scale,
            jitter=1,
            max_radius=max_radius,
            debug=False
        ).astype(bool)
        result |= bulge.astype(np.uint8)

    return result.astype(np.uint8)


# ------------------------------------------------------------------------------------------------
# 4) Función “flataware” final: detecta flancos antes de rotar, rota usando M y mapea
#    punto medio + normal, luego aplica mirror_profile_expand sobre la máscara rotada
# ------------------------------------------------------------------------------------------------
import numpy as np

def _safe_paste(dest: np.ndarray,
                src: np.ndarray,
                top: int,
                left: int,
                label_value: int,
                allowed_labels=(0, 3)):
    """
    Paste a 0/1 mask `src` into `dest` at [top:left], with:
      - cropping to frame bounds,
      - collision policy: only paint where dest==0, and abort if any src==1 hits a disallowed label.

    Returns True if paste occurred; False if blocked or fully out of bounds.
    """
    Hf, Wf = dest.shape
    h, w   = src.shape

    # Fully outside?
    if top >= Hf or left >= Wf or (top + h) <= 0 or (left + w) <= 0:
        return False

    ys0 = max(0, -top)
    ys1 = min(h, Hf - top)
    xs0 = max(0, -left)
    xs1 = min(w, Wf - left)
    if ys1 <= ys0 or xs1 <= xs0:
        return False

    yd0 = max(0, top)
    xd0 = max(0, left)

    src_crop  = src[ys0:ys1, xs0:xs1]
    dest_crop = dest[yd0:yd0 + (ys1 - ys0), xd0:xd0 + (xs1 - xs0)]

    # Collision check: any src==1 over labels NOT in allowed?
    # (allowed = {0, label_value, 3} typically; we check against disallowed)
    allowed = set(allowed_labels) | {int(label_value)}
    if np.any((src_crop == 1) & (~np.isin(dest_crop, list(allowed)))):
        return False

    # Paint only background
    paint = (src_crop == 1) & (dest_crop == 0)
    if np.any(paint):
        dest_crop[paint] = int(label_value)
        dest[yd0:yd0 + (ys1 - ys0), xd0:xd0 + (xs1 - xs0)] = dest_crop
        return True

    # Nothing to paint (all overlapped allowed non-zero)
    return False

import cv2, numpy as np

def jitter_edges(mask_np, max_offset=8):
    # mask_np: binary mask (0/label)
    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_np)
    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) != 2: 
            continue
        # jitter each vertex
        jittered = cnt + np.random.randint(-max_offset, max_offset+1, cnt.shape)
        jittered = np.clip(jittered, 0, mask_np.shape[0]-1)
        cv2.fillPoly(out, [jittered.reshape(-1,1,2)], 1)
    return out

def rotate_and_place_region_flataware2(
    sample_mask,
    region_info,
    label_value,
    max_shift=30,
    attempts=20,
    n_rays=20,
    debug=True
):
    """
    1) extrae binary0 = region_info['binary_mask'] (uint8 0/1, tamaño H0×W0).
    2) detecta tramos rectos *en esa binary0* (antes de rotar):  
         • recorre bordes (col=0, col=W0-1, fila=0, fila=H0-1), agrupa corridas contiguas,  
         • para cada run => calcula punto medio (row_mid, col_mid) y normal (n_row,n_col).  
       Guarda lista `flat_list = [(row_mid, col_mid, n_row, n_col), ...]`.  
    3) llama a rotate_region_no_crop_with_info(binary0) ⇒  
         rot_bitmap (L×L), angle, M, (cx,cy), (y0,x0,h,w,top,left,L).  
    4) por cada (row_mid,col_mid,n_row,n_col) en flat_list:  
         a) mapeo a coordenadas del canvas antes de rotar:  
            row_can = top + (row_mid - y0),  col_can = left + (col_mid - x0)  
         b) rotar ese punto usando M:  
            col_rot = M[0,0]*col_can + M[0,1]*row_can + M[0,2]  
            row_rot = M[1,0]*col_can + M[1,1]*row_can + M[1,2]  
         c) rotar la normal:  
            • head_can = (col_can + n_col, row_can + n_row)  
            • head_rot = (M[0,0]*head_x + M[0,1]*head_y + M[0,2],  
                          M[1,0]*head_x + M[1,1]*head_y + M[1,2])  
            • n_col_rot = head_rot_x − col_rot,  n_row_rot = head_rot_y − row_rot  
            • normalizar (n_row_rot,n_col_rot)  
         d) llamar mirror_profile_expand(rot_bitmap, row_rot, col_rot,  
                                        n_row_rot, n_col_rot, ...) y unionar resultado.  
    5) rot_bulged = rot_bitmap ∪ bulges_union.  
    6) pegar rot_bulged en “after” con desplazamiento polar (igual que antes).  
    7) fallback: si no placed, restaurar rot_bulged en sitio original.  
    8) debug: dibujar “Before” vs “After”.  
    """
    before = sample_mask.clone().numpy()
    after  = before.copy()

    # 1) extracción de binary0
    binary0 = region_info['binary_mask'].astype(np.uint8)
    H0, W0 = binary0.shape

    # 2) detectar flancos rectos ANTES de rotar
    flat_list = []  # [(row_mid, col_mid, n_row, n_col), ...]
    # 2A) lado izquierdo col=0
    filas = np.where(binary0[:, 0] == 1)[0]
    for run in _split_runs(filas):
        mid_r = int(round(run.mean()))
        mid_c = 0
        flat_list.append((mid_r, mid_c, 0.0, -1.0))
    # 2B) lado derecho col=W0-1
    filas = np.where(binary0[:, W0 - 1] == 1)[0]
    for run in _split_runs(filas):
        mid_r = int(round(run.mean()))
        mid_c = W0 - 1
        flat_list.append((mid_r, mid_c, 0.0, +1.0))
    # 2C) lado superior fila=0
    cols = np.where(binary0[0, :] == 1)[0]
    for run in _split_runs(cols):
        mid_c = int(round(run.mean()))
        mid_r = 0
        flat_list.append((mid_r, mid_c, -1.0, 0.0))
    # 2D) lado inferior fila=H0-1
    cols = np.where(binary0[H0 - 1, :] == 1)[0]
    for run in _split_runs(cols):
        mid_c = int(round(run.mean()))
        mid_r = H0 - 1
        flat_list.append((mid_r, mid_c, +1.0, 0.0))

    # 3) borramos la region original de “after”
    after[binary0 == 1] = 0

    # 4) rotamos sin tight-crop
    rot_bitmap, angle, M, (cx, cy), (y0, x0, h, w, top, left, L) = \
        rotate_region_no_crop_with_info(binary0, angle=None)

    # 5) aplicamos bulges en la máscara rotada
    bulges_union = np.zeros_like(rot_bitmap, dtype=np.uint8)

    for (r_mid, c_mid, n_r, n_c) in flat_list:
        # 5A) punto medio en canvas ANTES de rotar
        row_rel = r_mid - y0
        col_rel = c_mid - x0
        row_can = top + row_rel
        col_can = left + col_rel

        # 5B) rotamos el punto con la matriz M
        col_rot = int(round(M[0,0] * col_can + M[0,1] * row_can + M[0,2]))
        row_rot = int(round(M[1,0] * col_can + M[1,1] * row_can + M[1,2]))

        # 5C) rotamos la normal usando M: 
        head_can_x = col_can + n_c
        head_can_y = row_can + n_r
        head_rot_x = M[0,0] * head_can_x + M[0,1] * head_can_y + M[0,2]
        head_rot_y = M[1,0] * head_can_x + M[1,1] * head_can_y + M[1,2]

        n_col_rot = head_rot_x - col_rot
        n_row_rot = head_rot_y - row_rot
        norm = math.hypot(n_row_rot, n_col_rot) + 1e-9
        n_row_rot /= norm
        n_col_rot /= norm

        # 5D) invoco mirror_profile_expand en el punto rotado
        from spike_aware_v1 import mirror_profile_expand
        bulge = mirror_profile_expand(
            rot_bitmap,
            int(round(row_rot)), int(round(col_rot)),
            n_row_rot, n_col_rot,
            alpha=np.deg2rad(60),
            n_rays=n_rays,
            scale=1,
            jitter=1,
            max_radius=20,
            debug=False
        ).astype(bool)
        bulges_union |= bulge.astype(np.uint8)

    # 6) combinamos rot_bitmap ∪ bulges_union
    rot_bulged = ((rot_bitmap == 1) | (bulges_union == 1)).astype(np.uint8)

    rot_bulged = jitter_edges_soft(rot_bulged, sigma=0.6, max_px=5, p=0.25).astype(np.uint8)
    rot_bulged = roughen_mask(rot_bulged, iters=(1,2), ksize=(3,5))

    # 7) pegamos rot_bulged en la máscara global con desplazamiento polar
    ys_rb, xs_rb = np.nonzero(rot_bulged)
    if len(ys_rb) == 0:
        return sample_mask.clone()

    r0_rb, c0_rb = ys_rb.mean(), xs_rb.mean()
    Hf, Wf       = after.shape
    h_rb, w_rb   = rot_bulged.shape
    ry_rb, rx_rb = np.argwhere(rot_bulged).mean(axis=0)

    placed = False
    for _ in range(attempts):
        θ = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, max_shift)
        dy, dx = r * math.sin(θ), r * math.cos(θ)

        top   = int(round(r0_rb + dy - ry_rb))
        left  = int(round(c0_rb + dx - rx_rb))
        if _safe_paste(after, rot_bulged, top, left, label_value, allowed_labels=(0, 3)):
            placed = True
            break

    # 8) fallback
    if not placed:
        # Center-align original region (binary0) with the rotated+bulged mask
        ys0_b, xs0_b = np.nonzero(binary0)
        if len(ys0_b) > 0:
            cy0, cx0 = ys0_b.mean(), xs0_b.mean()   # original region center (full-frame)
            ys1_b, xs1_b = np.nonzero(rot_bulged)
            if len(ys1_b) > 0:
                cy1, cx1 = ys1_b.mean(), xs1_b.mean()
                top  = int(round(cy0 - cy1))
                left = int(round(cx0 - cx1))
                _ = _safe_paste(after, rot_bulged, top, left, label_value, allowed_labels=(0, 3))
        # else: nothing to restore when original is empty

    # 9) debug plot
    if debug:
        try:
            from dataset import translate_1d_to_rgb
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(translate_1d_to_rgb(before)); ax[0].set_title("Before"); ax[0].axis("off")
            ax[1].imshow(translate_1d_to_rgb(after));  ax[1].set_title("After");  ax[1].axis("off")
            plt.tight_layout(); plt.show()
        except Exception:
            pass
     
    return torch.from_numpy(after)
