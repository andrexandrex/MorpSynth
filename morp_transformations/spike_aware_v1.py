import numpy as np, cv2, matplotlib.pyplot as plt
from skimage.draw import polygon

# --------------------------------------------------------------------- #
# radial‑profile mirroring helper
# --------------------------------------------------------------------- #
from scipy.signal import savgol_filter, find_peaks
from skimage.morphology import erosion, disk
from skimage.morphology import binary_dilation
from sklearn.cluster import KMeans  
from skimage.measure import label         as sklabel
from skimage.draw import line as skline
import random
import os
def mirror_profile_expand(region_mask, row0, col0, n_row, n_col,
                          alpha=np.deg2rad(60),   # half‑fan angle (°/2)
                          n_rays=80,              # # directions in the fan
                          scale=1.2,              # outward = scale·inward
                          jitter=1,
                          max_radius=50,
                          debug = False):              # 0/1 px random jitter
    """
    Return a Boolean mask of new pixels produced by mirroring the inward
    radial profile (fan of angles) around (row0,col0) outwards.
    """
    #print('amen')
    H, W = region_mask.shape
    phi0 = np.arctan2(n_row, n_col)     
    thetas = np.linspace(-alpha, +alpha, n_rays)
    inward_dists = []
    out_rows, out_cols = [], []
    for th in thetas:
        # unit direction
        d_col =  np.cos(phi0 + th)
        d_row =  np.sin(phi0 + th)

        # ---- inward distance --------------------------------------------------
        di = 0
        while True:
            r = int(round(row0 - di*d_row))
            c = int(round(col0 - di*d_col))
            if r < 0 or r >= H or c < 0 or c >= W or region_mask[r, c] == 0:
                break
            di += 1
        inward_dists.append(di)


        # ---- outward distance -------------------------------------------------
        #noise = 0.2*np.random.randn()
        didi = int(scale * di + jitter)
        #do = min(didi, max_radius)
        if didi >= max_radius:
            # bring it back by 0‥30 % of max_radius with cosine easing
            ease = 0.7 + 0.3*np.random.rand()
            do   = int(max_radius * ease)
        else:
            do = didi   

        #do = int(scale * di * (1+noise))
        r_out = int(round(row0 + do*d_row))
        c_out = int(round(col0 + do*d_col))
        out_rows.append(r_out)
        out_cols.append(c_out)

    # close the polar polygon
    poly_rows = out_rows + [row0]
    poly_cols = out_cols + [col0]

    rr, cc = polygon(poly_rows, poly_cols, region_mask.shape)
    mask_out = np.zeros_like(region_mask, dtype=bool)
    mask_out[rr, cc] = True

    se = disk(2)                               # radius=2 pixels
    mask_closed  = cv2.morphologyEx(mask_out.astype(np.uint8),
                                cv2.MORPH_CLOSE,
                                se.astype(np.uint8))


    # ---- B. blur + threshold to round off the circle cut --------------
    blur = cv2.GaussianBlur(mask_closed*255, (0,0), sigmaX=1.5)
    mask_smooth = (blur > 100).astype(np.uint8)

    eroded = erosion(mask_smooth, disk(2))
    border = mask_smooth & (~eroded)

    
    # Randomly remove ~10% of border pixels
    br, bc = np.nonzero(border)
    num_holes = int(0.25 * len(br))
    if num_holes > 0:
        hole_idx = np.random.choice(len(br), size=num_holes, replace=False)
        mask_smooth[br[hole_idx], bc[hole_idx]] = 0

    # Final candidate = new pixels only
    candidate = mask_smooth.astype(bool) & (region_mask == 0)
    # ------------------------------------------------------------------
    #  B. extract the bulge contour & trace the two ribs
    # ------------------------------------------------------------------
    # (x  ,y  ) ≡ (col,row) —–––––––– NOTE: keep this convention everywhere
    #  d_out points *outwards* from the apex now ––––––––––––––––––––––––
    d_out = np.array([n_row, n_col], float)          # ✔  outward, not inward
    d_out /= (np.linalg.norm(d_out) + 1e-9)

    cnts, _ = cv2.findContours(candidate.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)

    # prepare debug containers so we can still plot something if no front arc

    debug_cnt  = None
    debug_tips = {"L": None, "R": None, "hitL": None, "hitR": None}

    for raw in cnts:
        pts = raw.squeeze(1)                 # (M,2) in (x,col , y,row)
        cnt = np.stack([pts[:, 1], pts[:, 0]], 1)      # (row,col)

        # keep only the points *in front* of the apex, w.r.t. d_out
        diffs = cnt - np.array([row0, col0])
        projs = diffs @ d_out                # signed distance along d_out
        front = projs > 0                    # strictly in front

        if not np.any(front):
            # remember the contour for debugging but keep looping
            debug_cnt = cnt
            continue

        # pick the two lateral extremes in the local (d_out, perp) frame
        perp  = np.array([ d_out[1], -d_out[0] ])      # 90° left of d_out
        sides = diffs @ perp                           # signed “horizontal”
        idxL  = np.where(front)[0][np.argmin(sides[front])]
        idxR  = np.where(front)[0][np.argmax(sides[front])]
        yL,xL = cnt[idxL];  yR,xR = cnt[idxR]

        vec_to_apex_L = np.array([row0 - yL, col0 - xL], float)
        vec_to_apex_L /= (np.linalg.norm(vec_to_apex_L) + 1e-9)

        d_in          = -d_out                         # old inward normal
        dir_vec_L     = d_in + vec_to_apex_L      # 50‑50 blend  → bisector
        dir_vec_L    /= (np.linalg.norm(dir_vec_L) + 1e-9)

        def trace(y0, x0, dir_vec=dir_vec_L):
            steps = max_radius + 5                    # safe upper bound
            for t in range(steps):
                y = int(round(y0 + t * dir_vec[0]))
                x = int(round(x0 + t * dir_vec[1]))
                if not (0 <= y < H and 0 <= x < W):
                    break
                if region_mask[y, x]:
                    rr, cc = skline(y0, x0, y, x)
                    good   = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    candidate[rr[good], cc[good]] = True
                    return (y, x)
            return None

        hitL = trace(yL, xL,dir_vec_L)     # *** existing call, now uses the new dir ***

        # Right tip ➜ inward hit  (re‑define trace again with its own bisector)
        vec_to_apex_R = np.array([row0 - yR, col0 - xR], float)
        vec_to_apex_R /= (np.linalg.norm(vec_to_apex_R) + 1e-9)

        dir_vec_R     = d_in + vec_to_apex_R
        dir_vec_R    /= (np.linalg.norm(dir_vec_R) + 1e-9)

        hitR = trace(yR, xR,dir_vec_R)     # *** existing call, now uses the new dir ***
        if hitL is not None:
            # triangle defined by apex → tip → hit
            poly_rows = [row0, yL, hitL[0]]
            poly_cols = [col0, xL, hitL[1]]
            rr_tri, cc_tri = polygon(poly_rows, poly_cols, region_mask.shape)
            candidate[rr_tri, cc_tri] = True

        # Right rib area
        if hitR is not None:
            poly_rows = [row0, yR, hitR[0]]
            poly_cols = [col0, xR, hitR[1]]
            rr_tri, cc_tri = polygon(poly_rows, poly_cols, region_mask.shape)
            candidate[rr_tri, cc_tri] = True
        # save for the final debug overlay
        debug_cnt           = cnt
        debug_tips["L"]     = (yL, xL)
        debug_tips["R"]     = (yR, xR)
        debug_tips["hitL"]  = hitL
        debug_tips["hitR"]  = hitR
        for tip_key, hit_key in (("L","hitL"), ("R","hitR")):
            tip = debug_tips[tip_key]
            hit = debug_tips[hit_key]
            if tip is not None and hit is not None:
                y0, x0 = tip
                y1, x1 = hit
                rr, cc = skline(y0, x0, y1, x1)
                valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                candidate[rr[valid], cc[valid]] = True

    # ------------------------------------------------------------------
    #  C. bullet‑proof debug plot – always shows *something*
    # ------------------------------------------------------------------
    if debug:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(region_mask, cmap='gray')
        ax.imshow(candidate, cmap='Reds', alpha=0.6)

        if debug_cnt is not None:
            ax.plot(debug_cnt[:, 1], debug_cnt[:, 0], '.', color='yellow', ms=2)
        for tag, pt in debug_tips.items():
            if pt is not None:
                y, x = pt
                ax.scatter(x, y,
                           s=80,
                           c={'L': 'lime', 'R': 'lime',
                              'hitL': 'cyan', 'hitR': 'cyan'}[tag],
                           label=tag)

        ax.scatter(col0, row0, c='black', s=50, label='apex')
        ax.quiver(col0, row0,
                  d_out[0], d_out[1],
                  angles='xy', scale_units='xy', scale=1,
                  width=0.005, headwidth=3, color='orange', label='d_out')

        ax.legend();  ax.axis('off');  plt.tight_layout();  plt.show()


    return candidate


def mirror_profile_modify(region_mask, row0, col0, n_row, n_col,
                          alpha=np.deg2rad(60), n_rays=80,
                          scale=1.2, jitter=1, max_radius=50,
                          mode="expand", debug=False):
    """
    mode='expand' -> use existing outward logic (returns add_mask, remove_mask=False)
    mode='shrink' -> carve an inward wedge (returns add_mask=False, remove_mask)
    """
    if mode == "expand":
        add = mirror_profile_expand(region_mask, row0, col0, n_row, n_col,
                                    alpha=alpha, n_rays=n_rays, scale=scale,
                                    jitter=jitter, max_radius=max_radius, debug=debug)
        return add.astype(bool), np.zeros_like(region_mask, bool)

    # --- inward "shrink" wedge (lightweight but consistent with your smoothing) ---
    H, W = region_mask.shape
    phi0 = np.arctan2(n_row, n_col)
    thetas = np.linspace(-alpha, +alpha, n_rays)

    in_rows, in_cols = [], []
    for th in thetas:
        d_col = np.cos(phi0 + th)
        d_row = np.sin(phi0 + th)

        # distance INSIDE the mask from apex
        di = 0
        while True:
            r = int(round(row0 - di*d_row))
            c = int(round(col0 - di*d_col))
            if r < 0 or r >= H or c < 0 or c >= W or (region_mask[r, c] == 0):
                break
            di += 1

        # how much to carve inward (can’t exceed di or max_radius)
        didi = int(scale * di + jitter)
        dd   = min(didi, max_radius, di)  # clamp so we stay inside
        r_in = int(round(row0 - dd * d_row))
        c_in = int(round(col0 - dd * d_col))
        in_rows.append(r_in); in_cols.append(c_in)

    # inward polygon: tips + apex
    poly_rows = in_rows + [row0]
    poly_cols = in_cols + [col0]
    rr, cc = polygon(poly_rows, poly_cols, region_mask.shape)

    mask_out = np.zeros_like(region_mask, dtype=bool)
    # only carve where we actually have region
    inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    mask_out[rr[inside], cc[inside]] = True
    mask_out &= region_mask.astype(bool)

    # smooth/close like your expand path
    se = disk(2)
    mask_closed = cv2.morphologyEx(mask_out.astype(np.uint8),
                                   cv2.MORPH_CLOSE, se.astype(np.uint8))
    blur = cv2.GaussianBlur(mask_closed*255, (0,0), sigmaX=1.5)
    mask_smooth = (blur > 100).astype(np.uint8)

    # optional raggedness (same spirit as expand)
    eroded = erosion(mask_smooth, disk(2))
    border = mask_smooth & (~eroded)
    br, bc = np.nonzero(border)
    num_holes = int(0.25 * len(br))
    if num_holes > 0:
        hole_idx = np.random.choice(len(br), size=num_holes, replace=False)
        mask_smooth[br[hole_idx], bc[hole_idx]] = 0

    remove = (mask_smooth.astype(bool) & region_mask.astype(bool))

    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].imshow(region_mask, cmap='gray'); ax[0].set_title('Original'); ax[0].axis('off')
        show = region_mask.copy().astype(np.uint8)
        show[remove] = 0
        ax[1].imshow(show, cmap='gray'); ax[1].set_title('After shrink (preview)'); ax[1].axis('off')
        plt.tight_layout(); plt.show()

    return np.zeros_like(region_mask, bool), remove














# --------------------------------------------------------------------- #
# main: apex‑based outward expansion with profile mirroring
# --------------------------------------------------------------------- #
def one_apex_outward_expand(region_mask, apices,
                            alpha_deg=60,
                            scale=1.2,max_radius = 50,
                            debug=False):
    """
    Grow a bulge from the apex farthest from centroid using
    radial‑profile mirroring (organic look).
    """
    print('JAAAAAAAAAAAAAAA')
    if len(apices) == 0:
        return region_mask.copy()

    # 1. choose farthest apex (apices are (col, row) pairs)
    rows, cols = np.nonzero(region_mask)
    cy, cx = rows.mean(), cols.mean()

    apex_cols = apices[:, 0]
    apex_rows = apices[:, 1]
    idx = np.argmax(np.hypot(apex_cols - cx, apex_rows - cy))
    col0, row0 = int(apex_cols[idx]), int(apex_rows[idx])

    # 2. outward normal via EDT gradient (with fallback)
    dist_bg = cv2.distanceTransform((~region_mask).astype(np.uint8),
                                    cv2.DIST_L2, 3)
    gy, gx = np.gradient(dist_bg.astype(float))
    n_row, n_col = gy[row0, col0], gx[row0, col0]
    if np.hypot(n_row, n_col) < 1e-3:           # fallback: centroid vector
        n_row, n_col = row0 - cy, col0 - cx
    nrm = np.hypot(n_row, n_col) + 1e-6
    n_row /= nrm; n_col /= nrm

    # 3. radial‑profile mirroring fan
    candidate = mirror_profile_expand(
        region_mask,
        row0, col0,
        n_row, n_col,
        alpha=np.deg2rad(alpha_deg),
        n_rays=60,
        scale=scale,
        max_radius=max_radius,
        jitter=1
    )

    grown = (region_mask | candidate).astype(np.uint8)

    # 4. debug plot
    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(region_mask, cmap='gray')
        axes[0].scatter([col0], [row0], c='red', s=60, edgecolors='white')
        axes[0].set_title("Original"); axes[0].axis('off')
        axes[1].imshow(candidate, cmap='OrRd')
        axes[1].scatter([col0], [row0], c='yellow', s=60, edgecolors='black')
        axes[1].set_title("New Pixels"); axes[1].axis('off')
        axes[2].imshow(grown, cmap='gray')
        axes[2].set_title("Grown"); axes[2].axis('off')
        plt.tight_layout(); plt.show()

    return grown

def contour_curvature_apices(region_mask,
                             smooth_window=11,        # must be odd
                             polyorder=3,
                             peak_prom_quantile=0.85,
                             min_separation_px=4,
                             radial_boost=0.5,
                             deriv_step=2,
                             debug=False):
    cnts, _ = cv2.findContours(region_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        # nothing to detect
        apices = np.empty((0, 2), dtype=int)
        return apices, np.array([]), np.empty((0,2), dtype=float)

    cnt = max(cnts, key=cv2.contourArea).squeeze(1)  # (N,2)  y,x
    if cnt.ndim != 2 or cnt.shape[0] < 3:
        # need at least 3 points to talk about curvature
        apices = np.empty((0, 2), dtype=int)
        return apices, np.array([]), cnt

    # ensure CCW orientation → outward curvature positive
    if cv2.contourArea(cnt, oriented=True) < 0:
        cnt = cnt[::-1]

    xs = cnt[:, 1].astype(float)
    ys = cnt[:, 0].astype(float)
    N  = len(xs)

    # ---- adapt smoothing window & polyorder safely ----
    def _largest_odd_leq(n):
        return n if n % 2 == 1 else n - 1

    # window must be odd and <= N; use up to N-1 to avoid degenerate periodic wrap
    w_max = max(3, _largest_odd_leq(N - 1))
    w     = min(smooth_window, w_max)
    if w < 3:  # still too small → bail out
        apices = np.empty((0, 2), dtype=int)
        return apices, np.array([]), cnt

    p = min(polyorder, max(1, w - 2))

    xs_s = savgol_filter(xs, w, p, mode='wrap')
    ys_s = savgol_filter(ys, w, p, mode='wrap')

    # ---- clamp derivative step and edge order constraints ----
    ds = max(1, min(int(deriv_step), max(1, N // 10)))
    # gradient needs at least (edge_order+1) points; keep edge_order=1
    if N < 2:
        apices = np.empty((0, 2), dtype=int)
        return apices, np.array([]), cnt

    dx  = np.gradient(xs_s, ds, edge_order=1)
    dy  = np.gradient(ys_s, ds, edge_order=1)
    ddx = np.gradient(dx,    ds, edge_order=1)
    ddy = np.gradient(dy,    ds, edge_order=1)

    denom = (dx*dx + dy*dy)**1.5 + 1e-6
    curv  = (dx*ddy - dy*ddx) / denom
    pos_curv = np.clip(curv, 0, None)

    if radial_boost > 0 and N > 2:
        cy, cx = ys_s.mean(), xs_s.mean()
        radial = np.hypot(ys_s - cy, xs_s - cx)
        pos_curv *= (1 + radial_boost *
                     (radial - radial.mean()) / (radial.std() + 1e-6))

    prom_thresh = np.quantile(pos_curv[pos_curv > 0], peak_prom_quantile) if np.any(pos_curv > 0) else 0
    peaks, _ = find_peaks(pos_curv, prominence=prom_thresh, distance=min_separation_px)

    apices = cnt[peaks] if len(peaks) else np.empty((0, 2), dtype=int)

    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(region_mask, cmap='gray')
        if len(apices):
            ax[0].scatter(apices[:, 0], apices[:, 1], c='red', s=16, label='apex')
        ax[0].set_title("Detected apices"); ax[0].axis('off')
        ax[1].plot(pos_curv, lw=0.8)
        if len(peaks):
            ax[1].scatter(peaks, pos_curv[peaks], c='red')
        ax[1].axhline(prom_thresh, ls='--', color='gray', lw=0.7)
        ax[1].set_xlabel("contour index"); ax[1].set_title("boosted outward curvature")
        plt.tight_layout(); plt.show()

    return apices, curv, cnt


def multi_apex_outward_expand(region_mask, apices,
                              n_apices=3,
                              alpha_deg=60,
                              scale=1.2,
                              max_radius=50,
                              debug=False):
    """
    Grow bulges from `n_apices` clusters of spikes (apices are (col,row)).
    """
    if len(apices) == 0:
        return region_mask.copy()

    # --- 1. blob centroid ----------------------------------------------------
    rows, cols = np.nonzero(region_mask)
    cy, cx = rows.mean(), cols.mean()

    # --- 2. cluster apices ---------------------------------------------------
    k = min(n_apices, len(apices))
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(apices)
    labels = km.labels_

    chosen = []
    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue
        # pick farthest from centroid within this cluster
        d = np.hypot(apices[idxs,0] - cx, apices[idxs,1] - cy)
        pick = idxs[np.argmax(d)]
        chosen.append(apices[pick])

    # --- 3. for each selected apex, grow a bulge -----------------------------
    candidate_union = np.zeros_like(region_mask, dtype=bool)
    dist_bg = cv2.distanceTransform((~region_mask).astype(np.uint8),
                                    cv2.DIST_L2, 3)
    gy, gx = np.gradient(dist_bg.astype(float))

    for col0, row0 in chosen:
        col0, row0 = int(col0), int(row0)
        n_row, n_col = gy[row0, col0], gx[row0, col0]
        if np.hypot(n_row, n_col) < 1e-3:
            n_row, n_col = row0 - cy, col0 - cx
        nrm = np.hypot(n_row, n_col) + 1e-6
        n_row, n_col = n_row/nrm, n_col/nrm

        candidate = mirror_profile_expand(
            region_mask,
            row0, col0,
            n_row, n_col,
            alpha=np.deg2rad(alpha_deg),
            n_rays=80,
            scale=scale,
            jitter=1,
            max_radius=max_radius,debug=False
        )
        candidate_union |= candidate.astype(bool)

    grown = (region_mask | candidate_union).astype(np.uint8)

    # --- 4. debug ------------------------------------------------------------
    if debug:
        fig, axes = plt.subplots(1,3, figsize=(15,5))

        # left: original + selected apices
        axes[0].imshow(region_mask, cmap='gray')
        xs, ys = zip(*chosen)
        axes[0].scatter(xs, ys, c='red', s=80,
                        edgecolors='white', label='apices')
        axes[0].set_title("Original + Apices")
        axes[0].axis('off'); axes[0].legend(loc='upper right')

        # middle: union of all candidate bulges
        axes[1].imshow(candidate_union, cmap='OrRd')
        axes[1].scatter(xs, ys, c='yellow', s=80,
                        edgecolors='black', label='apices')
        axes[1].set_title("New Pixels (Union)")
        axes[1].axis('off'); axes[1].legend(loc='upper right')

        # right: merged grown mask
        axes[2].imshow(grown, cmap='gray')
        axes[2].set_title("Grown Mask")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    return grown

def mirror_profile_shrink(region_mask, row0, col0, n_row, n_col,
                          alpha=np.deg2rad(60), n_rays=80,
                          scale=1.2, jitter=1, max_radius=50,
                          debug=False):
    """
    Carve an inward patch using the SAME front-arc + dual-rib logic as expand.
    Returns: remove_mask (bool)  [pixels to erase from region_mask]
    """
    H, W = region_mask.shape
    # outward & inward unit normals at the apex
    d_out = np.array([n_row, n_col], float)
    d_out /= (np.linalg.norm(d_out) + 1e-9)
    d_in  = -d_out

    # ---- A. inward fan polygon (like your current shrink) --------------------
    phi0 = np.arctan2(d_in[0], d_in[1])        # fan centered along d_in
    thetas = np.linspace(-alpha, +alpha, n_rays)

    in_rows, in_cols = [], []
    for th in thetas:
        d_col = np.cos(phi0 + th)              # (x, y) = (col, row)
        d_row = np.sin(phi0 + th)

        # walk INSIDE the region from apex
        di = 0
        while True:
            r = int(round(row0 + di*d_row))    # + because we go along d_in
            c = int(round(col0 + di*d_col))
            if r < 0 or r >= H or c < 0 or c >= W or (region_mask[r, c] == 0):
                break
            di += 1

        # how far to carve: scaled & clamped, and never exceeding di
        didi = int(scale * di + jitter)
        dd   = min(didi, max_radius, di)
        r_in = int(round(row0 + dd * d_row))
        c_in = int(round(col0 + dd * d_col))
        in_rows.append(r_in); in_cols.append(c_in)

    # inward polygon (tips + apex)
    poly_rows = in_rows + [row0]
    poly_cols = in_cols + [col0]
    rr, cc = polygon(poly_rows, poly_cols, region_mask.shape)

    remove = np.zeros_like(region_mask, dtype=bool)
    good   = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    remove[rr[good], cc[good]] = True
    remove &= region_mask.astype(bool)         # carve only current label

    # ---- B. find "front arc" of the removal candidate (same spirit as expand)
    cnts, _ = cv2.findContours(remove.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # For visualization / robustness
    debug_cnt  = None
    debug_tips = {"L": None, "R": None, "hitL": None, "hitR": None}

    for raw in cnts:
        pts = raw.squeeze(1)                   # (M,2) in (x=col, y=row)
        cnt = np.stack([pts[:, 1], pts[:, 0]], 1)      # (row,col)
        if cnt.ndim != 2 or len(cnt) < 3:
            continue

        # keep only the points in front w.r.t. d_in (we are cutting inward)
        diffs = cnt - np.array([row0, col0])
        projs = diffs @ d_in                   # signed distance along d_in
        front = projs > 0
        if not np.any(front):
            debug_cnt = cnt
            continue

        # L/R tips in the local (d_in, perp) frame
        perp  = np.array([ d_in[1], -d_in[0] ])
        sides = diffs @ perp
        idxL  = np.where(front)[0][np.argmin(sides[front])]
        idxR  = np.where(front)[0][np.argmax(sides[front])]
        yL,xL = cnt[idxL];  yR,xR = cnt[idxR]

        # Blend vectors so ribs taper naturally back to the body (toward outside)
        vec_to_apex_L = np.array([row0 - yL, col0 - xL], float)
        vec_to_apex_L /= (np.linalg.norm(vec_to_apex_L) + 1e-9)
        dir_vec_L = d_out + vec_to_apex_L       # like expand but swap roles
        dir_vec_L /= (np.linalg.norm(dir_vec_L) + 1e-9)

        vec_to_apex_R = np.array([row0 - yR, col0 - xR], float)
        vec_to_apex_R /= (np.linalg.norm(vec_to_apex_R) + 1e-9)
        dir_vec_R = d_out + vec_to_apex_R
        dir_vec_R /= (np.linalg.norm(dir_vec_R) + 1e-9)

        # trace ribs outward until we hit BACKGROUND: this draws tapered cuts
        def trace_to_bg(y0, x0, dir_vec):
            steps = max_radius + 5
            for t in range(steps):
                y = int(round(y0 + t * dir_vec[0]))
                x = int(round(x0 + t * dir_vec[1]))
                if not (0 <= y < H and 0 <= x < W):
                    break
                if not region_mask[y, x]:
                    rr2, cc2 = skline(y0, x0, y, x)
                    good2 = (rr2 >= 0) & (rr2 < H) & (cc2 >= 0) & (cc2 < W)
                    remove[rr2[good2], cc2[good2]] = True
                    return (y, x)
            return None

        hitL = trace_to_bg(yL, xL, dir_vec_L)
        hitR = trace_to_bg(yR, xR, dir_vec_R)

        # fill the two rib triangles to round the cut (like expand)
        if hitL is not None:
            poly_rows = [row0, yL, hitL[0]]
            poly_cols = [col0, xL, hitL[1]]
            rr_tri, cc_tri = polygon(poly_rows, poly_cols, region_mask.shape)
            remove[rr_tri, cc_tri] = True

        if hitR is not None:
            poly_rows = [row0, yR, hitR[0]]
            poly_cols = [col0, xR, hitR[1]]
            rr_tri, cc_tri = polygon(poly_rows, poly_cols, region_mask.shape)
            remove[rr_tri, cc_tri] = True

        debug_cnt          = cnt
        debug_tips["L"]    = (yL, xL)
        debug_tips["R"]    = (yR, xR)
        debug_tips["hitL"] = hitL
        debug_tips["hitR"] = hitR

    # ---- C. smoothing/imperfections to avoid straight-edged look -------------
    se = disk(2)
    closed = cv2.morphologyEx(remove.astype(np.uint8),
                              cv2.MORPH_CLOSE, se.astype(np.uint8))
    blur = cv2.GaussianBlur(closed*255, (0,0), sigmaX=1.5)
    remove_smooth = (blur > 100).astype(np.uint8)

    eroded = erosion(remove_smooth, disk(2))
    border = remove_smooth & (~eroded)
    br, bc = np.nonzero(border)
    num_holes = int(0.25 * len(br))
    if num_holes > 0:
        hole_idx = np.random.choice(len(br), size=num_holes, replace=False)
        remove_smooth[br[hole_idx], bc[hole_idx]] = 0

    remove_final = (remove_smooth.astype(bool) & region_mask.astype(bool))

    if debug:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(region_mask, cmap='gray')
        ax.imshow(remove_final, cmap='Reds', alpha=0.6)
        if debug_cnt is not None:
            ax.plot(debug_cnt[:, 1], debug_cnt[:, 0], '.', color='yellow', ms=2)
        for tag, pt in debug_tips.items():
            if pt is not None:
                y, x = pt
                ax.scatter(x, y,
                           s=80,
                           c={'L':'lime','R':'lime','hitL':'cyan','hitR':'cyan'}[tag],
                           label=tag)
        ax.scatter(col0, row0, c='black', s=50, label='apex')
        ax.quiver(col0, row0, d_out[0], d_out[1],
                  angles='xy', scale_units='xy', scale=1,
                  width=0.005, headwidth=3, color='orange', label='d_out')
        ax.legend(); ax.axis('off'); plt.tight_layout(); plt.show()

    return remove_final

from skimage.morphology import erosion, disk

def mirror_profile_shrink_rounded(region_mask, row0, col0, n_row, n_col,
                                  alpha=np.deg2rad(60),
                                  n_rays=120,             # denser = smoother
                                  scale=1.2,
                                  jitter=1,
                                  max_radius=50,
                                  root_frac=0.2,          # 0..0.4; % of inward dist to start cut
                                  min_root_px=2,          # clamp r_start ≥ this (avoid apex vertex)
                                  debug=False):
    """
    Carve an inward, *rounded* notch: annular sector between r_start and r_end
    along a fan centered on the inward normal. Returns a Boolean 'remove' mask.
    """
    H, W = region_mask.shape
    # outward/inward unit normals
    d_out = np.array([n_row, n_col], float); d_out /= (np.linalg.norm(d_out) + 1e-9)
    d_in  = -d_out

    # fan angles centered on inward
    phi0   = np.arctan2(d_in[0], d_in[1])  # (row, col) → atan2(y,x)
    thetas = np.linspace(-alpha, +alpha, n_rays)

    r_starts, r_ends = [], []
    pts_outer_rows, pts_outer_cols = [], []
    pts_inner_rows, pts_inner_cols = [], []

    # build per-ray inner/outer radii
    for th in thetas:
        d_col = np.cos(phi0 + th)
        d_row = np.sin(phi0 + th)

        # how far INSIDE the structure from apex (along +d_in)
        di = 0
        while True:
            r = int(round(row0 + di*d_row))
            c = int(round(col0 + di*d_col))
            if r < 0 or r >= H or c < 0 or c >= W or (region_mask[r, c] == 0):
                break
            di += 1

        # inward end (can’t exceed available interior or max_radius)
        didi = int(scale * di + jitter)
        r_end = int(min(didi, max_radius, di))
        if r_end <= 0:
            # nothing to carve on this ray
            continue

        # inward start: a fraction of the local thickness (avoid apex vertex)
        r_start = int(max(min_root_px, root_frac * di))
        r_start = min(r_start, r_end - 1)  # keep band non-empty

        # points for the annular sector (outer then inner)
        ro = int(round(row0 + r_end   * d_row)); co = int(round(col0 + r_end   * d_col))
        ri = int(round(row0 + r_start * d_row)); ci = int(round(col0 + r_start * d_col))

        pts_outer_rows.append(ro); pts_outer_cols.append(co)
        pts_inner_rows.append(ri); pts_inner_cols.append(ci)

    if len(pts_outer_rows) < 3:
        # no meaningful notch to carve
        return np.zeros_like(region_mask, dtype=bool)

    # polygon: outer arc → inner arc (reversed) → close
    poly_rows = pts_outer_rows + pts_inner_rows[::-1]
    poly_cols = pts_outer_cols + pts_inner_cols[::-1]

    rr, cc = polygon(poly_rows, poly_cols, region_mask.shape)
    remove = np.zeros_like(region_mask, dtype=bool)
    good   = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    remove[rr[good], cc[good]] = True
    remove &= region_mask.astype(bool)

    # smooth & de-angularize (same spirit as your expand)
    se     = disk(2)
    closed = cv2.morphologyEx(remove.astype(np.uint8), cv2.MORPH_CLOSE, se.astype(np.uint8))
    blur   = cv2.GaussianBlur(closed*255, (0,0), sigmaX=1.8)
    remove_smooth = (blur > 110).astype(np.uint8)

    # optional raggedness to avoid surgical edges
    eroded = erosion(remove_smooth, disk(2))
    border = remove_smooth & (~eroded)
    br, bc = np.nonzero(border)
    n_holes = int(0.20 * len(br))
    if n_holes > 0:
        pick = np.random.choice(len(br), size=n_holes, replace=False)
        remove_smooth[br[pick], bc[pick]] = 0

    return (remove_smooth.astype(bool) & region_mask.astype(bool))

def multi_apex_expand_mix(region_mask, apices,
                          n_apices=3,
                          alpha_deg=60,
                          # per-mode controls
                          scale_expand=1.2, scale_shrink=1.1,
                          max_radius_expand=50, max_radius_shrink=50,
                          p_expand=0.6,
                          n_rays=80,
                          debug=True,
                          deterministic_quota=False,
                          return_components=True,
                          debug_save_dir: str | None = None,
                          debug_prefix: str = "",):
    """
    Randomly EXPAND or SHRINK per selected apex.
    Returns final uint8 mask (0/1) with additions and removals applied.
    """
    if len(apices) == 0:
        return region_mask.copy().astype(np.uint8)

    # --- 1) centroid & cluster ---
    rows, cols = np.nonzero(region_mask)
    cy, cx = rows.mean(), cols.mean()

    k = min(int(n_apices), len(apices))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(apices)
    labels = km.labels_

    chosen = []
    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue
        d = np.hypot(apices[idxs,0] - cx, apices[idxs,1] - cy)  # farthest from centroid
        pick = idxs[np.argmax(d)]
        chosen.append(apices[pick])

    # --- 2) per-apex expand-or-shrink ---
    add_union    = np.zeros_like(region_mask, bool)
    remove_union = np.zeros_like(region_mask, bool)

    # deterministic expand/shrink quota
    modes = []
    if deterministic_quota:
        n_exp = int(round(p_expand * len(chosen)))
        modes = ["expand"] * n_exp + ["shrink"] * (len(chosen) - n_exp)
        random.shuffle(modes)

    # gradient of distance transform for normals
    dist_bg = cv2.distanceTransform((region_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    gy, gx = np.gradient(dist_bg.astype(float))

    for idx, (col0, row0) in enumerate(chosen):
        col0, row0 = int(col0), int(row0)

        # outward normal estimate
        n_row, n_col = gy[row0, col0], gx[row0, col0]
        if np.hypot(n_row, n_col) < 1e-3:
            n_row, n_col = row0 - cy, col0 - cx
        nrm = np.hypot(n_row, n_col) + 1e-6
        n_row, n_col = n_row/nrm, n_col/nrm

        mode = modes[idx] if modes else ("expand" if random.random() < p_expand else "shrink")

        if mode == "expand":
            add = mirror_profile_expand(
                region_mask, row0, col0, n_row, n_col,
                alpha=np.deg2rad(alpha_deg),
                n_rays=n_rays,
                scale=scale_expand,
                jitter=1,
                max_radius=max_radius_expand,
                debug=False
            ).astype(bool)
            remove = np.zeros_like(region_mask, bool)
        else:
            add = np.zeros_like(region_mask, bool)
            remove = mirror_profile_shrink_rounded(
                region_mask, row0, col0, n_row, n_col,
                alpha=np.deg2rad(alpha_deg),
                n_rays=max(100, n_rays),
                scale=scale_shrink,
                jitter=1,
                max_radius=max_radius_shrink,
                root_frac=0.2, min_root_px=2, debug=False
            )
        add_union    |= add
        remove_union |= remove

    out = region_mask.copy().astype(bool)
    out |= add_union
    out &= ~remove_union

    # --- 4) debug plotting (same style as outward) ---
    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3, figsize=(15,5))

        xs, ys = zip(*chosen) if len(chosen) > 0 else ([], [])

        # Original + apices
        axes[0].imshow(region_mask, cmap='gray')
        if len(xs) > 0:
            axes[0].scatter(xs, ys, c='red', s=80, edgecolors='white', label='apices')
            axes[0].legend(loc='upper right')
        axes[0].set_title("Original + Apices")

        # New pixels (EXPAND minus SHRINK)
        diff_union = (add_union & ~remove_union).astype(np.uint8)
        axes[1].imshow(diff_union, cmap='OrRd')
        if len(xs) > 0:
            axes[1].scatter(xs, ys, c='yellow', s=80, edgecolors='black')
        axes[1].set_title("New Pixels (Union)")

        # Final mask
        axes[2].imshow(out, cmap='gray')
        axes[2].set_title("Final Mask")

        plt.tight_layout()

        if debug_save_dir is not None:
            os.makedirs(debug_save_dir, exist_ok=True)
            out_path = os.path.join(debug_save_dir, f"{debug_prefix}_apex_mix.png")
            plt.savefig(out_path, dpi=150)
            print(f"Saved debug plot -> {out_path}")
        
        plt.show()

        plt.close(fig)
    # --- 5) return according to flag ---
    if return_components:
        return add_union.astype(np.uint8), remove_union.astype(np.uint8)
    return out.astype(np.uint8)