#!/usr/bin/env python3
"""
Diagnostic script to test collinear spine reconnection candidates
"""
import os
import sys
sys.path.insert(0, '/home/fathyyoussef162')
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
import pycocotools.mask as mask_util
from scripts.utils_metrics import compute_mask_bbox
import time

def find_endpoints_and_tangents(mask, pad=6, k_trace=10, min_branch_len=5):
    """
    Given a binary mask of an instance, extract medial spine endpoints and tangent angles.
    """
    y_min, y_max, x_min, x_max = compute_mask_bbox(mask)
    if y_max < y_min or x_max < x_min:
        return []
    y0 = max(0, y_min - pad)
    y1 = min(2048, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(2048, x_max + pad + 1)
    
    crop = mask[y0:y1, x0:x1]
    skel = skeletonize(crop)
    if not np.any(skel):
        return []
        
    neigh_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neigh = ndi.convolve(skel.astype(int), neigh_kernel, mode='constant')
    
    # Endpoints have exactly 1 neighbor in 8-connectivity
    eps = np.argwhere((skel > 0) & (neigh == 1))
    
    # If no endpoints (e.g. isolated loop), fall back to extreme points
    if len(eps) == 0:
        pts = np.argwhere(skel > 0)
        if len(pts) >= 2:
            eps = np.array([pts[0], pts[-1]])
        elif len(pts) == 1:
            eps = np.array([pts[0]])
        else:
            return []
            
    endpoints_info = []
    
    for ep in eps:
        curr = tuple(ep)
        visited = {curr}
        path = [curr]
        
        # Walk along skeleton
        for _ in range(k_trace):
            cy, cx = curr
            found = False
            # Check 8-neighborhood
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                        if skel[ny, nx] and (ny, nx) not in visited:
                            visited.add((ny, nx))
                            path.append((ny, nx))
                            curr = (ny, nx)
                            found = True
                            break
                if found:
                    break
            if not found:
                break
                
        # If branch is too short and meets a branch point (neighbor > 2), it's a minor spur
        if len(path) < min_branch_len and len(eps) > 2:
            # check if end of path is a junction
            if len(path) > 1 and neigh[path[-1]] > 2:
                continue
                
        # Tangent vector: pointing OUTWARDS from interior of ribbon towards the endpoint
        p_tip = np.array([ep[0] + y0, ep[1] + x0], dtype=float)
        p_in = np.array([path[-1][0] + y0, path[-1][1] + x0], dtype=float)
        vec_out = p_tip - p_in # vector pointing out of ribbon tip
        
        norm = np.linalg.norm(vec_out)
        if norm > 0:
            unit_vec = vec_out / norm
            # Angle in degrees [-180, 180]
            angle_deg = np.degrees(np.arctan2(vec_out[0], vec_out[1]))
        else:
            unit_vec = np.zeros(2)
            angle_deg = 0.0
            
        endpoints_info.append({
            'point': p_tip,
            'unit_vec_out': unit_vec,
            'angle_deg': angle_deg,
            'path_len': len(path)
        })
        
    return endpoints_info

def main():
    df = pd.read_csv('results/weight_space_programming/validation_results/submission_clean_zenith_1324.csv')
    df['stem'] = df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])
    
    stems = sorted(df['stem'].unique())
    print(f"Total active stems: {len(stems)}, Total rows: {len(df)}")
    
    total_candidates = []
    
    for stem in stems:
        sub = df[df['stem'] == stem].reset_index(drop=True)
        if len(sub) < 2:
            continue
            
        masks = [mask_util.decode({'size': [2048, 2048], 'counts': r.encode('ascii')}) > 0 for r in sub['segmentation_rle']]
        ep_list = [find_endpoints_and_tangents(m) for m in masks]
        
        for i in range(len(sub)):
            eps_i = ep_list[i]
            for j in range(i + 1, len(sub)):
                eps_j = ep_list[j]
                
                best_pair = None
                min_dist = float('inf')
                
                for ei in eps_i:
                    for ej in eps_j:
                        d = np.linalg.norm(ei['point'] - ej['point'])
                        if d < min_dist:
                            min_dist = d
                            best_pair = (ei, ej)
                            
                if min_dist <= 15.0 and best_pair is not None:
                    ei, ej = best_pair
                    # Check collinearity
                    # For two collinear lines, their orientation angle difference is:
                    # diff_mod180
                    ang_diff_180 = abs(ei['angle_deg'] - ej['angle_deg']) % 180
                    if ang_diff_180 > 90:
                        ang_diff_180 = 180 - ang_diff_180
                        
                    # Also vector between endpoints:
                    bridge_vec = ej['point'] - ei['point']
                    bridge_norm = np.linalg.norm(bridge_vec)
                    bridge_unit = bridge_vec / bridge_norm if bridge_norm > 0 else np.zeros(2)
                    
                    # Direction outward from ei towards ej:
                    # ei['unit_vec_out'] should align with bridge_unit
                    # ej['unit_vec_out'] should align with -bridge_unit (pointing towards ei)
                    dot_bridge_i = np.dot(ei['unit_vec_out'], bridge_unit)
                    dot_bridge_j = np.dot(ej['unit_vec_out'], -bridge_unit)
                    
                    # Angle between outward vector of i and bridge vector
                    if bridge_norm > 0:
                        cos_i = np.clip(dot_bridge_i, -1.0, 1.0)
                        angle_i_bridge = np.degrees(np.arccos(cos_i))
                        cos_j = np.clip(dot_bridge_j, -1.0, 1.0)
                        angle_j_bridge = np.degrees(np.arccos(cos_j))
                    else:
                        angle_i_bridge = 0.0
                        angle_j_bridge = 0.0
                        
                    total_candidates.append({
                        'stem': stem,
                        'inst_i': sub.loc[i, 'filament_id'],
                        'inst_j': sub.loc[j, 'filament_id'],
                        'area_i': masks[i].sum(),
                        'area_j': masks[j].sum(),
                        'dist': min_dist,
                        'ang_diff_180': ang_diff_180,
                        'angle_i': ei['angle_deg'],
                        'angle_j': ej['angle_deg'],
                        'dot_bridge_i': dot_bridge_i,
                        'dot_bridge_j': dot_bridge_j,
                        'angle_i_bridge': angle_i_bridge,
                        'angle_j_bridge': angle_j_bridge,
                        'pt_i': ei['point'],
                        'pt_j': ej['point']
                    })
                    
    df_cand = pd.DataFrame(total_candidates)
    print(f"\nTotal candidate pairs within 15px: {len(df_cand)}")
    if len(df_cand) > 0:
        print("\nCandidate pairs within 12px:")
        sub12 = df_cand[df_cand['dist'] <= 12.0]
        print(f"Count: {len(sub12)}")
        print(sub12[['stem', 'inst_i', 'inst_j', 'area_i', 'area_j', 'dist', 'ang_diff_180', 'angle_i_bridge', 'angle_j_bridge']].to_string())
        
        print("\nCandidate pairs with dist <= 12px AND ang_diff_180 <= 25 deg:")
        passed = sub12[sub12['ang_diff_180'] <= 25.0]
        print(f"Count: {len(passed)}")
        print(passed[['stem', 'inst_i', 'inst_j', 'area_i', 'area_j', 'dist', 'ang_diff_180', 'angle_i_bridge', 'angle_j_bridge']].to_string())

if __name__ == "__main__":
    main()
