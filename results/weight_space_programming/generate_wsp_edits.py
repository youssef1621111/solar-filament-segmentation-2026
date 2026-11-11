import os
import zipfile
import pickle
import io
import hashlib
import json
import numpy as np

BASE_CKPT = '/home/fathyyoussef162/results/weight_space_programming/original_checkpoints/efficientnet_b0_fold0.pt'
OUT_DIR = '/home/fathyyoussef162/results/weight_space_programming/edited_checkpoints'
LOG_DIR = '/home/fathyyoussef162/results/weight_space_programming/edit_logs'
RECIPE_DIR = '/home/fathyyoussef162/results/weight_space_programming/proposed_edits'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RECIPE_DIR, exist_ok=True)

class MockTensor:
    def __init__(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        self.size = size
        self.stride = stride
        self.storage = storage

class MockStorage:
    def __init__(self, name, *args):
        self.name = name

class TorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return lambda storage, offset, size, stride, rg, hooks: MockTensor(storage, offset, size, stride, rg, hooks)
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return lambda data, rg, hooks: data
        if module == 'torch' and name.endswith('Storage'):
            return lambda *args: MockStorage(name, *args)
        if module.startswith('torch'):
            return lambda *args, **kwargs: f'{module}.{name}'
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return pid

def load_baseline():
    with zipfile.ZipFile(BASE_CKPT, 'r') as z:
        pkl_name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
        prefix = pkl_name.rsplit('/', 1)[0]
        sd = TorchUnpickler(io.BytesIO(z.read(pkl_name))).load()
        
        def get_tensor(t_name):
            t = sd[t_name]
            key = t.storage[2]
            return np.frombuffer(z.read(f'{prefix}/data/{key}'), dtype=np.float32).reshape(t.size).copy(), key

        b3, k_b3 = get_tensor('final_conv.3.bias')
        w3, k_w3 = get_tensor('final_conv.3.weight')
        w0, k_w0 = get_tensor('final_conv.0.weight')
        return prefix, (b3, k_b3), (w3, k_w3), (w0, k_w0)

# Edit implementations
def edit_l1_bias(b, w, w0):
    b_new = b.copy() - 0.15
    return b_new, w.copy(), w0.copy()

def edit_l1_damp(b, w, w0):
    w_new = w.copy()
    w_new[0, 4, 0, 0] = 0.0
    w_new[0, 13, 0, 0] = 0.0
    return b.copy(), w_new, w0.copy()

def edit_l1_inhib(b, w, w0):
    w_new = w.copy()
    for c in [0, 2, 3, 5, 6, 7, 11, 14]:
        w_new[0, c, 0, 0] *= 1.15
    return b.copy(), w_new, w0.copy()

def edit_l1_composite(b, w, w0):
    b_new = b.copy() - 0.15
    w_new = w.copy()
    w_new[0, 4, 0, 0] = 0.0
    w_new[0, 13, 0, 0] = 0.0
    for c in [0, 2, 3, 5, 6, 7, 11, 14]:
        w_new[0, c, 0, 0] *= 1.10
    return b_new, w_new, w0.copy()

def edit_l2_svd(b, w, w0):
    mat = w0.reshape(16, -1)
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    # Truncate to rank 14
    mat_trunc = np.dot(u[:, :14] * s[:14], vt[:14, :])
    w0_new = mat_trunc.reshape(16, 16, 3, 3).astype(np.float32)
    return b.copy(), w.copy(), w0_new

EDITS = [
    {
        'id': 'WSP_L1_bias_m15',
        'level': 1,
        'title': 'Level 1: Classification Bias Realignment (-0.15)',
        'hypothesis': 'Shifting final_conv.3.bias by -0.15 moves the default operating point into negative logit space, suppressing weak background activations and slashing false positives.',
        'fn': edit_l1_bias
    },
    {
        'id': 'WSP_L1_damp_noisy',
        'level': 1,
        'title': 'Level 1: Excitatory Noisy Channel Suppression (Ch 4, 13)',
        'hypothesis': 'Channels 4 and 13 have negligible positive weights (+0.033, +0.030) compared to primary filament channels (+0.187). Setting them to 0 prunes noisy excitatory leakage.',
        'fn': edit_l1_damp
    },
    {
        'id': 'WSP_L1_boost_inhib',
        'level': 1,
        'title': 'Level 1: Inhibitory Background Suppressor Amplification (x1.15)',
        'hypothesis': 'Channels with negative weights act as solar background suppressors. Multiplying their weights by 1.15 increases background rejection over quiet sun regions.',
        'fn': edit_l1_inhib
    },
    {
        'id': 'WSP_L1_composite',
        'level': 1,
        'title': 'Level 1: Composite Head Reprogramming (Bias -0.15 + Zero Ch 4,13 + Boost Inhib x1.10)',
        'hypothesis': 'Synergistic combination of negative bias shift, noisy channel pruning, and moderate inhibitory amplification to achieve maximal FP reduction without sacrificing TP recall.',
        'fn': edit_l1_composite
    },
    {
        'id': 'WSP_L2_svd_trunc',
        'level': 2,
        'title': 'Level 2: Decoder Noise SVD Truncation (Rank 14 in final_conv.0)',
        'hypothesis': 'In final_conv.0.weight (16x144), the 2 lowest singular components encode high-frequency granulation noise. Truncating to rank 14 filters noise while preserving structural edges.',
        'fn': edit_l2_svd
    }
]

def main():
    prefix, (orig_b3, k_b3), (orig_w3, k_w3), (orig_w0, k_w0) = load_baseline()
    print('Baseline loaded successfully.')
    
    manifest = []
    
    for edit in EDITS:
        edit_id = edit['id']
        print(f'Applying {edit_id}...')
        mod_b, mod_w, mod_w0 = edit['fn'](orig_b3, orig_w3, orig_w0)
        
        modified_keys = {}
        diff_entries = []
        
        if not np.array_equal(mod_b, orig_b3):
            modified_keys[f'{prefix}/data/{k_b3}'] = mod_b.tobytes()
            delta = mod_b - orig_b3
            diff_entries.append({
                'tensor': 'final_conv.3.bias',
                'shape': list(orig_b3.shape),
                'frobenius_delta': float(np.linalg.norm(delta)),
                'orig_mean': float(orig_b3.mean()),
                'mod_mean': float(mod_b.mean()),
                'delta_mean': float(delta.mean())
            })
            
        if not np.array_equal(mod_w, orig_w3):
            modified_keys[f'{prefix}/data/{k_w3}'] = mod_w.tobytes()
            delta = mod_w - orig_w3
            diff_entries.append({
                'tensor': 'final_conv.3.weight',
                'shape': list(orig_w3.shape),
                'frobenius_delta': float(np.linalg.norm(delta)),
                'orig_mean': float(orig_w3.mean()),
                'mod_mean': float(mod_w.mean()),
                'delta_mean': float(delta.mean())
            })
            
        if not np.array_equal(mod_w0, orig_w0):
            modified_keys[f'{prefix}/data/{k_w0}'] = mod_w0.tobytes()
            delta = mod_w0 - orig_w0
            diff_entries.append({
                'tensor': 'final_conv.0.weight',
                'shape': list(orig_w0.shape),
                'frobenius_delta': float(np.linalg.norm(delta)),
                'orig_mean': float(orig_w0.mean()),
                'mod_mean': float(mod_w0.mean()),
                'delta_mean': float(delta.mean())
            })
            
        ckpt_path = os.path.join(OUT_DIR, f'{edit_id}.pt')
        with zipfile.ZipFile(BASE_CKPT, 'r') as zin:
            with zipfile.ZipFile(ckpt_path, 'w', compression=zipfile.ZIP_STORED) as zout:
                for item in zin.infolist():
                    if item.filename in modified_keys:
                        data = modified_keys[item.filename]
                    else:
                        data = zin.read(item.filename)
                    zout.writestr(item, data)
                    
        h = hashlib.sha256()
        with open(ckpt_path, 'rb') as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        sha256_hash = h.hexdigest()
        fsize = os.path.getsize(ckpt_path)
        
        diff_report = {
            'edit_id': edit_id,
            'level': edit['level'],
            'title': edit['title'],
            'hypothesis': edit['hypothesis'],
            'checkpoint_file': f'{edit_id}.pt',
            'sha256': sha256_hash,
            'size_bytes': fsize,
            'size_mb': round(fsize / (1024 * 1024), 2),
            'tensor_modifications': diff_entries
        }
        
        # Save individual diff log
        with open(os.path.join(LOG_DIR, f'{edit_id}_diff.json'), 'w') as f:
            json.dump(diff_report, f, indent=2)
            
        manifest.append(diff_report)
        total_delta = sum(t['frobenius_delta'] for t in diff_entries)
        print(f'  Created {ckpt_path} (SHA256: {sha256_hash[:16]}..., Frob Delta: {total_delta:.4f})')
        
    with open(os.path.join(RECIPE_DIR, 'edits_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('\nAll 5 WSP edits completed and recorded successfully.')

if __name__ == '__main__':
    main()
