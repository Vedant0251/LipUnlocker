"""
Data Augmentation Script for DeepLipUnlocker
Multiplies your existing recordings 10x using realistic transformations
so the LSTM sees diverse variations and generalizes far better.
"""
import numpy as np
import glob
import os

AUGMENTS_PER_SAMPLE = 10

def augment_sequence(data):
    """Apply one random augmentation to a (N, 40) sequence."""
    choice = np.random.randint(0, 5)
    
    if choice == 0:
        # Gaussian coordinate noise (simulates slight head wobble)
        noise = np.random.normal(0, 0.005, data.shape)
        return data + noise

    elif choice == 1:
        # Time stretch / compress (simulate speaking faster or slower)
        factor = np.random.uniform(0.80, 1.20)
        n = data.shape[0]
        new_len = max(10, int(n * factor))
        indices = np.linspace(0, n - 1, new_len)
        stretched = np.array([data[int(i)] for i in indices])
        # Pad or trim back to original length
        if len(stretched) < n:
            pad = np.tile(stretched[-1], (n - len(stretched), 1))
            stretched = np.vstack([stretched, pad])
        return stretched[:n]

    elif choice == 2:
        # Temporal jitter (uniformly shift each frame's coordinates slightly)
        jitter = np.random.uniform(-0.01, 0.01, data.shape)
        return np.clip(data + jitter, -2.0, 2.0)

    elif choice == 3:
        # Random frame dropout (simulate blink/detection miss frames)
        result = data.copy()
        n_drop = np.random.randint(1, max(2, len(data) // 10))
        drop_idx = np.random.choice(len(data), n_drop, replace=False)
        for idx in drop_idx:
            neighbor = max(0, idx - 1)
            result[idx] = result[neighbor]  # Replace with previous frame
        return result

    elif choice == 4:
        # Scale perturbation (simulate slight zoom variance)
        scale = np.random.uniform(0.92, 1.08)
        return data * scale

    return data


def augment_folder(label):
    folder = f'data/{label}'
    files = glob.glob(f'{folder}/*.npy')
    
    if not files:
        print(f"No files found in {folder}. Skipping.")
        return 0

    print(f"\nAugmenting [{label}]: {len(files)} original files...")
    
    # Find highest existing sample number
    existing_nums = []
    for f in files:
        try:
            existing_nums.append(int(os.path.splitext(os.path.basename(f))[0]))
        except:
            pass
    next_num = max(existing_nums) + 1 if existing_nums else 100

    generated = 0
    for f in files:
        original = np.load(f)
        if original.ndim != 2 or original.shape[1] != 40:
            print(f"  Skipping {f}: unexpected shape {original.shape}")
            continue

        for _ in range(AUGMENTS_PER_SAMPLE):
            aug = augment_sequence(original)
            save_path = f'{folder}/{next_num}.npy'
            np.save(save_path, aug)
            next_num += 1
            generated += 1

    print(f"  Generated {generated} augmented files -> {folder}/")
    return generated


if __name__ == '__main__':
    print("=" * 50)
    print("DeepLipUnlocker — Data Augmentation Pipeline")
    print("=" * 50)

    total = 0
    total += augment_folder('password')
    total += augment_folder('random')

    print(f"\n✔ Done! Generated {total} augmented training sequences.")
    print("Now run: python train_model.py")
