import numpy as np


def dice(pred, target):
    p = np.asarray(pred) > 0.5
    t = np.asarray(target) > 0.5
    if p.shape != t.shape:
        raise ValueError(f'Dice: shape mismatch {p.shape} / {t.shape}')
    n = int(p.sum()) + int(t.sum())
    return (2.0 * np.logical_and(p, t).sum() / n) if n else 1.0


def harm(gain, delta=0.02):
    return bool(gain < -delta)


def summarize_cases(rows, delta=0.02):
    if not rows:
        return {}
    gains = np.asarray([r['final_dice'] - r['initial_dice'] for r in rows], dtype=float)
    active = np.asarray([r['active_repairs'] > 0 for r in rows], dtype=bool)
    h = gains < -delta
    return {
        'index_records': len(rows),
        'unique_patients': len({r['patient_id'] for r in rows}),
        'mean_initial_dice': float(np.mean([r['initial_dice'] for r in rows])),
        'mean_final_dice': float(np.mean([r['final_dice'] for r in rows])),
        'mean_gain': float(gains.mean()),
        'harm_events_population': int(h.sum()),
        'harm_rate_population': float(h.mean()),
        'harm_events_active': int((h & active).sum()),
        'active_repairs_cases': int(active.sum()),
        'harm_rate_active': float((h & active).sum() / active.sum()) if active.any() else None,
        'any_negative_rate': float((gains < 0).mean()),
        'accepted_coverage': float(np.mean([not r['referred'] for r in rows])),
        'referrals': int(sum(r['referred'] for r in rows)),
        'total_action_calls': int(sum(r['active_repairs'] for r in rows)),
        'mean_action_calls': float(np.mean([r['active_repairs'] for r in rows])),
    }


def paired_patient_bootstrap(a, b, delta=0.02, repeats=1000, seed=2026):
    """Cluster paired harm-rate difference, A minus B. Same patient identities required."""
    def patient_harms(rows):
        by = {}
        for r in rows:
            by.setdefault(r['patient_id'], []).append((r['final_dice'] - r['initial_dice']) < -delta)
        return {k: int(any(v)) for k, v in by.items()}
    ah, bh = patient_harms(a), patient_harms(b)
    if set(ah) != set(bh):
        raise ValueError('Paired bootstrap requires identical patient sets')
    patients = sorted(ah)
    if not patients:
        raise ValueError('Empty set')
    diffs = np.asarray([ah[p] - bh[p] for p in patients], dtype=float)
    rng = np.random.default_rng(seed)
    sims = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(repeats)])
    return {'patients': len(patients), 'paired_difference_A_minus_B': float(diffs.mean()),
            'ci95_percentile': [float(x) for x in np.quantile(sims,[.025,.975])],
            'discordant_A_harm_only': int(((diffs == 1)).sum()),
            'discordant_B_harm_only': int(((diffs == -1)).sum())}
