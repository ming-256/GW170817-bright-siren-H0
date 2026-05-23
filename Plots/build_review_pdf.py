"""Bundle all REVIEW.md figures into one PDF, grouped into buckets.

Output: Results/gwtc1_phasemarg/plots/REVIEW_bundle.pdf
"""
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLOTS = os.path.join(REPO, 'Results', 'gwtc1_phasemarg', 'plots')
OUT = os.path.join(PLOTS, 'REVIEW_bundle.pdf')

BUCKETS = [
    ('1. Scaling study (speedup)', [
        'scaling_study_full.png',
    ]),
    ('2. GW150914 — XPHM validation', [
        'corner_GW150914_waveform_comparison.png',
        'corner_GW150914.png',
    ]),
    ('3. GW170817 — three-waveform set (KDE)', [
        'H0_waveform_comparison.png',
        'corner_GW170817_waveform_comparison.png',
        'corner_combined_waveforms.png',
    ]),
    ('4. GW170817 — XAS vs LVK full corner', [
        'corner_GW170817_XAS_vs_LVK.png',
    ]),
    ('5. Prior sensitivity — XAS primary (KDE)', [
        'H0_prior_sensitivity.png',
        'corner_reweighted_vs_sampled_flatZ_XAS.png',
        'H0_reweighted_vs_sampled_flatZ.png',
    ]),
    ('6. d_L–iota bimodality', [
        'bimodality.png',
        'dL_posterior.png',
    ]),
    ('7. Sky-prior / full-sky vs narrow', [
        'corner_full_sky_vs_narrow.png',
        'H0_full_sky_vs_narrow.png',
        'corner_sky_localization.png',
    ]),
    ('8. Heterodyned vs unheterodyned', [
        'corner_IMRPhenomD_hetero_vs_unhetero.png',
        'corner_TaylorF2_hetero_vs_unhetero.png',
        'corner_speedup_hetero_vs_unhetero.png',
        'ess_comparison_hetero_vs_unhetero.png',
        'corner_unheterodyned_vs_gwtc.png',
        'H0_unheterodyned_vs_gwtc.png',
        'gpu_vs_cpu_projection.png',
    ]),
    ('9. Phase marginalization schematic', [
        'phase_marginalization_schematic.png',
    ]),
]


def cover_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.78, 'GW170817 / GW150914 — figure review bundle',
             ha='center', va='center', fontsize=20, weight='bold')
    fig.text(0.5, 0.72, '2026-04-28 final-stretch state', ha='center', fontsize=12)
    fig.text(0.1, 0.6, 'Buckets:', fontsize=14, weight='bold')
    y = 0.56
    for title, files in BUCKETS:
        present = sum(os.path.exists(os.path.join(PLOTS, f)) for f in files)
        fig.text(0.12, y, f'{title}  ({present}/{len(files)} figures)', fontsize=11)
        y -= 0.035
    fig.text(0.1, 0.06,
             'See REVIEW.md for numbers, decisions, and provenance.\n'
             'Source PNGs in Results/gwtc1_phasemarg/plots/.',
             fontsize=9, color='0.4')
    pdf.savefig(fig); plt.close(fig)


def section_page(pdf, title):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.5, title, ha='center', va='center',
             fontsize=22, weight='bold')
    pdf.savefig(fig); plt.close(fig)


def figure_page(pdf, png_path, caption):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.85])
    img = mpimg.imread(png_path)
    ax.imshow(img); ax.axis('off')
    fig.text(0.5, 0.04, caption, ha='center', fontsize=9, color='0.3')
    pdf.savefig(fig); plt.close(fig)


with PdfPages(OUT) as pdf:
    cover_page(pdf)
    for title, files in BUCKETS:
        section_page(pdf, title)
        for f in files:
            p = os.path.join(PLOTS, f)
            if not os.path.exists(p):
                print(f'  skip missing: {f}')
                continue
            figure_page(pdf, p, f)
            print(f'  + {f}')

print(f'\nWrote {OUT}')
