"""
07_plot_and_report.py
Emplacement : PFA_Sign2Text/steps/07_plot_and_report.py

Génère TOUTES les figures et tableaux pour le rapport PFA :

  Figure 1 : Courbes loss + accuracy + BLEU-4 pour chaque experiment (déjà générées
             pendant l'entrainement — ce script les regroupe en planches).
  Figure 2 : Comparaison BLEU-4 par groupe (grouped bar chart).
  Figure 3 : Tableau de métriques complet (BLEU-1/2/3/4, ROUGE-L, CER, EM).
  Figure 4 : Courbe LR schedule (une seule fois, pour la section Implémentation).
  Figure 5 : Meilleur modèle — loss sur train / val / test.

Usage :
    cd PFA_Sign2Text
    python steps/07_plot_and_report.py

    # Si tu veux regénérer uniquement les tableaux
    python steps/07_plot_and_report.py --tables-only

Sorties (dans results/report_figures/) :
    fig1_curves_group1.png   ... fig1_curves_group4.png
    fig2_bleu_comparison.png
    fig3_metrics_table.png
    fig4_lr_schedule.png
    fig5_best_model_sets.png
    report_table.csv          <- copiable dans ton rapport
    report_table.tex          <- prêt pour LaTeX
"""

import os, sys, json, math, glob, argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

BASE        = str(Path(__file__).resolve().parent.parent)
RESULTS_DIR = os.path.join(BASE, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'report_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Ordre officiel des experiments
EXP_ORDER = [
    'g1_e1', 'g1_e2', 'g1_e3',
    'g2_e1', 'g2_e2', 'g2_e3',
    'g3_e1', 'g3_e2', 'g3_e3',
    'g4_e1', 'g4_e2', 'g4_e3',
]

GROUP_LABELS = {
    1: 'Architecture',
    2: 'Landmarks / Face',
    3: 'Tokenizer',
    4: 'Embedding',
}

GROUP_COLORS = {
    1: '#378ADD',   # bleu
    2: '#EF9F27',   # ambre
    3: '#7F77DD',   # violet
    4: '#1D9E75',   # vert
}

SHORT_NAMES = {
    'g1_e1': 'No CNN\nNo seg.',
    'g1_e2': 'No CNN\n+ seg.',
    'g1_e3': 'CNN\n+ seg.',
    'g2_e1': '100 face\npts',
    'g2_e2': '50 face\npts',
    'g2_e3': 'Hands\n+ pose',
    'g3_e1': 'Word\ntok.',
    'g3_e2': 'BPE\n2k',
    'g3_e3': 'BPE\n4k',
    'g4_e1': 'Scratch\nemb.',
    'g4_e2': 'FastText\nemb.',
    'g4_e3': 'AraBERT\nemb.',
}

FULL_NAMES = {
    'g1_e1': 'Scratch, no CNN, no segmentation',
    'g1_e2': 'Scratch, no CNN, + segmentation',
    'g1_e3': 'Scratch, CNN, + segmentation (baseline)',
    'g2_e1': '100 face points',
    'g2_e2': '50 face points (baseline)',
    'g2_e3': 'Hands + pose only',
    'g3_e1': 'Word tokenizer (baseline)',
    'g3_e2': 'BPE 2k vocab',
    'g3_e3': 'BPE 4k vocab',
    'g4_e1': 'Scratch embedding (baseline)',
    'g4_e2': 'FastText embedding',
    'g4_e3': 'AraBERT embedding',
}

GROUP_OF = {e: int(e[1]) for e in EXP_ORDER}


# ──────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────

def load_train_log(exp_id):
    """Charge le fichier JSONL de logs d'entrainement."""
    path = os.path.join(RESULTS_DIR, exp_id, 'train_log.jsonl')
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return pd.DataFrame(rows) if rows else None


def load_val_metrics(exp_id):
    path = os.path.join(RESULTS_DIR, exp_id, 'val_metrics.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_metrics(exp_id):
    path = os.path.join(RESULTS_DIR, exp_id, 'test_metrics.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all():
    """Retourne (df_val, df_test) avec une ligne par experiment."""
    val_rows, test_rows = [], []
    for exp_id in EXP_ORDER:
        vm = load_val_metrics(exp_id)
        tm = load_test_metrics(exp_id)
        g  = GROUP_OF[exp_id]
        if vm:
            vm['exp_id']    = exp_id
            vm['short_name']= SHORT_NAMES[exp_id]
            vm['full_name'] = FULL_NAMES[exp_id]
            vm['group']     = g
            val_rows.append(vm)
        if tm:
            tm['exp_id']    = exp_id
            tm['short_name']= SHORT_NAMES[exp_id]
            tm['full_name'] = FULL_NAMES[exp_id]
            tm['group']     = g
            test_rows.append(tm)
    df_val  = pd.DataFrame(val_rows)  if val_rows  else pd.DataFrame()
    df_test = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
    return df_val, df_test


# ──────────────────────────────────────────────────────────────
# FIGURE 1 : Courbes par groupe (planche 2×2 par groupe)
# ──────────────────────────────────────────────────────────────

def fig1_curves_by_group():
    """
    Pour chaque groupe, génère une figure avec 3 sous-figures (une par experiment),
    chacune montrant : loss + accuracy + BLEU-4.
    """
    for group_id in [1, 2, 3, 4]:
        exps = [e for e in EXP_ORDER if GROUP_OF[e] == group_id]
        logs = [(e, load_train_log(e)) for e in exps]
        logs = [(e, df) for e, df in logs if df is not None]

        if not logs:
            print(f'  Figure 1 groupe {group_id} : pas de données.')
            continue

        n = len(logs)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]   # shape (1, 3)

        fig.suptitle(f'Groupe {group_id} — {GROUP_LABELS[group_id]}', fontsize=13)

        for row, (exp_id, df) in enumerate(logs):
            color = GROUP_COLORS[group_id]
            name  = FULL_NAMES[exp_id]

            # Loss
            ax = axes[row, 0]
            ax.plot(df['epoch'], df['train_loss'], label='Train', color='steelblue', linewidth=1.5)
            ax.plot(df['epoch'], df['val_loss'],   label='Val',   color='tomato',    linewidth=1.5, linestyle='--')
            ax.set_title(f'{exp_id} — Loss\n{name}', fontsize=9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            # Accuracy
            ax = axes[row, 1]
            ax.plot(df['epoch'], df['train_acc'], label='Train', color='steelblue', linewidth=1.5)
            ax.plot(df['epoch'], df['val_acc'],   label='Val',   color='tomato',    linewidth=1.5, linestyle='--')
            ax.set_title(f'{exp_id} — Token Accuracy', fontsize=9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            # BLEU-4
            ax  = axes[row, 2]
            bdf = df[df['val_bleu4'].notna()]
            if len(bdf) > 0:
                ax.plot(bdf['epoch'], bdf['val_bleu4'],
                        'o-', color=color, linewidth=2, markersize=4, label='Val BLEU-4')
                best_i  = bdf['val_bleu4'].idxmax()
                best_ep = bdf.loc[best_i, 'epoch']
                best_bl = bdf.loc[best_i, 'val_bleu4']
                ax.axvline(x=best_ep, color='red', linestyle='--', alpha=0.5)
                ax.annotate(f'Best: {best_bl:.1f}\n(ep {int(best_ep)})',
                            xy=(best_ep, best_bl),
                            xytext=(best_ep + 0.5, max(0.1, best_bl - 1.5)),
                            fontsize=8, color='red',
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Pas encore calculé', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{exp_id} — BLEU-4', fontsize=9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('BLEU-4')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f'fig1_curves_group{group_id}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Figure 1 (groupe {group_id}) → {out}')


# ──────────────────────────────────────────────────────────────
# FIGURE 2 : Comparaison BLEU-4 tous groupes
# ──────────────────────────────────────────────────────────────

def fig2_bleu_comparison(df_val):
    if df_val.empty or 'bleu4' not in df_val.columns:
        print('  Figure 2 : pas de données BLEU-4.')
        return

    # Ordonner selon EXP_ORDER
    df = df_val.set_index('exp_id').reindex(EXP_ORDER).reset_index()
    df = df.dropna(subset=['bleu4'])

    if df.empty:
        print('  Figure 2 : aucune valeur BLEU-4 disponible.')
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    colors  = [GROUP_COLORS[GROUP_OF[e]] for e in df['exp_id']]
    labels  = [SHORT_NAMES.get(e, e)     for e in df['exp_id']]
    values  = df['bleu4'].tolist()
    x       = np.arange(len(df))

    bars = ax.bar(x, values, color=[c + 'cc' for c in colors],
                  edgecolor=colors, linewidth=0.8, width=0.65)

    # Ligne de base (g1_e3 = baseline complet)
    baseline_row = df[df['exp_id'] == 'g1_e3']
    if not baseline_row.empty:
        bl_val = baseline_row['bleu4'].values[0]
        ax.axhline(y=bl_val, color='gray', linestyle='--', linewidth=1,
                   label=f'Baseline (g1_e3) = {bl_val:.1f}')
        ax.legend(fontsize=9)

    # Valeurs au-dessus des barres
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('BLEU-4')
    ax.set_title('Comparaison BLEU-4 — Tous les experiments', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2 if values else 30)

    # Légende groupes
    patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Groupe {g}: {GROUP_LABELS[g]}')
               for g in [1, 2, 3, 4]]
    ax.legend(handles=patches, fontsize=8, loc='upper left', framealpha=0.7)

    # Séparateurs de groupes
    for sep in [2.5, 5.5, 8.5]:
        if sep < len(df):
            ax.axvline(x=sep, color='lightgray', linewidth=1)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig2_bleu_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure 2 → {out}')


# ──────────────────────────────────────────────────────────────
# FIGURE 3 : Tableau de métriques (image)
# ──────────────────────────────────────────────────────────────

def fig3_metrics_table(df_val, df_test):
    cols_display = ['exp_id', 'full_name', 'bleu1', 'bleu2', 'bleu3', 'bleu4',
                    'rouge_l', 'cer', 'exact_match', 'val_loss']
    cols_labels  = ['ID', 'Experiment', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4★',
                    'ROUGE-L', 'CER%', 'EM%', 'Val loss']

    if df_val.empty:
        print('  Figure 3 : pas de données.')
        return

    df_ord = df_val.set_index('exp_id').reindex(EXP_ORDER).reset_index()
    available_cols = [c for c in cols_display if c in df_ord.columns]
    df_disp = df_ord[available_cols].copy()

    # Format numérique
    for c in ['bleu1','bleu2','bleu3','bleu4','rouge_l','cer','exact_match','val_loss']:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].apply(lambda v: f'{v:.2f}' if pd.notna(v) else '—')

    fig, ax = plt.subplots(figsize=(20, 0.5 * len(df_disp) + 2))
    ax.axis('off')

    actual_labels = [cols_labels[cols_display.index(c)] for c in available_cols]
    table_data    = df_disp.values.tolist()

    tbl = ax.table(
        cellText   = table_data,
        colLabels  = actual_labels,
        cellLoc    = 'center',
        loc        = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Colorier l'en-tête
    for j in range(len(actual_labels)):
        tbl[0, j].set_facecolor('#2C2C2A')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Colorier les lignes par groupe
    group_bg = {1: '#E6F1FB', 2: '#FAEEDA', 3: '#EEEDFE', 4: '#E1F5EE'}
    for i, exp_id in enumerate(df_ord['exp_id'].tolist()):
        if pd.isna(exp_id):
            continue
        g   = GROUP_OF.get(exp_id, 1)
        clr = group_bg.get(g, '#FFFFFF')
        for j in range(len(actual_labels)):
            tbl[i + 1, j].set_facecolor(clr)

    # Mettre en gras la colonne BLEU-4
    if 'bleu4' in available_cols:
        j_bleu4 = available_cols.index('bleu4')
        for i in range(1, len(table_data) + 1):
            tbl[i, j_bleu4].set_text_props(fontweight='bold')

    ax.set_title('Tableau de métriques — Validation set', fontsize=12, pad=20)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig3_metrics_table.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure 3 → {out}')

    # Sauvegarde CSV + LaTeX
    csv_path = os.path.join(FIGURES_DIR, 'report_table.csv')
    df_disp.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'  Table CSV → {csv_path}')

    tex_path = os.path.join(FIGURES_DIR, 'report_table.tex')
    try:
        df_disp.to_latex(tex_path, index=False, caption='Résultats sur le validation set.',
                         label='tab:results', escape=True)
        print(f'  Table LaTeX → {tex_path}')
    except Exception as e:
        print(f'  LaTeX export skipped : {e}')


# ──────────────────────────────────────────────────────────────
# FIGURE 4 : LR Schedule
# ──────────────────────────────────────────────────────────────

def fig4_lr_schedule():
    base_lr     = 3e-4
    min_lr      = 1e-6
    warmup      = 100
    total_steps = 100 * 66   # 100 epochs × ~66 batches (estimation)

    steps = list(range(total_steps))
    lrs   = []
    for s in steps:
        if s <= warmup:
            lr = base_lr * (s / warmup)
        else:
            progress = min((s - warmup) / float(total_steps - warmup), 1.0)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        lrs.append(lr)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lrs, color='#EF9F27', linewidth=2)
    ax.axvline(x=warmup, color='steelblue', linestyle='--', linewidth=1,
               label=f'Fin warmup ({warmup} steps)')
    ax.set_xlabel('Steps d\'entrainement')
    ax.set_ylabel('Learning rate')
    ax.set_title('Scheduler : Warmup linéaire + Cosine Annealing', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig4_lr_schedule.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure 4 → {out}')


# ──────────────────────────────────────────────────────────────
# FIGURE 5 : Meilleur modèle — train / val / test sur même figure
# ──────────────────────────────────────────────────────────────

def fig5_best_model_sets():
    # Trouver le meilleur experiment selon BLEU-4 val
    best_exp = None
    best_bleu = 0.0
    for exp_id in EXP_ORDER:
        vm = load_val_metrics(exp_id)
        if vm and vm.get('bleu4', 0) > best_bleu:
            best_bleu = vm['bleu4']
            best_exp  = exp_id

    if best_exp is None:
        print('  Figure 5 : aucun experiment terminé.')
        return

    df = load_train_log(best_exp)
    tm = load_test_metrics(best_exp)

    if df is None:
        print(f'  Figure 5 : log manquant pour {best_exp}.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Meilleur modèle : {best_exp} — {FULL_NAMES[best_exp]}\n'
                 f'(Val BLEU-4 = {best_bleu:.2f})', fontsize=11)

    # Loss : train + val + test (point final)
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'], label='Train loss', color='steelblue', linewidth=1.5)
    ax.plot(df['epoch'], df['val_loss'],   label='Val loss',   color='tomato',    linewidth=1.5, linestyle='--')
    if tm and 'test_loss' in tm:
        last_ep = df['epoch'].max()
        ax.scatter([last_ep], [tm['test_loss']], color='seagreen', zorder=5,
                   s=80, label=f'Test loss = {tm["test_loss"]:.4f}')
    ax.set_title('Loss (train / val / test)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # BLEU-4 : val over epochs + test final
    ax  = axes[1]
    bdf = df[df['val_bleu4'].notna()]
    if len(bdf) > 0:
        ax.plot(bdf['epoch'], bdf['val_bleu4'],
                'o-', color=GROUP_COLORS[GROUP_OF[best_exp]], linewidth=2,
                markersize=4, label='Val BLEU-4')
        best_i  = bdf['val_bleu4'].idxmax()
        best_ep = bdf.loc[best_i, 'epoch']
        best_bl = bdf.loc[best_i, 'val_bleu4']
        ax.axvline(x=best_ep, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'Best val: {best_bl:.1f}', xy=(best_ep, best_bl),
                    xytext=(best_ep + 1, best_bl - 1.5), fontsize=9, color='red')

    if tm and 'bleu4' in tm:
        last_ep = df['epoch'].max()
        ax.scatter([last_ep], [tm['bleu4']], color='seagreen', zorder=5,
                   s=80, label=f'Test BLEU-4 = {tm["bleu4"]:.2f}')
    ax.set_title('BLEU-4 (val / test)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BLEU-4')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig5_best_model_sets.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure 5 → {out}  (meilleur: {best_exp})')


# ──────────────────────────────────────────────────────────────
# FIGURE 6 : Planche multi-groupes BLEU-4 (pour rapport final)
# ──────────────────────────────────────────────────────────────

def fig6_per_group_bleu(df_val):
    if df_val.empty or 'bleu4' not in df_val.columns:
        print('  Figure 6 : pas de données.')
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle('BLEU-4 par groupe d\'expériments', fontsize=13)

    for ax, group_id in zip(axes, [1, 2, 3, 4]):
        exps = [e for e in EXP_ORDER if GROUP_OF[e] == group_id]
        rows = df_val[df_val['exp_id'].isin(exps)].set_index('exp_id').reindex(exps)
        values = rows['bleu4'].tolist() if 'bleu4' in rows.columns else [0] * len(exps)
        labels = [SHORT_NAMES[e] for e in exps]
        color  = GROUP_COLORS[group_id]

        bars = ax.bar(range(len(exps)),
                      [v if not (isinstance(v, float) and math.isnan(v)) else 0 for v in values],
                      color=color + 'bb', edgecolor=color, linewidth=0.8, width=0.5)

        for bar, val in zip(bars, values):
            if isinstance(val, float) and not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f'G{group_id}: {GROUP_LABELS[group_id]}', fontsize=10)
        ax.set_ylabel('BLEU-4')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max([v for v in values if isinstance(v, float) and not math.isnan(v)] or [1]) * 1.25)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig6_per_group_bleu.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure 6 → {out}')


# ──────────────────────────────────────────────────────────────
# POINT D'ENTREE
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Génération figures et tableaux rapport PFA')
    parser.add_argument('--tables-only', action='store_true',
                        help='Générer uniquement le tableau de métriques')
    args = parser.parse_args()

    print('=' * 60)
    print('07_plot_and_report.py — Génération des figures')
    print('=' * 60)
    print(f'  Dossier source : {RESULTS_DIR}')
    print(f'  Dossier sortie : {FIGURES_DIR}')

    df_val, df_test = load_all()

    if df_val.empty:
        print('\n  Aucune donnée trouvée.')
        print('  Lance d\'abord : python steps/06_train_experiments.py --exp g1_e1')
        sys.exit(0)

    print(f'\n  {len(df_val)} experiments avec résultats val trouvés.')
    print(f'  {len(df_test)} experiments avec résultats test trouvés.')

    if args.tables_only:
        print('\nMode tables uniquement.')
        fig3_metrics_table(df_val, df_test)
    else:
        print('\nGénération de toutes les figures...\n')
        fig1_curves_by_group()
        fig2_bleu_comparison(df_val)
        fig3_metrics_table(df_val, df_test)
        fig4_lr_schedule()
        fig5_best_model_sets()
        fig6_per_group_bleu(df_val)

    print('\n' + '=' * 60)
    print('Résumé des résultats disponibles')
    print('=' * 60)
    if not df_val.empty and 'bleu4' in df_val.columns:
        cols = ['exp_id', 'bleu4', 'rouge_l', 'cer', 'exact_match', 'val_loss']
        cols = [c for c in cols if c in df_val.columns]
        df_sorted = df_val.set_index('exp_id').reindex(EXP_ORDER).reset_index()
        df_sorted = df_sorted[cols].dropna(subset=['bleu4'])
        print(df_sorted.to_string(index=False))

    print(f'\nFigures dans : {FIGURES_DIR}')
    print('Fichiers générés :')
    for f in sorted(os.listdir(FIGURES_DIR)):
        print(f'  {f}')
