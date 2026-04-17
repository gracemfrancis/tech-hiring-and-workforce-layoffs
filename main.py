"""
=============================================================
  TECH HIRING & LAYOFFS WORKFORCE ANALYZER (2000–2025)
  Dynamic ML-powered database for job market insights
=============================================================
  Dataset: https://www.kaggle.com/datasets/aryanmdev/tech-hiring-and-layoffs-workforce-data-20002025
  Requirements: pandas, numpy, matplotlib, scikit-learn, seaborn
=============================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import Normalize
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION & STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'text.color':       '#e6edf3',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
})

COLORS = {
    'hire':    '#3fb950',   # green
    'layoff':  '#f85149',   # red
    'neutral': '#58a6ff',   # blue
    'accent':  '#d29922',   # yellow/gold
    'purple':  '#bc8cff',
    'teal':    '#39d0d8',
}

DATASET_COLUMNS_EXPECTED = [
    'Year', 'Company', 'Industry', 'Country',
    'Layoffs', 'Hiring', 'Total_Employees',
    'Layoff_Percentage', 'Hire_Percentage'
]


# ─────────────────────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean the dataset from CSV."""
    print(f"\n📂  Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"\n❌  File not found: {filepath}")
        print("    Please download the dataset from Kaggle and place it in the same folder as this script.")
        print("    Expected filename: tech_hiring_layoffs.csv  (rename if needed)\n")
        sys.exit(1)

    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

    # ── normalise common column name variants ──────────────────
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    candidates = {
        'year':              ['year', 'date', 'fiscal_year', 'reported_year'],
        'company':           ['company', 'company_name', 'firm', 'organization'],
        'industry':          ['industry', 'sector', 'field'],
        'country':           ['country', 'location', 'region', 'country_region'],
        'layoffs':           ['layoffs', 'laid_off', 'layoff_count', 'employees_laid_off', 'number_laid_off'],
        'hiring':            ['hiring', 'hired', 'new_hires', 'hires', 'employees_hired', 'number_hired'],
        'total_employees':   ['total_employees', 'workforce', 'headcount', 'employees', 'total_workforce'],
        'layoff_percentage': ['layoff_percentage', 'layoff_pct', 'percentage_laid_off', 'layoff_%'],
        'hire_percentage':   ['hire_percentage', 'hire_pct', 'percentage_hired', 'hire_%'],
    }
    for standard, variants in candidates.items():
        for v in variants:
            if v in col_lower and standard.upper() not in [c.upper() for c in rename_map.values()]:
                rename_map[col_lower[v]] = standard.title().replace('_', '_')
                break

    # Build clean rename
    final_rename = {}
    for orig, std in rename_map.items():
        std_key = std.replace(' ', '_')
        final_rename[orig] = std_key.upper() if std_key.upper() in ['YEAR'] else std_key.capitalize()

    # Simple approach: just rename by position for known datasets
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'year' in cl or 'date' in cl:              col_map[c] = 'Year'
        elif 'company' in cl or 'firm' in cl:         col_map[c] = 'Company'
        elif 'industry' in cl or 'sector' in cl:      col_map[c] = 'Industry'
        elif 'country' in cl or 'location' in cl:     col_map[c] = 'Country'
        elif 'laid_off' in cl or 'layoff' in cl and 'pct' not in cl and '%' not in cl:
            col_map[c] = 'Layoffs'
        elif ('hired' in cl or 'hiring' in cl) and 'pct' not in cl and '%' not in cl:
            col_map[c] = 'Hiring'
        elif 'total' in cl and 'employ' in cl:        col_map[c] = 'Total_Employees'
        elif ('layoff' in cl) and ('pct' in cl or '%' in cl or 'percent' in cl):
            col_map[c] = 'Layoff_Pct'
        elif ('hire' in cl or 'hiring' in cl) and ('pct' in cl or '%' in cl or 'percent' in cl):
            col_map[c] = 'Hire_Pct'
    df.rename(columns=col_map, inplace=True)

    # ── coerce types ───────────────────────────────────────────
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        # Handle date strings like "2021-01-01"
        if df['Year'].isna().sum() > len(df) * 0.3:
            raw = pd.read_csv(filepath)
            raw.columns = [c.strip() for c in raw.columns]
            for c in raw.columns:
                if 'year' in c.lower() or 'date' in c.lower():
                    df['Year'] = pd.Series(pd.to_datetime(raw[c], errors='coerce')).dt.year
                    break
        df['Year'] = df['Year'].astype('Int64')

    for num_col in ['Layoffs', 'Hiring', 'Total_Employees', 'Layoff_Pct', 'Hire_Pct']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    # ── derived columns ─────────────────────────────────────────
    if 'Layoffs' in df.columns and 'Hiring' in df.columns:
        df['Net_Change'] = df['Hiring'].fillna(0) - df['Layoffs'].fillna(0)

    if 'Layoffs' in df.columns and 'Total_Employees' in df.columns:
        mask = df['Total_Employees'] > 0
        if 'Layoff_Pct' not in df.columns:
            df['Layoff_Pct'] = np.where(mask, df['Layoffs'] / df['Total_Employees'] * 100, np.nan)

    df.dropna(subset=['Year'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅  Loaded {len(df):,} records  |  Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────
def _save_show(fig, title_slug: str, output_dir: str):
    path = os.path.join(output_dir, f"{title_slug}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"   💾  Saved → {path}")


def _styled_title(ax, text, fontsize=14):
    ax.set_title(text, fontsize=fontsize, color='#e6edf3', pad=10,
                 fontweight='bold', fontstyle='italic')


def _bar_gradient(ax, x, y, cmap_name='RdYlGn', label='', orient='v'):
    norm = Normalize(float(min(y)), float(max(y)))
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = [cmap(norm(v)) for v in y]
    if orient == 'v':
        bars = ax.bar(x, y, color=colors, label=label, edgecolor='#0d1117', linewidth=0.5)
    else:
        bars = ax.barh(x, y, color=colors, label=label, edgecolor='#0d1117', linewidth=0.5)
    return bars


# ─────────────────────────────────────────────────────────────
#  OVERVIEW ANALYSIS  (no year filter)
# ─────────────────────────────────────────────────────────────
def global_overview(df: pd.DataFrame, output_dir: str):
    print("\n" + "═"*60)
    print("  📊  GLOBAL OVERVIEW  (All Years)")
    print("═"*60)

    fig = plt.figure(figsize=(20, 14), facecolor='#0d1117')
    fig.suptitle('TECH WORKFORCE GLOBAL OVERVIEW  2000 – 2025',
                 fontsize=18, color='#58a6ff', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Layoffs vs Hiring per Year (line) ──────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    yr = df.groupby('Year')[['Layoffs', 'Hiring']].sum().dropna(how='all')
    if not yr.empty:
        if 'Layoffs' in yr.columns:
            ax1.fill_between(yr.index, yr['Layoffs'], alpha=0.3, color=COLORS['layoff'])
            ax1.plot(yr.index, yr['Layoffs'], color=COLORS['layoff'], linewidth=2.5, label='Total Layoffs', marker='o', markersize=4)
        if 'Hiring' in yr.columns:
            ax1.fill_between(yr.index, yr['Hiring'], alpha=0.2, color=COLORS['hire'])
            ax1.plot(yr.index, yr['Hiring'], color=COLORS['hire'], linewidth=2.5, label='Total Hiring', marker='s', markersize=4)
        ax1.legend(framealpha=0.2, facecolor='#161b22', edgecolor='#30363d')
        ax1.grid(True, alpha=0.3)
    _styled_title(ax1, 'Annual Layoffs vs Hiring Trend')
    ax1.set_xlabel('Year'); ax1.set_ylabel('Headcount')

    # ── 2. Net Change donut ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if 'Net_Change' in df.columns:
        pos = (df['Net_Change'] > 0).sum()
        neg = (df['Net_Change'] <= 0).sum()
        pie_result = ax2.pie(
            [pos, neg], labels=['Net Positive', 'Net Negative'],
            autopct='%1.1f%%', colors=[COLORS['hire'], COLORS['layoff']],
            wedgeprops={'linewidth': 2, 'edgecolor': '#0d1117'},
            startangle=140, pctdistance=0.8)
        if len(pie_result) == 3:
            for at in pie_result[2]:
                at.set_color('#e6edf3')
                at.set_fontsize(10)
        ax2.add_patch(Circle((0, 0), 0.55, color='#161b22'))
    _styled_title(ax2, 'Net Change Distribution')

    # ── 3. Top 10 Industries by Layoffs ──────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    if 'Industry' in df.columns and 'Layoffs' in df.columns:
        top_ind = df.groupby('Industry')['Layoffs'].sum().nlargest(10).sort_values()
        _bar_gradient(ax3, top_ind.index, np.array(top_ind.values, dtype=float), 'Reds', orient='h')
        ax3.set_xlabel('Total Layoffs')
        ax3.tick_params(axis='y', labelsize=8)
    _styled_title(ax3, 'Top 10 Industries by Total Layoffs')

    # ── 4. Country heatmap (top 8) ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if 'Country' in df.columns and 'Layoffs' in df.columns:
        ctry = df.groupby('Country')['Layoffs'].sum().nlargest(8)
        colors_c = plt.colormaps['Oranges'](np.linspace(0.4, 1.0, len(ctry)))
        ax4.barh(ctry.index, np.array(ctry.values, dtype=float), color=colors_c)
        ax4.set_xlabel('Total Layoffs', fontsize=8)
        ax4.tick_params(labelsize=8)
    _styled_title(ax4, 'Top Countries\nby Layoffs')

    # ── 5. Year-over-Year % Change ────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    if 'Layoffs' in yr.columns:
        yoy = yr['Layoffs'].pct_change() * 100
        bar_colors = [COLORS['layoff'] if v > 0 else COLORS['hire'] for v in yoy]
        ax5.bar(yoy.index, yoy.values, color=bar_colors, edgecolor='#0d1117', linewidth=0.4)
        ax5.axhline(0, color='#8b949e', linewidth=1, linestyle='--')
        ax5.set_xlabel('Year'); ax5.set_ylabel('YoY Change (%)')
    _styled_title(ax5, 'Year-over-Year % Change in Layoffs')

    # ── 6. Summary Stats ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    stats_text = []
    if 'Layoffs' in df.columns:
        stats_text.append(f"Total Layoffs:   {df['Layoffs'].sum():>12,.0f}")
    if 'Hiring' in df.columns:
        stats_text.append(f"Total Hiring:    {df['Hiring'].sum():>12,.0f}")
    if 'Company' in df.columns:
        stats_text.append(f"Companies:       {df['Company'].nunique():>12,}")
    if 'Industry' in df.columns:
        stats_text.append(f"Industries:      {df['Industry'].nunique():>12,}")
    stats_text.append(f"Year Range:      {df['Year'].min()} – {df['Year'].max()}")
    stats_text.append(f"Total Records:   {len(df):>12,}")

    box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02",
                          facecolor='#1c2128', edgecolor='#58a6ff', linewidth=1.5,
                          transform=ax6.transAxes)
    ax6.add_patch(box)
    ax6.text(0.5, 0.95, '📈 DATASET STATS', transform=ax6.transAxes,
             ha='center', va='top', fontsize=11, color='#58a6ff', fontweight='bold')
    for i, line in enumerate(stats_text):
        ax6.text(0.12, 0.82 - i * 0.13, line, transform=ax6.transAxes,
                 fontsize=9, color='#e6edf3', fontfamily='monospace')

    _save_show(fig, 'global_overview', output_dir)


# ─────────────────────────────────────────────────────────────
#  SINGLE YEAR DEEP DIVE
# ─────────────────────────────────────────────────────────────
def year_deep_dive(df: pd.DataFrame, year: int, output_dir: str):
    dfy = df[df['Year'] == year].copy()
    if dfy.empty:
        print(f"\n⚠️  No data found for year {year}.")
        available = sorted(df['Year'].dropna().unique())
        print(f"   Available years: {list(available)}")
        return

    print(f"\n{'═'*60}")
    print(f"  🔍  DEEP DIVE: {year}  ({len(dfy):,} records)")
    print(f"{'═'*60}")

    # Console summary
    if 'Layoffs' in dfy.columns:
        print(f"  Total Layoffs   : {dfy['Layoffs'].sum():,.0f}")
    if 'Hiring' in dfy.columns:
        print(f"  Total Hiring    : {dfy['Hiring'].sum():,.0f}")
    if 'Net_Change' in dfy.columns:
        nc = dfy['Net_Change'].sum()
        symbol = '📈' if nc >= 0 else '📉'
        print(f"  Net Change      : {symbol} {nc:,.0f}")
    if 'Company' in dfy.columns:
        print(f"  Companies       : {dfy['Company'].nunique()}")
    if 'Industry' in dfy.columns:
        print(f"  Industries      : {dfy['Industry'].nunique()}")
    if 'Country' in dfy.columns:
        print(f"  Countries       : {dfy['Country'].nunique()}")

    fig = plt.figure(figsize=(22, 16), facecolor='#0d1117')
    fig.suptitle(f'TECH WORKFORCE ANALYSIS  ◆  {year}',
                 fontsize=20, color=COLORS['neutral'], fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    # ── 1. Layoffs vs Hiring bar (top industries) ─────────────
    ax1 = fig.add_subplot(gs[0, :2])
    if 'Industry' in dfy.columns:
        ind_grp = dfy.groupby('Industry')[['Layoffs', 'Hiring']].sum().fillna(0).nlargest(8, 'Layoffs')
        x = np.arange(len(ind_grp))
        w = 0.38
        ax1.bar(x - w/2, ind_grp['Layoffs'], w, color=COLORS['layoff'], label='Layoffs', alpha=0.9)
        if 'Hiring' in ind_grp.columns:
            ax1.bar(x + w/2, ind_grp['Hiring'], w, color=COLORS['hire'], label='Hiring', alpha=0.9)
        ax1.set_xticks(x); ax1.set_xticklabels(ind_grp.index, rotation=30, ha='right', fontsize=8)
        ax1.legend(framealpha=0.2); ax1.grid(True, alpha=0.3)
    _styled_title(ax1, f'Layoffs vs Hiring by Industry — {year}')

    # ── 2. Top 10 Companies by Layoffs ───────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    if 'Company' in dfy.columns and 'Layoffs' in dfy.columns:
        top_co = dfy.groupby('Company')['Layoffs'].sum().nlargest(10).sort_values()
        colors_c = plt.colormaps['Reds'](np.linspace(0.4, 1.0, len(top_co)))
        ax2.barh(top_co.index, np.array(top_co.values, dtype=float), color=colors_c)
        ax2.set_xlabel('Layoffs')
        ax2.tick_params(axis='y', labelsize=8)
    _styled_title(ax2, f'Top 10 Companies by Layoffs — {year}')

    # ── 3. Layoff % distribution ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    if 'Layoff_Pct' in dfy.columns:
        data_pct = dfy['Layoff_Pct'].dropna()
        ax3.hist(data_pct, bins=30, color=COLORS['layoff'], alpha=0.8, edgecolor='#0d1117')
        ax3.axvline(data_pct.mean(), color=COLORS['accent'], linewidth=2, linestyle='--',
                    label=f'Mean: {data_pct.mean():.1f}%')
        ax3.axvline(data_pct.median(), color=COLORS['teal'], linewidth=2, linestyle=':',
                    label=f'Median: {data_pct.median():.1f}%')
        ax3.legend(framealpha=0.2)
    elif 'Layoffs' in dfy.columns:
        ax3.hist(dfy['Layoffs'].dropna(), bins=30, color=COLORS['layoff'], alpha=0.8)
    ax3.set_xlabel('Value'); ax3.set_ylabel('Count')
    _styled_title(ax3, f'Layoff Distribution — {year}')

    # ── 4. Net Change pie ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if 'Net_Change' in dfy.columns:
        pos = (dfy['Net_Change'] > 0).sum()
        neg = (dfy['Net_Change'] < 0).sum()
        zer = (dfy['Net_Change'] == 0).sum()
        vals = [v for v in [pos, neg, zer] if v > 0]
        labs = [l for l, v in zip(['Growing', 'Shrinking', 'Stable'], [pos, neg, zer]) if v > 0]
        clrs = [COLORS['hire'], COLORS['layoff'], COLORS['neutral']][:len(vals)]
        pie_result4 = ax4.pie(vals, labels=labs, autopct='%1.1f%%',
                                        colors=clrs, startangle=90,
                                        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5},
                                        pctdistance=0.82)
        if len(pie_result4) == 3:
            for at in pie_result4[2]:
                at.set_color('#e6edf3')
                at.set_fontsize(9)
        ax4.add_patch(Circle((0, 0), 0.55, color='#161b22'))
    _styled_title(ax4, 'Company\nHealth Split')

    # ── 5. Country breakdown ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 3])
    if 'Country' in dfy.columns and 'Layoffs' in dfy.columns:
        ctry = dfy.groupby('Country')['Layoffs'].sum().nlargest(7).sort_values()
        clrs_c = plt.colormaps['OrRd'](np.linspace(0.4, 1.0, len(ctry)))
        ax5.barh(ctry.index, np.array(ctry.values, dtype=float), color=clrs_c)
        ax5.tick_params(axis='y', labelsize=8)
    _styled_title(ax5, 'Top Countries\nby Layoffs')

    # ── 6. Industry share of layoffs (horizontal 100% bar) ────
    ax6 = fig.add_subplot(gs[2, :2])
    if 'Industry' in dfy.columns and 'Layoffs' in dfy.columns:
        ind_share = dfy.groupby('Industry')['Layoffs'].sum().nlargest(10)
        total = ind_share.sum()
        pcts = ind_share / total * 100
        palette = plt.colormaps['tab10'](np.linspace(0, 1, len(pcts)))
        left = 0
        for (ind, val), col in zip(pcts.items(), palette):
            ax6.barh(['Share'], val, left=left, color=col, label=ind, edgecolor='#0d1117')
            if val > 4:
                ax6.text(left + val/2, 0, f'{val:.1f}%', ha='center', va='center',
                         fontsize=8, color='#0d1117', fontweight='bold')
            left += val
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5,
                   fontsize=8, framealpha=0.1, facecolor='#161b22')
        ax6.set_xlim(0, 100); ax6.set_xlabel('% Share of Layoffs')
    _styled_title(ax6, f'Industry Share of Total Layoffs — {year}')

    # ── 7. Top Hiring Companies ───────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2:])
    if 'Company' in dfy.columns and 'Hiring' in dfy.columns:
        top_hire = dfy.groupby('Company')['Hiring'].sum().nlargest(10).sort_values()
        colors_h = plt.colormaps['Greens'](np.linspace(0.4, 1.0, len(top_hire)))
        ax7.barh(top_hire.index, np.array(top_hire.values, dtype=float), color=colors_h)
        ax7.set_xlabel('New Hires')
        ax7.tick_params(axis='y', labelsize=8)
    _styled_title(ax7, f'Top 10 Hiring Companies — {year}')

    _save_show(fig, f'deep_dive_{year}', output_dir)


# ─────────────────────────────────────────────────────────────
#  YEAR RANGE COMPARISON
# ─────────────────────────────────────────────────────────────
def year_range_analysis(df: pd.DataFrame, start: int, end: int, output_dir: str):
    dfr = df[(df['Year'] >= start) & (df['Year'] <= end)].copy()
    if dfr.empty:
        print(f"\n⚠️  No data for range {start}–{end}"); return

    print(f"\n{'═'*60}")
    print(f"  📅  RANGE ANALYSIS: {start} – {end}  ({len(dfr):,} records)")
    print(f"{'═'*60}")

    fig = plt.figure(figsize=(22, 14), facecolor='#0d1117')
    fig.suptitle(f'WORKFORCE TREND ANALYSIS  ◆  {start} – {end}',
                 fontsize=18, color=COLORS['purple'], fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    yr = dfr.groupby('Year')[['Layoffs', 'Hiring']].sum()

    # ── 1. Trend lines ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    if 'Layoffs' in yr.columns:
        ax1.plot(yr.index, yr['Layoffs'], color=COLORS['layoff'], lw=2.5, marker='o', ms=5, label='Layoffs')
        # Regression line
        X = yr.index.values.reshape(-1, 1)
        y_l = yr['Layoffs'].values
        if len(X) > 2:
            lr = LinearRegression().fit(X, y_l)
            ax1.plot(yr.index, lr.predict(X), color=COLORS['layoff'], lw=1.5, linestyle='--', alpha=0.6, label='Layoff Trend')
    if 'Hiring' in yr.columns:
        ax1.plot(yr.index, yr['Hiring'], color=COLORS['hire'], lw=2.5, marker='s', ms=5, label='Hiring')
        X = yr.index.values.reshape(-1, 1)
        y_h = yr['Hiring'].values
        if len(X) > 2:
            lr2 = LinearRegression().fit(X, y_h)
            ax1.plot(yr.index, lr2.predict(X), color=COLORS['hire'], lw=1.5, linestyle='--', alpha=0.6, label='Hiring Trend')
    ax1.legend(framealpha=0.2); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Year'); ax1.set_ylabel('Headcount')
    _styled_title(ax1, f'Layoffs & Hiring Trends with Regression  [{start}–{end}]')

    # ── 2. Stacked area ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if 'Layoffs' in yr.columns and 'Hiring' in yr.columns:
        ax2.stackplot(yr.index, yr['Layoffs'], yr['Hiring'],
                      labels=['Layoffs', 'Hiring'],
                      colors=[COLORS['layoff'], COLORS['hire']], alpha=0.7)
        ax2.legend(loc='upper left', framealpha=0.2)
    _styled_title(ax2, 'Stacked Volume')

    # ── 3. Industry evolution heatmap ─────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    if 'Industry' in dfr.columns and 'Layoffs' in dfr.columns:
        pivot = dfr.pivot_table(values='Layoffs', index='Industry', columns='Year', aggfunc='sum').fillna(0)
        top_inds = pivot.sum(axis=1).nlargest(8).index
        pivot = pivot.loc[top_inds]
        sns.heatmap(pivot, ax=ax3, cmap='Reds', linewidths=0.3, linecolor='#0d1117',
                    cbar_kws={'shrink': 0.7}, annot=len(pivot.columns) <= 10,
                    fmt='.0f', annot_kws={'size': 7})
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.tick_params(axis='y', labelsize=8)
    _styled_title(ax3, 'Industry Layoffs Heatmap by Year')

    # ── 4. Net Change over years ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if 'Net_Change' in dfr.columns:
        nc = dfr.groupby('Year')['Net_Change'].sum()
        bar_c = [COLORS['hire'] if v >= 0 else COLORS['layoff'] for v in nc.values]
        ax4.bar(nc.index, np.array(nc.values, dtype=float), color=bar_c, edgecolor='#0d1117')
        ax4.axhline(0, color='#8b949e', lw=1.2, ls='--')
        ax4.set_xlabel('Year')
    _styled_title(ax4, 'Net Workforce\nChange by Year')

    _save_show(fig, f'range_analysis_{start}_{end}', output_dir)


# ─────────────────────────────────────────────────────────────
#  ML MODELS
# ─────────────────────────────────────────────────────────────
def run_ml_models(df: pd.DataFrame, output_dir: str):
    print(f"\n{'═'*60}")
    print("  🤖  ML MODELS & PREDICTIONS")
    print(f"{'═'*60}")

    # ── Feature Engineering ───────────────────────────────────
    df_ml = df.copy()
    le_industry = LabelEncoder()
    le_country  = LabelEncoder()

    if 'Industry' in df_ml.columns:
        df_ml['Industry_Enc'] = le_industry.fit_transform(df_ml['Industry'].fillna('Unknown'))
    if 'Country' in df_ml.columns:
        df_ml['Country_Enc'] = le_country.fit_transform(df_ml['Country'].fillna('Unknown'))

    feature_cols = ['Year']
    if 'Industry_Enc' in df_ml.columns: feature_cols.append('Industry_Enc')
    if 'Country_Enc'  in df_ml.columns: feature_cols.append('Country_Enc')
    if 'Total_Employees' in df_ml.columns: feature_cols.append('Total_Employees')
    if 'Hiring' in df_ml.columns: feature_cols.append('Hiring')

    fig = plt.figure(figsize=(22, 16), facecolor='#0d1117')
    fig.suptitle('ML MODEL INSIGHTS  ◆  PREDICTIONS & CLUSTERING',
                 fontsize=18, color=COLORS['teal'], fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── A. Layoff Prediction (Random Forest) ─────────────────
    ax_rf = fig.add_subplot(gs[0, :2])
    if 'Layoffs' in df_ml.columns:
        ml_data = df_ml[feature_cols + ['Layoffs']].dropna()
        if len(ml_data) > 50:
            X = ml_data[feature_cols].values
            y = ml_data['Layoffs'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_s, y_train)
            y_pred = rf.predict(X_test_s)

            mae = mean_absolute_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            print(f"\n  🌲 Random Forest — Layoff Prediction")
            print(f"     MAE : {mae:,.0f}  |  R² : {r2:.3f}")

            ax_rf.scatter(y_test, y_pred, alpha=0.5, s=20,
                          c=[COLORS['layoff'] if e > 0 else COLORS['hire']
                             for e in (y_pred - y_test)], edgecolors='none')
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax_rf.plot(lims, lims, color=COLORS['accent'], lw=2, ls='--', label='Perfect Fit')
            ax_rf.set_xlabel('Actual Layoffs'); ax_rf.set_ylabel('Predicted Layoffs')
            ax_rf.text(0.05, 0.92, f'R² = {r2:.3f}  |  MAE = {mae:,.0f}',
                       transform=ax_rf.transAxes, color=COLORS['teal'], fontsize=9)
            ax_rf.legend(framealpha=0.2)

            # Feature importance subplot
            ax_fi = fig.add_subplot(gs[0, 2])
            importances = rf.feature_importances_
            feat_names  = feature_cols
            sorted_idx  = np.argsort(importances)
            ax_fi.barh([feat_names[i] for i in sorted_idx],
                       importances[sorted_idx],
                       color=COLORS['teal'])
            _styled_title(ax_fi, 'Feature Importance\n(Random Forest)')
    _styled_title(ax_rf, 'Random Forest: Actual vs Predicted Layoffs')

    # ── B. KMeans Clustering ──────────────────────────────────
    ax_km = fig.add_subplot(gs[1, :2])
    cluster_feat = [c for c in ['Layoffs', 'Hiring', 'Net_Change'] if c in df_ml.columns]
    if len(cluster_feat) >= 2:
        km_data = df_ml[cluster_feat].dropna()
        if len(km_data) > 20:
            scaler_km = StandardScaler()
            km_scaled = scaler_km.fit_transform(km_data)
            inertias = []
            k_range = range(2, min(9, len(km_data)//10 + 2))
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(km_scaled)
                inertias.append(km.inertia_)
            ax_km.plot(list(k_range), inertias, color=COLORS['purple'], marker='D', ms=6, lw=2)
            ax_km.set_xlabel('Number of Clusters (k)'); ax_km.set_ylabel('Inertia')
            ax_km.grid(True, alpha=0.3)

            # Best k scatter
            best_k = 4
            km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km_final.fit_predict(km_scaled)
            df_ml['Cluster'] = np.nan
            df_ml.loc[km_data.index, 'Cluster'] = labels

            ax_scatter = fig.add_subplot(gs[1, 2])
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(km_scaled)
            scatter_colors = plt.colormaps['Set2'](np.linspace(0, 1, best_k))
            for c_id in range(best_k):
                mask = labels == c_id
                ax_scatter.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                                   s=15, alpha=0.6, color=scatter_colors[c_id],
                                   label=f'Cluster {c_id}')
            ax_scatter.legend(framealpha=0.2, fontsize=8)
            _styled_title(ax_scatter, 'KMeans Clusters\n(PCA 2D View)')
            print(f"\n  🔵 KMeans Clustering: {best_k} clusters identified")
            for c_id in range(best_k):
                c_data = km_data[labels == c_id]
                print(f"     Cluster {c_id}: {len(c_data):,} companies | "
                      f"Avg Layoffs: {c_data['Layoffs'].mean():,.0f}" if 'Layoffs' in c_data else "")
    _styled_title(ax_km, 'Elbow Curve — Optimal K Selection')

    # ── C. Gradient Boosting Layoff Forecast ──────────────────
    ax_gb = fig.add_subplot(gs[2, :2])
    yr_agg = df.groupby('Year').agg(
        Total_Layoffs=('Layoffs', 'sum'),
        Total_Hiring=('Hiring',  'sum') if 'Hiring' in df.columns else ('Layoffs', 'count')
    ).dropna(how='all').reset_index()

    if len(yr_agg) > 10:
        X_yr = yr_agg[['Year']].values
        y_yr = yr_agg['Total_Layoffs'].values
        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        gb.fit(X_yr, y_yr)
        future_years = np.arange(yr_agg['Year'].min(), yr_agg['Year'].max() + 4).reshape(-1, 1)
        preds = gb.predict(future_years)

        ax_gb.plot(yr_agg['Year'], y_yr, color=COLORS['layoff'], lw=2.5, marker='o', ms=4, label='Historical')
        future_mask = future_years.flatten() > yr_agg['Year'].max()
        ax_gb.plot(future_years[~future_mask], preds[~future_mask],
                   color=COLORS['accent'], lw=2, ls='--', label='GB Fit')
        ax_gb.plot(future_years[future_mask],  preds[future_mask],
                   color=COLORS['teal'], lw=2.5, ls='--', marker='^', ms=6, label='Forecast')
        ax_gb.axvline(yr_agg['Year'].max(), color='#8b949e', lw=1.2, ls=':',
                      label=f"Last data: {yr_agg['Year'].max()}")
        ax_gb.legend(framealpha=0.2); ax_gb.grid(True, alpha=0.3)
        ax_gb.set_xlabel('Year'); ax_gb.set_ylabel('Total Layoffs')

        print(f"\n  📈 Gradient Boosting Forecast:")
        for yr_f, pr_f in zip(future_years[future_mask].flatten(), preds[future_mask]):
            print(f"     {yr_f}: ~{pr_f:,.0f} projected layoffs")
    _styled_title(ax_gb, 'Gradient Boosting: Layoff Forecast (next 3 years)')

    # ── D. Model comparison ───────────────────────────────────
    ax_mc = fig.add_subplot(gs[2, 2])
    if 'Layoffs' in df_ml.columns:
        ml_data = df_ml[feature_cols + ['Layoffs']].dropna()
        if len(ml_data) > 50:
            X = ml_data[feature_cols].values
            y = ml_data['Layoffs'].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
            models = {
                'Linear\nRegression': LinearRegression(),
                'Random\nForest':     RandomForestRegressor(n_estimators=50, random_state=42),
                'Gradient\nBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            }
            r2s = []
            for name, model in models.items():
                model.fit(X_tr_s, y_tr)
                r2s.append(max(0, r2_score(y_te, model.predict(X_te_s))))
            bar_colors = [COLORS['neutral'], COLORS['hire'], COLORS['accent']]
            ax_mc.bar(list(models.keys()), r2s, color=bar_colors, edgecolor='#0d1117')
            ax_mc.set_ylim(0, 1); ax_mc.set_ylabel('R² Score')
            for i, v in enumerate(r2s):
                ax_mc.text(i, v + 0.02, f'{v:.3f}', ha='center', color='#e6edf3', fontsize=9)
    _styled_title(ax_mc, 'Model Comparison\n(R² Score)')

    _save_show(fig, 'ml_insights', output_dir)


# ─────────────────────────────────────────────────────────────
#  JOB MARKET INTELLIGENCE
# ─────────────────────────────────────────────────────────────
def job_market_intelligence(df: pd.DataFrame, output_dir: str):
    print(f"\n{'═'*60}")
    print("  💼  JOB MARKET INTELLIGENCE")
    print(f"{'═'*60}")

    fig = plt.figure(figsize=(22, 14), facecolor='#0d1117')
    fig.suptitle('JOB MARKET INTELLIGENCE DASHBOARD',
                 fontsize=18, color=COLORS['accent'], fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Safest Industries (lowest avg layoff %) ───────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if 'Industry' in df.columns and 'Layoff_Pct' in df.columns:
        safe = df.groupby('Industry')['Layoff_Pct'].mean().nsmallest(8).sort_values(ascending=False)
        ax1.barh(safe.index, np.array(safe.values, dtype=float), color=COLORS['hire'])
        ax1.set_xlabel('Avg Layoff %')
        ax1.tick_params(labelsize=8)
    elif 'Industry' in df.columns and 'Layoffs' in df.columns and 'Total_Employees' in df.columns:
        df_temp = df[df['Total_Employees'] > 0].copy()
        df_temp['_pct'] = df_temp['Layoffs'] / df_temp['Total_Employees'] * 100
        safe = df_temp.groupby('Industry')['_pct'].mean().nsmallest(8).sort_values(ascending=False)
        ax1.barh(safe.index, np.array(safe.values, dtype=float), color=COLORS['hire'])
        ax1.set_xlabel('Avg Layoff %')
        ax1.tick_params(labelsize=8)
    _styled_title(ax1, '🟢 Safest Industries\n(Lowest Layoff Rate)')

    # ── Fastest growing industries ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Industry' in df.columns and 'Hiring' in df.columns:
        growth = df.groupby('Industry')['Hiring'].sum().nlargest(8).sort_values()
        ax2.barh(growth.index, np.array(growth.values, dtype=float),
                 color=plt.colormaps['Greens'](np.linspace(0.4, 1, len(growth))))
        ax2.set_xlabel('Total Hires')
        ax2.tick_params(labelsize=8)
    _styled_title(ax2, '🚀 Top Hiring\nIndustries')

    # ── Recovery ratio (hiring/layoff) ───────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if 'Industry' in df.columns and 'Layoffs' in df.columns and 'Hiring' in df.columns:
        grp = df.groupby('Industry')[['Layoffs', 'Hiring']].sum()
        grp = grp[(grp['Layoffs'] > 0) & (grp['Hiring'] > 0)]
        grp['Recovery_Ratio'] = grp['Hiring'] / grp['Layoffs']
        top_r = grp['Recovery_Ratio'].nlargest(8).sort_values()
        clrs_r = [COLORS['hire'] if r >= 1 else COLORS['layoff'] for r in np.array(top_r.values, dtype=float)]
        ax3.barh(top_r.index, np.array(top_r.values, dtype=float), color=clrs_r)
        ax3.axvline(1.0, color=COLORS['accent'], lw=1.5, ls='--', label='Break-even')
        ax3.set_xlabel('Hiring / Layoff Ratio')
        ax3.tick_params(labelsize=8)
        ax3.legend(framealpha=0.2, fontsize=8)
    _styled_title(ax3, '⚖️ Industry Recovery\nRatio (Hire/Layoff)')

    # ── Recent 5-year trend per industry ─────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    recent_years = sorted(df['Year'].dropna().unique())[-5:]
    df_recent = df[df['Year'].isin(recent_years)]
    if 'Industry' in df_recent.columns and 'Layoffs' in df_recent.columns:
        piv = df_recent.pivot_table(values='Layoffs', index='Year', columns='Industry', aggfunc='sum').fillna(0)
        top_i = piv.sum().nlargest(6).index
        for ind in top_i:
            if ind in piv.columns:
                ax4.plot(piv.index, piv[ind], marker='o', ms=5, lw=2, label=ind)
        ax4.legend(framealpha=0.2, fontsize=8)
        ax4.grid(True, alpha=0.3)
    _styled_title(ax4, f'Recent 5-Year Layoff Trends by Industry ({recent_years[0]}–{recent_years[-1]})')

    # ── Country opportunity (high hire, low layoff) ───────────
    ax5 = fig.add_subplot(gs[1, 2])
    if 'Country' in df.columns and 'Layoffs' in df.columns and 'Hiring' in df.columns:
        ctry_grp = df.groupby('Country')[['Layoffs', 'Hiring']].sum()
        ctry_grp = ctry_grp[(ctry_grp['Layoffs'] > 0) | (ctry_grp['Hiring'] > 0)]
        ctry_grp['Score'] = ctry_grp['Hiring'].fillna(0) - ctry_grp['Layoffs'].fillna(0)
        top_opp = ctry_grp['Score'].nlargest(8).sort_values()
        clrs_co = [COLORS['hire'] if s > 0 else COLORS['layoff'] for s in np.array(top_opp.values, dtype=float)]
        ax5.barh(top_opp.index, np.array(top_opp.values, dtype=float), color=clrs_co)
        ax5.axvline(0, color='#8b949e', lw=1.2)
        ax5.tick_params(labelsize=8)
    _styled_title(ax5, '🌏 Country Opportunity\nScore (Hire − Layoff)')

    # Print top picks
    print("\n  📌 QUICK MARKET INTELLIGENCE:")
    if 'Industry' in df.columns and 'Hiring' in df.columns:
        top3 = df.groupby('Industry')['Hiring'].sum().nlargest(3).index.tolist()
        print(f"     Top 3 Hiring Industries : {', '.join(top3)}")
    if 'Industry' in df.columns and 'Layoffs' in df.columns:
        worst3 = df.groupby('Industry')['Layoffs'].sum().nlargest(3).index.tolist()
        print(f"     Top 3 Layoff Industries : {', '.join(worst3)}")
    if 'Country' in df.columns and 'Hiring' in df.columns:
        top_c = df.groupby('Country')['Hiring'].sum().idxmax()
        print(f"     Highest Hiring Country  : {top_c}")

    _save_show(fig, 'job_market_intelligence', output_dir)


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────
def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║     TECH HIRING & LAYOFFS  ◆  WORKFORCE ANALYZER v1.0       ║
║     Dataset: Kaggle 2000–2025  |  ML-powered insights        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    print_banner()

    # ── Find dataset ──────────────────────────────────────────
    search_names = [
        'tech_hiring_layoffs.csv',
        'tech_hiring_and_layoffs.csv',
        'tech-hiring-layoffs.csv',
        'hiring_layoffs.csv',
        'workforce_data.csv',
    ]
    # Also look for any CSV in current dir
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    search_names = search_names + csv_files

    df = None
    for name in search_names:
        if os.path.exists(name):
            df = load_data(name)
            break

    if df is None:
        print("\n⚠️  No dataset CSV found in the current directory.")
        filepath = input("   Enter the full path to your CSV file: ").strip().strip('"')
        df = load_data(filepath)

    # ── Output directory ──────────────────────────────────────
    output_dir = 'workforce_output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁  Charts will be saved to: ./{output_dir}/")

    available_years = sorted(df['Year'].dropna().unique().tolist())
    print(f"\n📅  Available years: {available_years[0]} – {available_years[-1]}")

    # ── Menu loop ─────────────────────────────────────────────
    while True:
        print("""
┌─────────────────────────────────────────────────┐
│                   MAIN MENU                     │
│  1 → Global Overview (all years)                │
│  2 → Deep Dive (specific year)                  │
│  3 → Year Range Analysis                        │
│  4 → ML Models & Predictions                    │
│  5 → Job Market Intelligence                    │
│  6 → Full Report (run all)                      │
│  7 → Dataset Summary                            │
│  0 → Exit                                       │
└─────────────────────────────────────────────────┘""")
        choice = input("  Choose an option: ").strip()

        if choice == '0':
            print("\n👋  Goodbye!\n")
            break

        elif choice == '1':
            global_overview(df, output_dir)

        elif choice == '2':
            yr_input = input(f"  Enter year {available_years[0]}–{available_years[-1]}: ").strip()
            try:
                year = int(yr_input)
                year_deep_dive(df, year, output_dir)
            except ValueError:
                print("  ❌  Please enter a valid 4-digit year.")

        elif choice == '3':
            try:
                start = int(input(f"  Start year: ").strip())
                end   = int(input(f"  End year  : ").strip())
                year_range_analysis(df, start, end, output_dir)
            except ValueError:
                print("  ❌  Invalid year input.")

        elif choice == '4':
            run_ml_models(df, output_dir)

        elif choice == '5':
            job_market_intelligence(df, output_dir)

        elif choice == '6':
            print("\n🚀  Running full report...")
            global_overview(df, output_dir)
            job_market_intelligence(df, output_dir)
            run_ml_models(df, output_dir)
            # Deep dive on most recent year
            latest = int(df['Year'].max())
            year_deep_dive(df, latest, output_dir)
            print("\n✅  Full report complete! Check ./workforce_output/")

        elif choice == '7':
            print(f"\n{'─'*50}")
            print(f"  Columns     : {list(df.columns)}")
            print(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} cols")
            print(f"  Year Range  : {df['Year'].min()} – {df['Year'].max()}")
            print(f"  Missing %   :")
            for c in df.columns:
                pct = df[c].isna().mean() * 100
                if pct > 0:
                    print(f"    {c:<25}: {pct:.1f}%")
            print(df.describe(include='all').T[['count','mean','min','max']].to_string())
            print(f"{'─'*50}")
        else:
            print("  ⚠️  Invalid option. Please choose 0–7.")


if __name__ == '__main__':
    main()