"""
Data Exploratory Analysis for Stroke Risk Dataset
Generates various visualizations and saves them as PDFs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_PATH = "stroke_risk_dataset.csv"
OUTPUT_DIR = Path("visualized_pdfs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load the stroke risk dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def save_figure(fig, filename):
    """Save figure as PDF without title (filename is the title)."""
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}.pdf")


def plot_age_distribution(df):
    """Plot age distribution with risk overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram with KDE
    sns.histplot(data=df, x='Age', hue='At Risk (Binary)',
                 kde=True, ax=ax, bins=30, alpha=0.6)

    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(['Not At Risk', 'At Risk'], loc='upper right')

    # Add annotation for mean ages
    mean_risk = df[df['At Risk (Binary)'] == 1]['Age'].mean()
    mean_no_risk = df[df['At Risk (Binary)'] == 0]['Age'].mean()
    ax.annotate(f'Mean Age (At Risk): {mean_risk:.1f}', xy=(0.02, 0.95),
                xycoords='axes fraction', fontsize=10)
    ax.annotate(f'Mean Age (Not At Risk): {mean_no_risk:.1f}', xy=(0.02, 0.90),
                xycoords='axes fraction', fontsize=10)

    save_figure(fig, "age_distribution_by_risk")


def plot_stroke_risk_distribution(df):
    """Plot stroke risk percentage distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    sns.histplot(data=df, x='Stroke Risk (%)', kde=True, ax=axes[0],
                 bins=30, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Stroke Risk (%)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].axvline(df['Stroke Risk (%)'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['Stroke Risk (%)'].mean():.1f}%")
    axes[0].axvline(df['Stroke Risk (%)'].median(), color='green', linestyle='--',
                    label=f"Median: {df['Stroke Risk (%)'].median():.1f}%")
    axes[0].legend()

    # Box plot by risk category
    sns.boxplot(data=df, x='At Risk (Binary)', y='Stroke Risk (%)', ax=axes[1],
                palette=['lightgreen', 'salmon'])
    axes[1].set_xlabel('At Risk Category', fontsize=12)
    axes[1].set_ylabel('Stroke Risk (%)', fontsize=12)
    axes[1].set_xticklabels(['Not At Risk (0)', 'At Risk (1)'])

    save_figure(fig, "stroke_risk_percentage_distribution")


def plot_correlation_heatmap(df):
    """Plot point-biserial correlation between categorical features and continuous targets."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Separate categorical (binary) and continuous columns
    continuous_cols = ['Age', 'Stroke Risk (%)']
    categorical_cols = [col for col in df.columns if col not in continuous_cols]

    # Calculate point-biserial correlations (categorical vs continuous targets)
    corr_data = []
    for cat_col in categorical_cols:
        corr_age = df[cat_col].corr(df['Age'])
        corr_risk = df[cat_col].corr(df['Stroke Risk (%)'])
        corr_data.append({
            'Feature': cat_col,
            'Age': corr_age,
            'Stroke Risk (%)': corr_risk
        })

    corr_df = pd.DataFrame(corr_data).set_index('Feature')
    corr_df = corr_df.sort_values('Stroke Risk (%)', ascending=False)

    # Plot heatmap
    sns.heatmap(corr_df, annot=True, fmt='.3f',
                cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.5,
                annot_kws={'size': 9},
                cbar_kws={'label': 'Point-Biserial Correlation'})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel('Continuous Variables', fontsize=12)
    ax.set_ylabel('Categorical Features', fontsize=12)

    save_figure(fig, "pointbiserial_correlation_heatmap")


def plot_symptom_frequency(df):
    """Plot frequency of each symptom."""
    # Get symptom columns (binary features)
    symptom_cols = [col for col in df.columns if col not in ['Age', 'Stroke Risk (%)', 'At Risk (Binary)']]

    # Calculate frequencies
    symptom_freq = df[symptom_cols].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(symptom_freq)))
    bars = ax.barh(range(len(symptom_freq)), symptom_freq.values, color=colors)

    ax.set_yticks(range(len(symptom_freq)))
    ax.set_yticklabels(symptom_freq.index, fontsize=10)
    ax.set_xlabel('Frequency (Number of Patients)', fontsize=12)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, symptom_freq.values)):
        ax.text(val + 5, i, f'{val}', va='center', fontsize=9)

    # Add percentage annotation
    total = len(df)
    ax.annotate(f'Total patients: {total}', xy=(0.95, 0.05),
                xycoords='axes fraction', fontsize=10, ha='right')

    save_figure(fig, "symptom_frequency")


def plot_symptom_vs_risk(df):
    """Plot symptom presence vs stroke risk."""
    symptom_cols = [col for col in df.columns if col not in ['Age', 'Stroke Risk (%)', 'At Risk (Binary)']]

    # Calculate mean stroke risk for each symptom presence/absence
    risk_by_symptom = []
    for col in symptom_cols:
        risk_present = df[df[col] == 1]['Stroke Risk (%)'].mean()
        risk_absent = df[df[col] == 0]['Stroke Risk (%)'].mean()
        risk_by_symptom.append({
            'Symptom': col,
            'Present': risk_present,
            'Absent': risk_absent,
            'Difference': risk_present - risk_absent
        })

    risk_df = pd.DataFrame(risk_by_symptom).sort_values('Difference', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(risk_df))
    width = 0.35

    bars1 = ax.barh(x - width/2, risk_df['Absent'], width, label='Symptom Absent',
                    color='lightblue', alpha=0.8)
    bars2 = ax.barh(x + width/2, risk_df['Present'], width, label='Symptom Present',
                    color='coral', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(risk_df['Symptom'], fontsize=9)
    ax.set_xlabel('Mean Stroke Risk (%)', fontsize=12)
    ax.legend(loc='lower right')

    # Add difference annotations
    for i, (idx, row) in enumerate(risk_df.iterrows()):
        diff = row['Difference']
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'{diff:+.1f}%', xy=(max(row['Present'], row['Absent']) + 1, i),
                   fontsize=8, color=color, va='center')

    save_figure(fig, "symptom_impact_on_stroke_risk")


def plot_binary_risk_distribution(df):
    """Plot distribution of binary risk classification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    risk_counts = df['At Risk (Binary)'].value_counts()
    colors = ['lightgreen', 'salmon']
    explode = (0.05, 0.05)

    axes[0].pie(risk_counts.values, explode=explode, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
    axes[0].legend(['Not At Risk (0)', 'At Risk (1)'], loc='lower left')

    # Bar chart with counts
    bars = axes[1].bar(['Not At Risk', 'At Risk'], risk_counts.values,
                       color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Number of Patients', fontsize=12)

    # Add count labels
    for bar, count in zip(bars, risk_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}', ha='center', fontsize=12, fontweight='bold')

    save_figure(fig, "binary_risk_class_distribution")


def plot_age_vs_stroke_risk(df):
    """Plot age vs stroke risk scatter with regression."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with hue for binary risk
    scatter = ax.scatter(df['Age'], df['Stroke Risk (%)'],
                        c=df['At Risk (Binary)'], cmap='RdYlGn_r',
                        alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    # Add regression line
    z = np.polyfit(df['Age'], df['Stroke Risk (%)'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Age'].min(), df['Age'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Linear fit (slope={z[0]:.2f})')

    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Stroke Risk (%)', fontsize=12)
    ax.legend(loc='upper left')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('At Risk (Binary)', fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not At Risk', 'At Risk'])

    # Add correlation annotation
    corr = df['Age'].corr(df['Stroke Risk (%)'])
    ax.annotate(f'Correlation: {corr:.3f}', xy=(0.02, 0.95),
                xycoords='axes fraction', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_figure(fig, "age_vs_stroke_risk_scatter")


def plot_symptom_count_analysis(df):
    """Analyze relationship between number of symptoms and risk."""
    symptom_cols = [col for col in df.columns if col not in ['Age', 'Stroke Risk (%)', 'At Risk (Binary)']]

    # Calculate symptom count for each patient
    df['Symptom_Count'] = df[symptom_cols].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot: Symptom count vs Stroke Risk
    sns.boxplot(data=df, x='Symptom_Count', y='Stroke Risk (%)', ax=axes[0],
                palette='coolwarm')
    axes[0].set_xlabel('Number of Symptoms', fontsize=12)
    axes[0].set_ylabel('Stroke Risk (%)', fontsize=12)

    # Bar plot: Symptom count vs At Risk percentage
    risk_by_count = df.groupby('Symptom_Count')['At Risk (Binary)'].agg(['mean', 'count'])
    risk_by_count['mean'] = risk_by_count['mean'] * 100

    bars = axes[1].bar(risk_by_count.index, risk_by_count['mean'],
                       color=plt.cm.RdYlGn_r(risk_by_count['mean']/100),
                       edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Number of Symptoms', fontsize=12)
    axes[1].set_ylabel('Percentage At Risk (%)', fontsize=12)

    # Add count annotations
    for i, (idx, row) in enumerate(risk_by_count.iterrows()):
        axes[1].annotate(f'n={int(row["count"])}', xy=(idx, row['mean'] + 2),
                        ha='center', fontsize=8)

    # Clean up
    df.drop('Symptom_Count', axis=1, inplace=True)

    save_figure(fig, "symptom_count_vs_risk_analysis")


def plot_top_risk_factors(df):
    """Identify and plot top risk factors based on correlation with stroke risk."""
    # Calculate correlations with both targets
    corr_continuous = df.corr()['Stroke Risk (%)'].drop(['Stroke Risk (%)', 'At Risk (Binary)'])
    corr_binary = df.corr()['At Risk (Binary)'].drop(['Stroke Risk (%)', 'At Risk (Binary)'])

    # Combine into dataframe
    corr_df = pd.DataFrame({
        'Stroke Risk (%)': corr_continuous,
        'At Risk (Binary)': corr_binary
    }).sort_values('Stroke Risk (%)', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(corr_df))
    width = 0.35

    bars1 = ax.barh(x - width/2, corr_df['Stroke Risk (%)'], width,
                    label='Correlation with Stroke Risk (%)', color='steelblue', alpha=0.8)
    bars2 = ax.barh(x + width/2, corr_df['At Risk (Binary)'], width,
                    label='Correlation with At Risk (Binary)', color='coral', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(corr_df.index, fontsize=10)
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars1, corr_df['Stroke Risk (%)']):
        ax.text(val + 0.01 if val >= 0 else val - 0.05, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=8)

    save_figure(fig, "feature_correlation_with_targets")


def plot_categorical_risk_comparison(df):
    """Create grouped bar chart comparing risk rates across all categorical features."""
    symptom_cols = [col for col in df.columns if col not in ['Age', 'Stroke Risk (%)', 'At Risk (Binary)']]

    # Calculate at-risk rate for each symptom presence/absence
    risk_rates = []
    for col in symptom_cols:
        rate_present = df[df[col] == 1]['At Risk (Binary)'].mean() * 100
        rate_absent = df[df[col] == 0]['At Risk (Binary)'].mean() * 100
        risk_rates.append({
            'Feature': col,
            'Present': rate_present,
            'Absent': rate_absent,
            'Lift': rate_present - rate_absent
        })

    risk_df = pd.DataFrame(risk_rates).sort_values('Lift', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(risk_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, risk_df['Absent'], width, label='Feature Absent',
                   color='lightblue', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, risk_df['Present'], width, label='Feature Present',
                   color='salmon', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(risk_df['Feature'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('At Risk Rate (%)', fontsize=12)
    ax.set_xlabel('Categorical Features', fontsize=12)
    ax.legend(loc='upper right')

    # Add lift annotations on top
    for i, (idx, row) in enumerate(risk_df.iterrows()):
        lift = row['Lift']
        color = 'green' if lift > 0 else 'red'
        max_height = max(row['Present'], row['Absent'])
        ax.annotate(f'{lift:+.1f}%', xy=(i, max_height + 1),
                   ha='center', fontsize=7, color=color, fontweight='bold')

    ax.set_ylim(0, ax.get_ylim()[1] + 5)

    save_figure(fig, "categorical_risk_rate_comparison")


def plot_violin_symptoms(df):
    """Create violin plots for stroke risk by symptom presence."""
    symptom_cols = [col for col in df.columns if col not in ['Age', 'Stroke Risk (%)', 'At Risk (Binary)']]

    # Select top 6 symptoms by correlation
    corr_with_risk = df[symptom_cols].corrwith(df['Stroke Risk (%)']).abs().nlargest(6)
    top_symptoms = corr_with_risk.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, symptom in enumerate(top_symptoms):
        sns.violinplot(data=df, x=symptom, y='Stroke Risk (%)', ax=axes[i],
                      palette=['lightblue', 'coral'], inner='box')
        axes[i].set_xlabel(f'{symptom}', fontsize=10)
        axes[i].set_ylabel('Stroke Risk (%)' if i % 3 == 0 else '', fontsize=10)
        axes[i].set_xticklabels(['Absent', 'Present'])

        # Add mean annotation
        mean_absent = df[df[symptom] == 0]['Stroke Risk (%)'].mean()
        mean_present = df[df[symptom] == 1]['Stroke Risk (%)'].mean()
        axes[i].annotate(f'Mean: {mean_absent:.1f}%', xy=(0, axes[i].get_ylim()[1]-5),
                        ha='center', fontsize=8)
        axes[i].annotate(f'Mean: {mean_present:.1f}%', xy=(1, axes[i].get_ylim()[1]-5),
                        ha='center', fontsize=8)

    plt.tight_layout()
    save_figure(fig, "violin_top_symptoms_vs_stroke_risk")


def plot_heatmap_symptom_cooccurrence(df):
    """Plot heatmap of symptom co-occurrence."""
    symptom_cols = [col for col in df.columns if col not in ['Age', 'Stroke Risk (%)', 'At Risk (Binary)']]

    # Calculate co-occurrence matrix
    symptom_df = df[symptom_cols]
    cooccurrence = symptom_df.T.dot(symptom_df)

    # Normalize by diagonal (convert to percentage)
    diag = np.diag(cooccurrence)
    cooccurrence_pct = cooccurrence / diag[:, None] * 100

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(cooccurrence_pct, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=ax, square=True, linewidths=0.5,
                annot_kws={'size': 8},
                cbar_kws={'label': 'Co-occurrence (%)'})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    save_figure(fig, "symptom_cooccurrence_heatmap")


def plot_risk_distribution_by_age_group(df):
    """Plot risk distribution by age groups."""
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100],
                             labels=['<30', '30-45', '45-60', '60-75', '75+'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of stroke risk by age group
    sns.boxplot(data=df, x='Age_Group', y='Stroke Risk (%)', ax=axes[0],
                palette='coolwarm', order=['<30', '30-45', '45-60', '60-75', '75+'])
    axes[0].set_xlabel('Age Group', fontsize=12)
    axes[0].set_ylabel('Stroke Risk (%)', fontsize=12)

    # Stacked bar: At Risk proportion by age group
    risk_by_age = df.groupby(['Age_Group', 'At Risk (Binary)']).size().unstack(fill_value=0)
    risk_by_age_pct = risk_by_age.div(risk_by_age.sum(axis=1), axis=0) * 100

    risk_by_age_pct.plot(kind='bar', stacked=True, ax=axes[1],
                         color=['lightgreen', 'salmon'], edgecolor='black')
    axes[1].set_xlabel('Age Group', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(['Not At Risk', 'At Risk'], loc='upper left')

    # Add count annotations
    for i, (idx, row) in enumerate(risk_by_age.iterrows()):
        total = row.sum()
        axes[1].annotate(f'n={total}', xy=(i, 105), ha='center', fontsize=9)

    # Clean up
    df.drop('Age_Group', axis=1, inplace=True)

    save_figure(fig, "risk_distribution_by_age_group")


def main():
    """Run all exploratory data analysis visualizations."""
    print("Loading data...")
    df = load_data()

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTarget distributions:")
    print(f"  Stroke Risk (%): mean={df['Stroke Risk (%)'].mean():.2f}, std={df['Stroke Risk (%)'].std():.2f}")
    print(f"  At Risk (Binary): {df['At Risk (Binary)'].value_counts().to_dict()}")

    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50 + "\n")

    # Generate all visualizations
    plot_age_distribution(df)
    plot_stroke_risk_distribution(df)
    plot_binary_risk_distribution(df)
    plot_correlation_heatmap(df)
    plot_symptom_frequency(df)
    plot_symptom_vs_risk(df)
    plot_age_vs_stroke_risk(df)
    plot_symptom_count_analysis(df)
    plot_top_risk_factors(df)
    plot_categorical_risk_comparison(df)
    plot_violin_symptoms(df)
    plot_heatmap_symptom_cooccurrence(df)
    plot_risk_distribution_by_age_group(df)

    print("\n" + "="*50)
    print("All visualizations saved to 'visualized_pdfs/' directory")
    print("="*50)


if __name__ == "__main__":
    main()
