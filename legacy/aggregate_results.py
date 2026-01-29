#!/usr/bin/env python
"""
Aggregate CV and cross-disease test results into a master table.

Usage:
    python aggregate_results.py
    python aggregate_results.py --datasets SCP1884 Skin3 --output master_table.csv
    # Conditional 결과 집계
    python aggregate_results.py \
        --datasets SCP1884 Skin3 \
        --res_dir_prefix "/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_" \
        --cross_dir_prefix "./cross_perfs_conditional_" \
        --output master_table_conditional.csv \
        --summary_output summary_table_conditional.csv
    python aggregate_results.py \
        --datasets SCP1884 \
        --res_dir_prefix "/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v2_" \
        --cross_dir_prefix "./cross_perfs_conditional_v2_" \
        --output 0D_v2_master_table_conditional.csv \
        --summary_output 0D_v2_summary_table_conditional.csv
    python aggregate_results.py \
        --datasets SCP1884 \
        --res_dir_prefix "/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_" \
        --cross_dir_prefix "./cross_perfs_conditional_v3_" \
        --output 0D_v3_master_table_conditional.csv \
        --summary_output 0D_v3_summary_table_conditional_.csv
    python aggregate_results.py \
        --datasets SCP1884 Skin3 \
        --res_dir_prefix "/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_" \
        --cross_dir_prefix "./cross_perfs_conditional_v3_" \
        --output 0D_v3_master_table_conditional.csv \
        --summary_output 0D_v3_summary_table_conditional.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def load_cv_results(res_dir_prefix, dataset):
    """Load CV results from individual model CSV files."""
    res_dir = f"{res_dir_prefix}{dataset}"
    
    if not os.path.exists(res_dir):
        print(f"Warning: CV results directory not found: {res_dir}")
        return pd.DataFrame()
    
    cv_results = []
    
    # Find all CSV files (not in subdirectories)
    csv_files = glob(f"{res_dir}/*.csv")
    
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('.csv', '')
        
        try:
            df = pd.read_csv(csv_file)
            if 'AUC' in df.columns:
                cv_results.append({
                    'model': model_name,
                    'trained': dataset,
                    'cv_auc_mean': df['AUC'].mean(),
                    'cv_auc_std': df['AUC'].std(),
                    'cv_n_exps': len(df)
                })
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")
            continue
    
    return pd.DataFrame(cv_results)


def load_test_results(cross_dir_prefix, dataset):
    """Load cross-disease test results."""
    cross_dir = f"{cross_dir_prefix}{dataset}"
    
    if not os.path.exists(cross_dir):
        print(f"Warning: Test results directory not found: {cross_dir}")
        return pd.DataFrame()
    
    # Try different file naming patterns
    possible_files = [
        f"{cross_dir}/perfs_exps.csv",
        f"{cross_dir}/perfs_exps_colon.csv",
        f"{cross_dir}/perfs_exps_{dataset}.csv"
    ]
    
    test_df = None
    for f in possible_files:
        if os.path.exists(f):
            test_df = pd.read_csv(f)
            print(f"Loaded test results from: {f}")
            break
    
    if test_df is None:
        print(f"Warning: No test results file found in {cross_dir}")
        return pd.DataFrame()
    
    # Group by model and tested dataset, calculate mean/std
    grouped = test_df.groupby(['model', 'trained', 'tested']).agg({
        'auc': ['mean', 'std', 'count'],
        'auprc': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['model', 'trained', 'tested', 
                       'test_auc_mean', 'test_auc_std', 'test_n_exps',
                       'test_auprc_mean', 'test_auprc_std']
    
    return grouped


def create_master_table(datasets, res_dir_prefix, cross_dir_prefix):
    """Create master table combining CV and test results."""
    
    all_cv_results = []
    all_test_results = []
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Load CV results
        cv_df = load_cv_results(res_dir_prefix, dataset)
        if not cv_df.empty:
            all_cv_results.append(cv_df)
            print(f"  CV results: {len(cv_df)} models")
        
        # Load test results
        test_df = load_test_results(cross_dir_prefix, dataset)
        if not test_df.empty:
            all_test_results.append(test_df)
            print(f"  Test results: {len(test_df)} model-dataset combinations")
    
    # Combine all results
    if all_cv_results:
        cv_combined = pd.concat(all_cv_results, ignore_index=True)
    else:
        cv_combined = pd.DataFrame()
    
    if all_test_results:
        test_combined = pd.concat(all_test_results, ignore_index=True)
    else:
        test_combined = pd.DataFrame()
    
    # Merge CV and test results
    if not cv_combined.empty and not test_combined.empty:
        master_table = test_combined.merge(
            cv_combined[['model', 'trained', 'cv_auc_mean', 'cv_auc_std', 'cv_n_exps']],
            on=['model', 'trained'],
            how='left'
        )
    elif not cv_combined.empty:
        master_table = cv_combined
    elif not test_combined.empty:
        master_table = test_combined
    else:
        print("Error: No results found!")
        return pd.DataFrame()
    
    # Reorder columns
    col_order = ['model', 'trained', 'cv_auc_mean', 'cv_auc_std', 'cv_n_exps',
                 'tested', 'test_auc_mean', 'test_auc_std', 'test_auprc_mean', 'test_auprc_std', 'test_n_exps']
    col_order = [c for c in col_order if c in master_table.columns]
    master_table = master_table[col_order]
    
    # Sort
    master_table = master_table.sort_values(['trained', 'model', 'tested']).reset_index(drop=True)
    
    return master_table


def create_summary_table(master_table):
    """Create a summary table: best model per trained dataset."""
    
    if master_table.empty:
        return pd.DataFrame()
    
    summary = []
    
    for trained in master_table['trained'].unique():
        subset = master_table[master_table['trained'] == trained]
        
        # Best model by CV AUC
        if 'cv_auc_mean' in subset.columns:
            best_cv = subset.drop_duplicates('model').nlargest(1, 'cv_auc_mean')
            if not best_cv.empty:
                summary.append({
                    'trained': trained,
                    'best_by': 'cv_auc',
                    'model': best_cv['model'].values[0],
                    'cv_auc_mean': best_cv['cv_auc_mean'].values[0],
                    'cv_auc_std': best_cv['cv_auc_std'].values[0]
                })
        
        # Average test performance per model
        if 'test_auc_mean' in subset.columns:
            model_avg = subset.groupby('model').agg({
                'test_auc_mean': 'mean',
                'test_auprc_mean': 'mean'
            }).reset_index()
            
            best_test = model_avg.nlargest(1, 'test_auc_mean')
            if not best_test.empty:
                summary.append({
                    'trained': trained,
                    'best_by': 'avg_test_auc',
                    'model': best_test['model'].values[0],
                    'avg_test_auc': best_test['test_auc_mean'].values[0],
                    'avg_test_auprc': best_test['test_auprc_mean'].values[0]
                })
    
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description='Aggregate CV and test results into master table')
    
    parser.add_argument('--datasets', nargs='+', default=['SCP1884', 'Skin3'],
                        help='List of trained datasets')
    parser.add_argument('--res_dir_prefix', type=str, 
                        default='/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_',
                        help='Prefix for CV results directory')
    parser.add_argument('--cross_dir_prefix', type=str,
                        default='./cross_perfs_all_quantized_freeze_and_projection_',
                        help='Prefix for cross-disease test results directory')
    parser.add_argument('--output', type=str, default='master_table.csv',
                        help='Output CSV filename for master table')
    parser.add_argument('--summary_output', type=str, default='summary_table.csv',
                        help='Output CSV filename for summary table')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Aggregating results")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"CV results prefix: {args.res_dir_prefix}")
    print(f"Test results prefix: {args.cross_dir_prefix}")
    print("=" * 60)
    
    # Create master table
    master_table = create_master_table(
        datasets=args.datasets,
        res_dir_prefix=args.res_dir_prefix,
        cross_dir_prefix=args.cross_dir_prefix
    )
    
    if not master_table.empty:
        master_table.to_csv(args.output, index=False)
        print(f"\nMaster table saved to: {args.output}")
        print(f"Total rows: {len(master_table)}")
        print(f"\nPreview:")
        print(master_table.head(10).to_string())
        
        # Create summary table
        summary_table = create_summary_table(master_table)
        if not summary_table.empty:
            summary_table.to_csv(args.summary_output, index=False)
            print(f"\nSummary table saved to: {args.summary_output}")
            print(summary_table.to_string())
    else:
        print("No results to save!")


if __name__ == '__main__':
    main()