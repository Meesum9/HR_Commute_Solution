import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

def create_commute_matrix(df):
    """Create commute time matrix from DataFrame."""
    # Get branch columns (BranchA through BranchD)
    branch_cols = [col for col in df.columns if col.startswith('Branch')]
    
    # Extract just the commute times
    commute_times = df[branch_cols].values
    
    return commute_times, df['Employee Name'].values, branch_cols

def optimize_assignments(commute_times):
    """Apply Hungarian Algorithm to optimize assignments."""
    return linear_sum_assignment(commute_times)

def print_results(employees, branches, commute_times, row_ind, col_ind):
    """Print the optimization results."""
    print("\nOptimal Employee to Branch Assignments:")
    print("----------------------------------------")
    for emp, loc in zip(row_ind, col_ind):
        print(f"{employees[emp]} → {branches[loc]}")
        print(f"Commute time: {commute_times[emp, loc]:.2f} minutes")
        print("----------------------------------------")

    total_commute_time = commute_times[row_ind, col_ind].sum()
    print(f"\nTotal Optimized Commute Time: {total_commute_time:.2f} minutes")

def main():
    # Read the CSV file
    df = pd.read_csv('Employee_Commute_Time_Data_with_Merged_Address.csv')
    
    # Create commute time matrix
    commute_times, employees, branches = create_commute_matrix(df)
    
    # Apply Hungarian Algorithm
    row_ind, col_ind = optimize_assignments(commute_times)
    
    # Print results
    print_results(employees, branches, commute_times, row_ind, col_ind)

if __name__ == "__main__":
    main()
