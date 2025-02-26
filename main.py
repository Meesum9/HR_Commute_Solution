import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

def create_commute_matrix(df, max_per_branch):
    """Create commute time matrix from DataFrame with branch capacity constraints."""
    branch_cols = ['BranchA', 'BranchB', 'BranchC', 'BranchD']
    n_employees = len(df)
    n_branches = len(branch_cols)
    
    # Create expanded matrix to handle branch capacity
    total_slots = n_branches * max_per_branch
    commute_times = np.full((n_employees, total_slots), np.inf)
    
    # Fill the matrix with actual commute times, repeated for each slot
    for i in range(max_per_branch):
        start_col = i * n_branches
        end_col = (i + 1) * n_branches
        commute_times[:, start_col:end_col] = df[branch_cols].values
    
    return commute_times, df['Employee Name'].values, branch_cols

def optimize_assignments(commute_times):
    """Apply Hungarian Algorithm to optimize assignments."""
    return linear_sum_assignment(commute_times)

def print_results(employees, branches, commute_times, row_ind, col_ind, max_per_branch):
    """Print the optimization results."""
    print("\nOptimal Employee to Branch Assignments:")
    print("----------------------------------------")
    
    # Create a summary dictionary to count employees per branch
    branch_assignments = {branch: [] for branch in branches}
    n_branches = len(branches)
    
    # Print individual assignments and collect summary data
    for emp, loc in zip(row_ind, col_ind):
        employee_name = employees[emp]
        actual_branch_idx = loc % n_branches
        branch_name = branches[actual_branch_idx]
        commute = commute_times[emp, loc]
        
        print(f"Employee: {employee_name}")
        print(f"Assigned to: {branch_name}")
        print(f"Commute time: {commute:.2f} minutes")
        print("----------------------------------------")
        
        # Collect for summary
        branch_assignments[branch_name].append((employee_name, commute))

    # Print summary statistics
    valid_commute_times = commute_times[row_ind, col_ind]
    total_commute_time = valid_commute_times.sum()
    avg_commute_time = total_commute_time / len(employees)
    
    print("\nSummary Statistics:")
    print("===================")
    print(f"Total Employees Assigned: {len(employees)}")
    print(f"Maximum employees per branch: {max_per_branch}")
    print(f"Total Commute Time Saved : {total_commute_time:.2f} minutes")
    print(f"Average Commute Time: {avg_commute_time:.2f} minutes")
    
    # Print branch-wise distribution
    print("\nBranch-wise Distribution:")
    print("========================")
    for branch in branches:
        assigned_employees = branch_assignments[branch]
        print(f"\n{branch} ({len(assigned_employees)} employees):")
        for emp_name, commute in assigned_employees:
            print(f"  - {emp_name} ({commute:.2f} minutes)")

def main():
    try:
        # Read the CSV file
        df = pd.read_csv('employee_commute_data.csv')
        print("Successfully loaded the CSV file!")
        
        # Set maximum employees per branch (adjust as needed)
        max_per_branch = 5  # This allows up to 5 employees per branch
        
        # Create commute time matrix
        commute_times, employees, branches = create_commute_matrix(df, max_per_branch)
        
        # Verify data dimensions
        print(f"\nProcessing data for:")
        print(f"Number of employees: {len(employees)}")
        print(f"Number of branches: {len(branches)}")
        print(f"Maximum employees per branch: {max_per_branch}")
        print(f"Commute time matrix shape: {commute_times.shape}")
        
        if len(employees) < 1:
            raise ValueError("No employees found in the data!")
            
        # Apply Hungarian Algorithm
        row_ind, col_ind = optimize_assignments(commute_times)
        
        # Print detailed results
        print_results(employees, branches, commute_times, row_ind, col_ind, max_per_branch)
        
    except FileNotFoundError:
        print("Error: Could not find 'employee_commute_data.csv'. Please make sure the file exists in the same directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
