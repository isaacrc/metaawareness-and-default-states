import os
import numpy as np

# Define behavioral data directory
behav_dir = '../../data/behavioral/' # Update path as needed
dataDir = '../../data/work/' # Update path as needed

# List of subjects
file_list = sorted(os.listdir(behav_dir))[1:]
subjects = file_list[1:15] + file_list[16:-2] # exclude subs w/o bpress
print(f'num subs: {len(subjects)} listed as {subjects[0]}')



# Directory to save results
save_dir = os.path.join(dataDir, "button_press_counts/")
os.makedirs(save_dir, exist_ok=True)

def process_subject(sub):
    """Processes a single subject's behavioral data."""
    file_path = os.path.join(behav_dir, f"{sub}_behav.npy")
    #print(file_path)
    
    if not os.path.exists(file_path):
        print(f"❌ Missing data for {sub}. Skipping...")
        return None

    # Load behavioral dictionary
    sub_dic = np.load(file_path, allow_pickle=True).item()
    
    # Initialize counts
    button_press_counts = {"External": 0, "Internal": 0}

    # Iterate over conditions (External, Internal)
    for condition in button_press_counts.keys():
        if condition not in sub_dic:
            print(f"⚠️ Warning: {condition} not found for {sub}. Skipping...")
            continue
        
        # Iterate over movies (e.g., 'shrek', 'oragami', etc.)
        for movie in sub_dic[condition]:
            # Iterate over runs
            for run in sub_dic[condition][movie]:
                run_data = sub_dic[condition][movie][run]
                
                # Get 'bpress' list, ensuring it's not -1
                bpress = run_data.get("bpress", [])
                if isinstance(bpress, list):  # Only count valid lists
                    button_press_counts[condition] += len(bpress)  # Count button presses

    # Save results
    save_path = os.path.join(save_dir, f"{sub}_button_press.npy")
    np.save(save_path, button_press_counts)

    print(f"✅ Processed {sub}: {button_press_counts}")
    return button_press_counts


def main():
    """Main function to iterate through all subjects."""
    all_results = {}

    for sub in subjects:
        sub = sub[:7]
        result = process_subject(sub)
        if result:
            all_results[sub] = result

    # Save aggregated results
    np.save(os.path.join(save_dir, "all_subjects_button_presses.npy"), all_results)
    
    print("🎉 Processing complete! All results saved.")


if __name__ == "__main__":
    main()
