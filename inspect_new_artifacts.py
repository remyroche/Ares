
import os
import glob
import pandas as pd
import datetime
import time

OUTCOMES_DIR = "outcomes"

def check_artifacts():
    print(f"--- Checking for new artifacts in {OUTCOMES_DIR} ---")
    
    # Get all files in outcomes
    files = glob.glob(os.path.join(OUTCOMES_DIR, "*"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    current_time = datetime.datetime.now()
    cutoff_time = current_time - datetime.timedelta(minutes=30) # Look for recent files
    
    recent_files = []
    for f in files:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        if mtime > cutoff_time:
            recent_files.append((f, mtime))
            
    if not recent_files:
        print("No recent artifacts found (last 30 mins).")
        return

    print(f"Found {len(recent_files)} recent artifacts:")
    for f, mtime in recent_files[:10]:
        print(f"  {mtime.strftime('%H:%M:%S')} - {os.path.basename(f)}")
        
    # Check for specific L3 inputs
    events_files = [f for f, _ in recent_files if "layer2_events_" in f and f.endswith(".parquet")]
    labels_files = [f for f, _ in recent_files if "layer2_labels_" in f and f.endswith(".parquet")] 
    trials_files = [f for f, _ in recent_files if "layer2_geometry_trials_" in f and f.endswith(".parquet")]
    
    if events_files:
        print(f"\n✅ Found Layer 3 Input (Events): {events_files[0]}")
    else:
        print("\n⏳ Layer 3 Input (Events) NOT found yet.")
        
    if labels_files:
        print(f"✅ Found Layer 3 Input (Labels): {labels_files[0]}")
    else:
        print("⏳ Layer 3 Input (Labels) NOT found yet.")

    if trials_files:
        print(f"\n✅ Found Geometry Trials: {trials_files[0]}")
        try:
            df = pd.read_parquet(trials_files[0])
            print("  Columns:", df.columns.tolist())
            if 'prediction_inverted' in df.columns:
                print("  ✅ 'prediction_inverted' column verification passed.")
                print("  Values distribution:")
                print(df['prediction_inverted'].value_counts())
            elif 'trials' in df.columns:
                 # Check inside 'trials' JSON/Struct
                 print("  Checking inside 'trials' column...")
                 first_val = df['trials'].iloc[0]
                 print(f"  First trial type: {type(first_val)}")
                 # (Parsing logic similar to previous script could go here)
                 if isinstance(first_val, str) and "prediction_inverted" in first_val:
                     print("  ✅ 'prediction_inverted' found in JSON string.")
                 else:
                     print("  ⚠️ 'prediction_inverted' NOT found in standard inspection.")
            else:
                print("  ❌ 'prediction_inverted' column NOT found.")
        except Exception as e:
            print(f"  ❌ Error reading parquet: {e}")
    else:
         print("\n⏳ Geometry Trials NOT found yet.")

if __name__ == "__main__":
    check_artifacts()
