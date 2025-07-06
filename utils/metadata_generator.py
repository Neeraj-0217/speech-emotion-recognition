import os 
import pandas as pd

# 1. RAVDESS Parser 
def parse_ravdess_metadata(ravdess_base_path):
    """
    Parses the RAVDESS dataset structure and extracts metadata.
    Focuses on speech audio files.
    """

    file_paths = []
    raw_emotions = []
    genders = []
    actors = []
    
    # Emotion mapping for RAVDESS
    emotion_mapping = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    if not os.path.exists(ravdess_base_path):
        print(f"Warning: RAVDESS speech path not found at {ravdess_base_path}. Skipping RAVDESS parsing.")
        return pd.DataFrame()

    print(f"Parsing RAVDESS from: {ravdess_base_path}")
    for actor_folder in sorted(os.listdir(ravdess_base_path)):
        actor_path = os.path.join(ravdess_base_path, actor_folder)
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    parts = filename.split('.')[0].split('-')
                    if len(parts) != 7:
                        print(f"Warning: Skipping malformed RAVDESS filename: {filename}")
                        continue
                    
                    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor_code = parts

                    if modality == '03' and vocal_channel == '01': # Audio-only, speech
                        full_path = os.path.join(actor_path, filename)
                        
                        emotion = emotion_mapping.get(emotion_code, 'unknown')
                        if emotion == 'unknown':
                            print(f"Warning: Unknown RAVDESS emotion code {emotion_code} in file {filename}. Skipping.")
                            continue
                        
                        actor_id = int(actor_code)
                        gender = 'female' if actor_id % 2 == 0 else 'male'

                        file_paths.append(full_path)
                        raw_emotions.append(emotion)
                        genders.append(gender)
                        actors.append(actor_id)

    df = pd.DataFrame({
        'filepath': file_paths,
        'raw_emotion': raw_emotions,
        'gender': genders,
        'actor': actors,
        'dataset': 'RAVDESS'
    })
    print(f"RAVDESS: Found {len(df)} files.")
    return df

# 2. CREMA-D Parser 
def parse_cremad_metadata(cremad_base_path):
    """
    Parses the CREMA-D dataset structure and extracts metadata.
    """

    file_paths = []
    raw_emotions = []
    genders = []
    actors = [] 

    # CREMA-D emotion map from filename part
    emotion_map = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
        'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
    }
    
    if not os.path.exists(cremad_base_path):
        print(f"Warning: CREMA-D audio directory not found at {cremad_base_path}. Skipping CREMA-D parsing.")
        return pd.DataFrame()

    print(f"Parsing CREMA-D from: {cremad_base_path}")
    for filename in os.listdir(cremad_base_path):
        if filename.endswith('.wav'):
            parts = filename.split('_') # e.g., ['1001', 'DFA', 'ANG', 'XX.wav']
            
            if len(parts) >= 3:
                actor_id = parts[0] # e.g., '1001'
                emotion_code = parts[2] # e.g., 'ANG'
                
                emotion = emotion_map.get(emotion_code, 'unknown')
                if emotion == 'unknown':
                    print(f"Warning: Unknown CREMA-D emotion code {emotion_code} in file {filename}. Skipping.")
                    continue

                gender = 'unknown' # Placeholder for CREMA-D gender
                try:
                    # For quick inference: CREMA-D actors are 1001 to 1091. Male: 1001-1049, Female: 1050-1091
                    actor_int = int(actor_id)
                    if 1000 <= actor_int <= 1049: gender = 'male'
                    elif 1050 <= actor_int <= 1091: gender = 'female'
                except ValueError:
                    pass # Keep as 'unknown' if not integer

                file_paths.append(os.path.join(cremad_base_path, filename))
                raw_emotions.append(emotion)
                genders.append(gender)
                actors.append(actor_id) 

    df = pd.DataFrame({
        'filepath': file_paths,
        'raw_emotion': raw_emotions,
        'gender': genders,
        'actor': actors,
        'dataset': 'CREMA-D'
    })
    print(f"CREMA-D: Found {len(df)} files.")
    return df

# 3. Placeholder for TESS Parser 
def parse_tess_metadata(tess_base_path):
    """
    Parses the TESS dataset structure and extracts metadata.
    """
    file_paths = []
    raw_emotions = []
    genders = []
    actors = [] # Speaker ID

    # TESS emotion map from filename part
    emotion_map = {
        'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
        'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
        'ps': 'surprised' # 'ps' is pleasant surprise
    }
    
    if not os.path.exists(tess_base_path):
        print(f"Warning: TESS base path not found at {tess_base_path}. Skipping TESS parsing.")
        return pd.DataFrame()

    print(f"Parsing TESS from: {tess_base_path}")
    for speaker_emotion_folder in os.listdir(tess_base_path):
        folder_path = os.path.join(tess_base_path, speaker_emotion_folder)
        if os.path.isdir(folder_path):
            parts = speaker_emotion_folder.split('_') # e.g., ['OAF', 'angry'] or ['YAF', 'happy']
            if len(parts) >= 2:
                speaker_id_prefix = parts[0] # e.g., 'YAF' or 'OAF'
                folder_emotion_code = parts[1] # e.g., 'angry'
                
                gender = 'female' if 'AF' in speaker_id_prefix else 'male' # YAF/OAF for female, other prefixes for male.
                actor = speaker_id_prefix 

                for filename in os.listdir(folder_path):
                    if filename.endswith('.wav'):
                        file_emotion_code = filename.split('_')[1] # e.g., 'happy' from YAF_happy_sad.wav

                        emotion = emotion_map.get(folder_emotion_code, 'unknown')
                        if emotion == 'unknown':
                            print(f"Warning: Unknown TESS emotion code {folder_emotion_code} in folder {speaker_emotion_folder}. Skipping.")
                            continue

                        file_paths.append(os.path.join(folder_path, filename))
                        raw_emotions.append(emotion)
                        genders.append(gender)
                        actors.append(actor)

    df = pd.DataFrame({
        'filepath': file_paths,
        'raw_emotion': raw_emotions,
        'gender': genders,
        'actor': actors,
        'dataset': 'TESS'
    })
    print(f"TESS: Found {len(df)} files.")
    return df

# --- 4. Function to combine and standardize all datasets ---
def create_combined_metadata_df(
    ravdess_path, cremad_path, tess_path,
    master_emotion_map=None, save_path='metadata/extracted_features.csv'
):
    """
    Parses specified datasets, standardizes emotion labels, and combines them into a single DataFrame.
    """

    if master_emotion_map is None:
        master_emotion_map = {
            'neutral': 'neutral',
            'calm': 'neutral', 
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgust': 'disgust',
            'surprised': 'surprised',
            'ps': 'surprised' # TESS's 'pleasant surprise' to 'surprised'
        }
    
    all_dfs = []

    # Parse RAVDESS
    ravdess_df = parse_ravdess_metadata(ravdess_path)
    if not ravdess_df.empty:
        all_dfs.append(ravdess_df)

    # Parse CREMA-D
    cremad_df = parse_cremad_metadata(cremad_path)
    if not cremad_df.empty:
        all_dfs.append(cremad_df)

    # Parse TESS
    tess_df = parse_tess_metadata(tess_path)
    if not tess_df.empty:
        all_dfs.append(tess_df)

    if not all_dfs:
        print("No datasets were successfully parsed. Returning empty DataFrame.")
        return pd.DataFrame()

    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Apply master emotion mapping
    # Filter out emotions not in our master_emotion_map
    original_len = len(combined_df)
    combined_df['emotion'] = combined_df['raw_emotion'].map(master_emotion_map)
    combined_df.dropna(subset=['emotion'], inplace=True) # Remove rows where mapping resulted in NaN
    if len(combined_df) < original_len:
        print(f"Removed {original_len - len(combined_df)} entries due to unmapped/unknown emotions.")

    print(f"\nCombined DataFrame created with {len(combined_df)} entries.")
    print("Emotion distribution in combined dataset:")
    print(combined_df['emotion'].value_counts())
    print("\nDataset origin distribution:")
    print(combined_df['dataset'].value_counts())
    
    # Save the combined DataFrame
    combined_df.to_csv(save_path, index=False)
    print(f"\nCombined metadata saved to '{save_path}'")
    
    return combined_df

def main():
    RAVDESS_BASE_PATH = 'datasets/RAVDESS' 
    CREMAD_BASE_PATH = 'datasets/CREMA-D'   
    TESS_BASE_PATH = 'datasets/TESS'       

    # Example master emotion map 
    my_master_emotion_map = {
        'neutral': 'neutral',
        'calm': 'neutral',       # RAVDESS calm -> neutral
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'fearful': 'fearful',
        'disgust': 'disgust',
        'surprised': 'surprised',
        'ps': 'surprised'
    }

    combined_df = create_combined_metadata_df(
        RAVDESS_BASE_PATH, 
        CREMAD_BASE_PATH, 
        TESS_BASE_PATH,
        master_emotion_map=my_master_emotion_map
    )

if __name__ == '__main__':
    main()