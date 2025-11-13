#!/usr/bin/env python3
"""
Extract all data related to a specific participant from HD_EPIC_Narrations.pkl
"""

import pickle
import pandas as pd
import argparse


def extract_participant_data(pickle_path, participant_id, output_path=None):
    """
    Extract all narrations for a specific participant.

    Args:
        pickle_path: Path to HD_EPIC_Narrations.pkl
        participant_id: Participant ID to extract (e.g., 'P01' or 1)
        output_path: Optional output path for CSV file

    Returns:
        DataFrame with filtered data
    """
    # Load the pickle file
    print(f"Loading data from {pickle_path}...")
    df = pickle.load(open(pickle_path, 'rb'))

    # Convert participant_id to proper format if needed
    if isinstance(participant_id, int):
        participant_id = f'P{participant_id:02d}'

    # Filter by participant
    print(f"Filtering for participant: {participant_id}")
    participant_df = df[df['participant_id'] == participant_id].copy()

    print(f"\nFound {len(participant_df)} narrations for {participant_id}")
    print(f"Video IDs: {participant_df['video_id'].unique()}")
    print(f"\nDataFrame shape: {participant_df.shape}")
    print(f"\nColumn names:\n{list(participant_df.columns)}")
    print(f"\nFirst few rows:")
    print(participant_df.head())

    # Save to CSV if output path specified
    if output_path:
        participant_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return participant_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract participant data from HD_EPIC_Narrations.pkl')
    parser.add_argument('--participant', '-p', type=int, default=1,
                        help='Participant ID number (default: 1 for P01)')
    parser.add_argument('--input', '-i',
                        default='./hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl',
                        help='Input pickle file path')
    parser.add_argument('--output', '-o',
                        help='Output CSV file path (optional)')

    args = parser.parse_args()

    extract_participant_data(args.input, args.participant, args.output)
