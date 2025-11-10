#!/usr/bin/env python3
"""
Analyze food abundance across videos, participants, and settings.

This addresses the concern that multiple videos from the same participant
might contain the same physical food instance (e.g., same onion in P02_01, P02_02).

Video naming convention:
- P{participant_id}_{video_num}
- video_num >= 100: "new collection" (different setting)
- video_num < 100: "original collection" (likely same setting/kitchen)
"""

import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def parse_video_id(video_id: str) -> Tuple[str, str, int]:
    """Parse video ID into participant, session type, and video number.

    Args:
        video_id: Video ID like 'P02_03' or 'P02_112'

    Returns:
        Tuple of (participant_id, session_type, video_num)
        session_type is 'original' (< 100) or 'new_collection' (>= 100)
    """
    parts = video_id.split('_')
    participant_id = parts[0]
    video_num = int(parts[1])

    session_type = 'new_collection' if video_num >= 100 else 'original'

    return participant_id, session_type, video_num


def analyze_food_abundance(food_per_video_path: str = 'food_per_video.json') -> Dict:
    """Analyze food abundance across videos, participants, and settings.

    Returns:
        Dictionary with food class statistics
    """
    # Load food per video data
    print(f"Loading food per video data from {food_per_video_path}...")
    with open(food_per_video_path, 'r') as f:
        videos = json.load(f)
    print(f"✓ Loaded {len(videos)} videos")

    # Build food abundance statistics
    food_stats = defaultdict(lambda: {
        'videos': set(),
        'participants': set(),
        'settings': set(),  # (participant, session_type)
        'participant_video_count': defaultdict(int),
        'participant_session_types': defaultdict(set),
    })

    for video_id, video_data in videos.items():
        participant_id, session_type, video_num = parse_video_id(video_id)

        # Get all food classes in this video
        for food_item in video_data['food_items']:
            food_class = food_item['noun_key']

            # Add to statistics
            food_stats[food_class]['videos'].add(video_id)
            food_stats[food_class]['participants'].add(participant_id)
            food_stats[food_class]['settings'].add((participant_id, session_type))
            food_stats[food_class]['participant_video_count'][participant_id] += 1
            food_stats[food_class]['participant_session_types'][participant_id].add(session_type)

    # Convert to serializable format and calculate metrics
    result = {}

    for food_class, stats in sorted(food_stats.items()):
        # Calculate contamination risk
        videos_count = len(stats['videos'])
        participants_count = len(stats['participants'])
        settings_count = len(stats['settings'])

        # Risk: if videos_count >> settings_count, high risk of same instance
        contamination_risk = videos_count / settings_count if settings_count > 0 else 0

        # Find participants with multiple videos
        multi_video_participants = {
            p: count for p, count in stats['participant_video_count'].items()
            if count > 1
        }

        # Calculate diversity metrics
        participant_distribution = dict(stats['participant_video_count'])
        max_videos_per_participant = max(stats['participant_video_count'].values()) if stats['participant_video_count'] else 0

        # Assess risk level
        if contamination_risk > 2.0:
            risk_level = 'HIGH'
        elif contamination_risk > 1.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        result[food_class] = {
            'total_videos': videos_count,
            'unique_participants': participants_count,
            'unique_settings': settings_count,
            'contamination_risk_ratio': round(contamination_risk, 2),
            'risk_level': risk_level,
            'max_videos_per_participant': max_videos_per_participant,
            'participants_with_multiple_videos': len(multi_video_participants),
            'participant_distribution': participant_distribution,
            'multi_video_participants': multi_video_participants,
            'videos': sorted(list(stats['videos'])),
            'participants': sorted(list(stats['participants'])),
        }

    return result


def print_abundance_statistics(food_abundance: Dict):
    """Print food abundance statistics."""
    print("\n" + "=" * 100)
    print("FOOD ABUNDANCE ANALYSIS: Videos, Participants, and Settings")
    print("=" * 100)
    print("\nGoal: Assess risk of same physical food instance appearing in multiple videos")
    print("- Videos: Total number of videos containing this food")
    print("- Participants: Number of unique participants")
    print("- Settings: Unique (participant, session_type) combinations")
    print("- Contamination Risk: videos/settings ratio (>1.5 = concern)")
    print("\n" + "=" * 100)

    # Sort by contamination risk
    sorted_foods = sorted(
        food_abundance.items(),
        key=lambda x: x[1]['contamination_risk_ratio'],
        reverse=True
    )

    print(f"\n{'Food Class':<20} {'Videos':<8} {'Participants':<14} {'Settings':<10} {'Risk':<6} {'Max/P':<6} {'Multi-P':<8} {'Level':<8}")
    print("-" * 100)

    for food_class, stats in sorted_foods:
        print(
            f"{food_class:<20} "
            f"{stats['total_videos']:<8} "
            f"{stats['unique_participants']:<14} "
            f"{stats['unique_settings']:<10} "
            f"{stats['contamination_risk_ratio']:<6.2f} "
            f"{stats['max_videos_per_participant']:<6} "
            f"{stats['participants_with_multiple_videos']:<8} "
            f"{stats['risk_level']:<8}"
        )

    # High risk foods
    high_risk = [f for f, s in sorted_foods if s['risk_level'] in ['HIGH', 'MEDIUM']]
    print(f"\n\nHIGH/MEDIUM CONTAMINATION RISK FOODS: {len(high_risk)}")
    print("=" * 100)

    for food_class in high_risk[:20]:  # Top 20
        stats = food_abundance[food_class]
        print(f"\n{food_class.upper()}: {stats['contamination_risk_ratio']:.2f}x risk")
        print(f"  Videos: {stats['total_videos']}, Participants: {stats['unique_participants']}, Settings: {stats['unique_settings']}")
        print(f"  Participants with multiple videos: {stats['participants_with_multiple_videos']}")

        # Show top participants with most videos
        top_participants = sorted(
            stats['multi_video_participants'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        if top_participants:
            print(f"  Top contributors:", end="")
            for p, count in top_participants:
                print(f" {p}({count})", end="")
            print()


def print_participant_session_breakdown(food_abundance: Dict):
    """Print breakdown of original vs new collection sessions."""
    print("\n" + "=" * 100)
    print("PARTICIPANT SESSION TYPE BREAKDOWN")
    print("=" * 100)

    # Analyze original vs new_collection distribution
    all_participants = set()
    for food_class, stats in food_abundance.items():
        all_participants.update(stats['participants'])

    print(f"\nTotal participants: {len(all_participants)}")

    # Load food_per_video to get session breakdown
    with open('food_per_video.json', 'r') as f:
        videos = json.load(f)

    participant_sessions = defaultdict(lambda: {'original': set(), 'new_collection': set()})

    for video_id in videos.keys():
        participant_id, session_type, video_num = parse_video_id(video_id)
        participant_sessions[participant_id][session_type].add(video_id)

    print(f"\n{'Participant':<15} {'Original':<12} {'New Coll.':<12} {'Total':<8}")
    print("-" * 50)

    for participant_id in sorted(participant_sessions.keys()):
        sessions = participant_sessions[participant_id]
        original_count = len(sessions['original'])
        new_coll_count = len(sessions['new_collection'])
        total = original_count + new_coll_count

        print(f"{participant_id:<15} {original_count:<12} {new_coll_count:<12} {total:<8}")


def generate_distractor_recommendations(food_abundance: Dict):
    """Generate recommendations for distractor selection."""
    print("\n" + "=" * 100)
    print("DISTRACTOR SELECTION RECOMMENDATIONS")
    print("=" * 100)

    print("\nBased on contamination risk analysis:\n")

    low_risk = [f for f, s in food_abundance.items() if s['risk_level'] == 'LOW']
    medium_risk = [f for f, s in food_abundance.items() if s['risk_level'] == 'MEDIUM']
    high_risk = [f for f, s in food_abundance.items() if s['risk_level'] == 'HIGH']

    print(f"LOW RISK ({len(low_risk)} foods):")
    print("  ✓ Safe to use any video as distractor")
    print("  ✓ Low likelihood of same physical instance")
    print()

    print(f"MEDIUM RISK ({len(medium_risk)} foods):")
    print("  ⚠ Prefer different participants for distractors")
    print("  ⚠ If same participant, use different session types")
    print(f"  Foods: {', '.join(medium_risk[:10])}")
    if len(medium_risk) > 10:
        print(f"  ... and {len(medium_risk) - 10} more")
    print()

    print(f"HIGH RISK ({len(high_risk)} foods):")
    print("  ⚠️ MUST use different participants for distractors")
    print("  ⚠️ Multiple videos from same participant likely same instance")
    print(f"  Foods: {', '.join(high_risk[:10])}")
    if len(high_risk) > 10:
        print(f"  ... and {len(high_risk) - 10} more")
    print()

    print("\nRECOMMENDED STRATEGY:")
    print("1. For LOW risk foods: Sample distractors from any video")
    print("2. For MEDIUM risk foods: Prefer different participants; if same, use different session types")
    print("3. For HIGH risk foods: ONLY use different participants")
    print("4. Track unique settings (participant + session_type) rather than just videos")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze food abundance across videos, participants, and settings"
    )
    parser.add_argument(
        '--food-per-video',
        default='food_per_video.json',
        help='Food per video JSON (default: food_per_video.json)'
    )
    parser.add_argument(
        '--output',
        default='food_abundance_analysis.json',
        help='Output JSON file (default: food_abundance_analysis.json)'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("FOOD ABUNDANCE ANALYSIS")
    print("=" * 100)

    # Analyze
    food_abundance = analyze_food_abundance(args.food_per_video)

    # Print statistics
    print_abundance_statistics(food_abundance)
    print_participant_session_breakdown(food_abundance)
    generate_distractor_recommendations(food_abundance)

    # Save results
    print(f"\n\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(food_abundance, f, indent=2)
    print(f"✓ Saved to {args.output}")

    print("\n" + "=" * 100)
    print("✓ Analysis complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
