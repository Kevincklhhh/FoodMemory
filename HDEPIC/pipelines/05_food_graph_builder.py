#!/usr/bin/env python3
"""
Food Graph Builder - Build Spatio-Temporal Food Graph

This script processes food blocks to build a spatio-temporal graph that tracks
food items through cooking processes. It uses VLM to:
1. Initialize food state from inventory arrivals (using trigger_text)
2. Infer transactions (TRANSFER, SPLIT, MERGE, CONSUME, UPDATE) from narrations

Pipeline Modes:
- Standard: Narrations + Video → Transactions (single-stage VLM)
- Two-Stage: State Descriptions → Transactions (uses pre-computed descriptions from 04_state_description.py)

Pipeline Steps:
1. Load existing graph state (if resuming)
2. Load inventory arrivals and food blocks
3. Process blocks chronologically
4. For each block: match arrivals, init food state, infer transactions
5. Build BlockGraph snapshots and LineageEdges
6. Save spatio_temporal_graph.json

Usage:
    # Standard pipeline (video + narrations)
    python 05_food_graph_builder.py --video-ids P01-20240203-121517 --model qwen

    # Two-stage pipeline (uses pre-computed state descriptions)
    python 05_food_graph_builder.py --video-ids P01-20240203-121517 --use-descriptions

    # Local mode with single video
    python 05_food_graph_builder.py --video-ids P01-20240202-161948 --local --use-descriptions

    # Local mode with video range
    python 05_food_graph_builder.py --local --start-video P01-20240202-110250 --end-video P01-20240202-161948 --use-descriptions
"""

import json
import sys
import argparse
import requests
import base64
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-api'))
try:
    from openai_api import OpenAIAPI
except ImportError:
    print("WARNING: Could not import OpenAIAPI. GPT-4o will not be available.")
    OpenAIAPI = None

# Import food_graph module
from food_graph import (
    SpatioTemporalGraph,
    BlockGraph,
    FoodNode,
    ContainerNode,
    FoodState,
    ContainmentEdge,
    LineageEdge,
    NodeStatus,
    LocationRegistry,  # V2: unified location model
    process_block,
    generate_instance_id,
    reset_instance_counters,
)
from food_graph.vlm_prompts import (
    FOOD_INIT_SYSTEM_PROMPT,
    TRANSACTION_SYSTEM_PROMPT,
    TRANSACTION_FROM_DESCRIPTIONS_PROMPT,
    build_food_init_prompt,
    build_transaction_prompt,
    build_transaction_prompt_from_descriptions,
    parse_food_init_response,
    parse_vlm_response,
    create_log_entry,
    load_state_descriptions,
)
from food_graph.graph_operations import apply_transactions_batch, snapshot_active_state

# Global logger
logger = logging.getLogger(__name__)

# Default paths (relative to this script's location in pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class VLMClient:
    """Handles communication with VLM APIs (Qwen and GPT-4o)"""

    def __init__(self, model_name: str = 'qwen', use_video: bool = True):
        self.model_name = model_name
        self.use_video = use_video
        self.openai_api = None

        if model_name == 'gpt-4o':
            if OpenAIAPI is None:
                raise ImportError("OpenAIAPI not found. Cannot use gpt-4o.")
            print(f"[VLMClient] Initializing GPT-4o API...")
            self.openai_api = OpenAIAPI(deployment='gpt-4o')
            # GPT-4o in this pipeline is text-only for now as per requirements
            self.use_video = False 

    def encode_video_base64(self, video_path: Path) -> str:
        """Encode video file to base64"""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """Dispatch query to appropriate model"""
        if self.model_name == 'gpt-4o':
            return self._query_openai(system_prompt, user_prompt, max_tokens)
        else:
            return self._query_qwen(system_prompt, user_prompt, video_path, max_tokens, temperature)

    def _query_openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Query GPT-4o (Text Only)"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            completion = self.openai_api.chat_completion(messages, max_tokens=max_tokens)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"  ✗ OpenAI API Error: {e}")
            return ""

    def _query_qwen(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Query Qwen3-VL"""
        messages = [{"role": "system", "content": system_prompt}]
        
        user_content = []
        if self.use_video and video_path and video_path.exists():
            video_base64 = self.encode_video_base64(video_path)
            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}
            })
        
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if self.use_video and video_path and video_path.exists():
            data["extra_body"] = {
                "mm_processor_kwargs": {
                    "fps": 1,
                    "do_sample_frames": True
                }
            }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ✗ Qwen API Error: {e}")
            return ""


class GraphManager:
    """Manages the Spatio-Temporal Graph state and updates"""

    def __init__(self, stg: SpatioTemporalGraph, event_log_file: Path, location_registry: LocationRegistry):
        self.stg = stg
        self.event_log_file = event_log_file
        self.location_registry = location_registry  # V2: unified location model
        self.current_graph = None

    def initialize_block(self, video_id: str, block_id: int, start_time: float, end_time: float, prev_graph: Optional[BlockGraph]):
        """Initialize a new block graph"""
        if prev_graph:
            self.current_graph = prev_graph
        else:
            self.current_graph = BlockGraph(
                video_id=video_id,
                block_id=block_id,
                block_start_time=start_time,
                block_end_time=end_time,
                food_nodes={},
                container_nodes={},
                containment_edges=[]
            )
        
        # Update metadata
        self.current_graph.video_id = video_id
        self.current_graph.block_id = block_id
        self.current_graph.block_start_time = start_time
        self.current_graph.block_end_time = end_time

    def add_arrivals(self, arrivals: List[Dict], init_states: List[Dict], block_idx: int) -> List[str]:
        """Add new food arrivals to the graph"""
        new_ids = []
        for arrival, init_state in zip(arrivals, init_states):
            semantic_name = arrival.get('semantic_name', '')
            if not semantic_name:
                continue

            instance_id = generate_instance_id(semantic_name)

            # V2 model: resolve initial location via registry
            raw_location = init_state.get('state', {}).get('location') or arrival.get('location')
            location = self.location_registry.get_or_create_location(raw_location) if raw_location else None

            node = FoodNode(
                instance_id=instance_id,
                food_noun=semantic_name,
                status=NodeStatus.ACTIVE,
                state=FoodState(
                    form_state=init_state.get('state', {}).get('form_state', 'whole'),
                    quantity=init_state.get('state', {}).get('quantity', 'full'),
                    count=init_state.get('state', {}).get('count')
                ),
                location=location,  # V2: unified location
                parent_instance=None,
                created_at=arrival.get('timestamp', self.current_graph.block_start_time),
                created_in_video=self.current_graph.video_id,
                created_in_block=block_idx  # Use global block_idx
            )
            self.current_graph.food_nodes[instance_id] = node
            new_ids.append(instance_id)

            self._log_event("food_created", {
                "food_id": instance_id,
                "food_noun": semantic_name,
                "state": init_state.get('state', {})
            })
        return new_ids

    def process_new_containers(self, containers: List[Dict]):
        """Add new containers to the graph"""
        for container in containers:
            container_id = container.get('container_id')
            zone = container.get('zone', 'unknown')
            if container_id and container_id not in self.current_graph.container_nodes:
                self.current_graph.container_nodes[container_id] = ContainerNode(
                    container_id=container_id,
                    zone=zone,
                    created_at=self.current_graph.block_start_time,
                    created_in_video=self.current_graph.video_id,
                    created_in_block=self.current_graph.block_id
                )
                self._log_event("container_created", {
                    "container_id": container_id,
                    "zone": zone,
                    "source": "vlm_transaction"
                })

    def apply_transactions(self, transactions: List[Dict], block: Dict) -> Tuple[List[LineageEdge], List[str]]:
        """Apply transactions and return lineage edges + warnings"""
        # V2 model: pass location_registry for auto-numbering containers
        lineage_edges, warnings = apply_transactions_batch(
            self.current_graph, transactions, block, self.location_registry
        )

        for txn in transactions:
            self._log_event("transaction", txn)

        return lineage_edges, warnings

    def snapshot(self) -> BlockGraph:
        """Take a snapshot of the current active state"""
        return snapshot_active_state(self.current_graph, self.current_graph.block_id)

    def _log_event(self, event_type: str, data: Dict):
        """Log event to file"""
        event = {
            "video_id": self.current_graph.video_id,
            "block_id": self.current_graph.block_id,
            "type": event_type,
            **data
        }
        with open(self.event_log_file, 'a') as f:
            f.write(json.dumps(event) + "\n")


class PipelineRunner:
    """Orchestrates the food graph building pipeline"""

    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_logging()
        self.vlm_client = VLMClient(model_name=args.model, use_video=not args.no_video)
        
        # Initialize Graph
        if args.resume and self.graph_file.exists():
            print(f"\n[Setup] Loading existing graph from {self.graph_file}")
            self.stg = SpatioTemporalGraph.load(str(self.graph_file))
        else:
            print("\n[Setup] Creating new graph")
            self.stg = SpatioTemporalGraph(metadata={"videos_processed": [], "model": args.model})
            if not args.resume:
                reset_instance_counters()

        # V2 model: Create location registry for auto-numbering containers
        self.location_registry = LocationRegistry()

        self.graph_manager = GraphManager(self.stg, self.event_log_file, self.location_registry)

    def setup_paths(self):
        self.arrivals_dir = Path(self.args.arrivals_dir)
        self.blocks_dir = Path(self.args.blocks_dir)
        self.output_dir = Path(self.args.output_dir)
        self.clips_dir = Path(self.args.clips_dir) if not self.args.no_video else None
        self.descriptions_dir = Path(self.args.descriptions_dir) if self.args.use_descriptions else None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_log_dir = self.output_dir / 'vlm_logs'
        self.vlm_log_dir.mkdir(parents=True, exist_ok=True)

        self.graph_file = self.output_dir / 'spatio_temporal_graph.json'
        self.event_log_file = self.output_dir / 'event_log.jsonl'

    def setup_logging(self):
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        log_file = self.output_dir / 'graph_builder.log'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        
        graph_logger = logging.getLogger('food_graph')
        graph_logger.setLevel(logging.WARNING)
        graph_logger.addHandler(file_handler)

    def load_data(self, video_id: str) -> Tuple[List[Dict], List[Dict], Dict[int, Path]]:
        """Load arrivals, blocks, and clips for a video"""
        # Arrivals
        arrivals_file = self.arrivals_dir / f"{video_id}_arrivals.json"
        arrivals = []
        if arrivals_file.exists():
            with open(arrivals_file, 'r') as f:
                arrivals = json.load(f).get('arrivals', [])

        # Blocks
        blocks_file = self.blocks_dir / f"{video_id}_food_blocks.json"
        food_blocks = []
        if blocks_file.exists():
            with open(blocks_file, 'r') as f:
                data = json.load(f)
                food_blocks = [b for b in data.get('blocks', []) if b.get('has_food_action', False)]
                food_blocks.sort(key=lambda x: x['block_id'])

        # Clips
        clips = {}
        if self.clips_dir:
            manifest_file = self.clips_dir / video_id / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                    for bid, path in manifest.get('clips', {}).items():
                        clips[int(bid)] = Path(path)

        return arrivals, food_blocks, clips

    def match_arrivals(self, arrivals: List[Dict], block: Dict) -> List[Dict]:
        """Match arrivals to block time range"""
        start, end = block['block_start_time'], block['block_end_time']
        return [a for a in arrivals if start <= a.get('timestamp', 0) < end]

    def run(self):
        # Determine videos
        if self.args.video_ids:
            video_ids = self.args.video_ids
        elif self.args.all_videos:
            arrival_files = sorted(self.arrivals_dir.glob("P01-*_arrivals.json"))
            video_ids = [f.stem.replace("_arrivals", "") for f in arrival_files]
        else:
            print("ERROR: Specify --video-ids or --all-videos")
            return

        # Filter processed
        if self.args.resume:
            processed = self.stg.get_videos_processed()
            video_ids = [v for v in video_ids if v not in processed]

        pipeline_mode = "two-stage (state descriptions → transactions)" if self.args.use_descriptions else "standard (narrations + video)"
        print(f"\n[Pipeline] Processing {len(video_ids)} videos with {self.args.model}")
        print(f"[Pipeline] Mode: {pipeline_mode}")
        if self.args.use_descriptions:
            print(f"[Pipeline] Descriptions dir: {self.descriptions_dir}")
        
        prev_graph = self.stg.get_last_block_graph()
        if not self.args.resume:
            open(self.event_log_file, 'w').close()

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n{'='*70}")
            print(f"[Video {i}/{len(video_ids)}] {video_id}")
            print("=" * 70)

            arrivals, food_blocks, clips = self.load_data(video_id)
            print(f"  Arrivals: {len(arrivals)}")
            print(f"  Food blocks: {len(food_blocks)}")
            
            if not food_blocks:
                print("  Skipping - no food blocks")
                continue

            prev_graph = self.process_video(video_id, arrivals, food_blocks, clips, prev_graph)
            
            # Update metadata and save
            if video_id not in self.stg.metadata.get("videos_processed", []):
                self.stg.metadata.setdefault("videos_processed", []).append(video_id)
            self.stg.save(str(self.graph_file))
            print(f"  Checkpoint saved ({len(self.stg.block_graphs)} total blocks)")

    def process_video(self, video_id: str, arrivals: List[Dict], food_blocks: List[Dict], clips: Dict[int, Path], prev_graph: Optional[BlockGraph]) -> BlockGraph:
        used_arrivals = set()
        
        # Initialize graph manager with previous state or new
        if prev_graph is None:
             # Create dummy start graph
             self.graph_manager.initialize_block(video_id, 0, 0.0, 0.0, None)
        else:
             self.graph_manager.current_graph = prev_graph

        for block in food_blocks:
            block_id = block['block_id']
            block['video_id'] = video_id
            # Add global block_idx (position in block_graphs array)
            block['block_idx'] = len(self.stg.block_graphs)

            if self.args.verbose:
                print(f"\n  Block {block_id}: {block['block_start_time']:.1f}s - {block['block_end_time']:.1f}s")

            # 1. Initialize Block
            self.graph_manager.initialize_block(
                video_id, block_id, 
                block['block_start_time'], block['block_end_time'], 
                self.graph_manager.current_graph
            )

            # Get video clip for this block (used for both init and transactions)
            video_path = clips.get(block_id)

            # 2. Handle Arrivals
            new_ids = []
            matched = [a for a in self.match_arrivals(arrivals, block) if a.get('semantic_name') not in used_arrivals]
            for a in matched: used_arrivals.add(a.get('semantic_name'))

            if matched:
                if self.args.verbose: print(f"    Found {len(matched)} new arrivals")
                # Pass block_start_time for clip-relative timestamps
                user_prompt = build_food_init_prompt(matched, block['block_start_time'])
                # Send video clip for visual count estimation
                response = self.vlm_client.query(
                    FOOD_INIT_SYSTEM_PROMPT,
                    user_prompt,
                    video_path=video_path
                )
                init_states = parse_food_init_response(response)

                # Log VLM call
                self.save_vlm_log(video_id, block_id, "init", FOOD_INIT_SYSTEM_PROMPT, user_prompt, response, init_states)

                # Add to graph (pass block_idx for proper lineage tracking)
                new_ids = self.graph_manager.add_arrivals(matched, init_states, block['block_idx'])

            # 3. Infer Transactions

            # Check if using pre-computed state descriptions (two-stage pipeline)
            descriptions = None
            if self.args.use_descriptions and self.descriptions_dir:
                descriptions = load_state_descriptions(video_id, block_id, str(self.descriptions_dir))

            if descriptions:
                # Two-stage pipeline: use state descriptions + video for verification
                user_prompt = build_transaction_prompt_from_descriptions(
                    self.graph_manager.current_graph, block, descriptions, newly_created_food_ids=new_ids
                )
                system_prompt = TRANSACTION_FROM_DESCRIPTIONS_PROMPT
                response = self.vlm_client.query(
                    system_prompt,
                    user_prompt,
                    video_path=video_path,  # Video for grounding transactions
                    max_tokens=2000
                )
                if self.args.verbose:
                    print(f"    Using {len(descriptions)} pre-computed state descriptions")
            else:
                # Original pipeline: narrations + video
                user_prompt = build_transaction_prompt(self.graph_manager.current_graph, block, newly_created_food_ids=new_ids)

                # Add video instruction if available
                if self.vlm_client.use_video and video_path:
                    user_prompt = f"**VIDEO CLIP ATTACHED**: Review the video to confirm what happens in this block.\n\n{user_prompt}"

                system_prompt = TRANSACTION_SYSTEM_PROMPT
                response = self.vlm_client.query(
                    system_prompt,
                    user_prompt,
                    video_path=video_path,
                    max_tokens=2000
                )

            result = parse_vlm_response(response)

            if self.args.verbose:
                txn_count = len(result.get('transactions', []))
                print(f"      - {txn_count} transactions")

            # 4. Update Graph
            # V2 model: new_containers no longer used (locations auto-numbered by registry)
            # Keep for backward compatibility with V1 VLM responses
            if result.get('new_containers'):
                self.graph_manager.process_new_containers(result.get('new_containers', []))
            lineage, warnings = self.graph_manager.apply_transactions(result.get('transactions', []), block)

            # Save VLM log with execution warnings
            log_type = "transactions_from_descriptions" if descriptions else "transactions"
            self.save_vlm_log(video_id, block_id, log_type, system_prompt, user_prompt, response, result, warnings)

            if warnings:
                print(f"    WARNINGS: {warnings}")

            # 5. Snapshot
            snapshot = self.graph_manager.snapshot()
            self.stg.block_graphs.append(snapshot)
            self.stg.lineage_edges.extend(lineage)

            if not self.args.verbose and (block_id + 1) % 1 == 0:
                print(f"    Progress: {block_id + 1}/{len(food_blocks)} blocks")

        return self.graph_manager.current_graph

    def save_vlm_log(self, video_id, block_id, log_type, sys_p, user_p, resp, parsed, warnings=None):
        video_log_dir = self.vlm_log_dir / video_id
        video_log_dir.mkdir(parents=True, exist_ok=True)
        entry = create_log_entry(video_id, block_id, log_type, sys_p, user_p, resp, parsed)

        # Add warnings to entry if present
        if warnings:
            entry["execution_warnings"] = warnings

        # Save JSON
        with open(video_log_dir / f"block_{block_id:03d}_{log_type}.json", 'w') as f:
            json.dump(entry, f, indent=2)

        # Save Formatted Text
        with open(video_log_dir / f"block_{block_id:03d}_{log_type}.txt", 'w') as f:
            f.write(f"=== SYSTEM PROMPT ===\n{sys_p}\n\n")
            f.write(f"=== USER PROMPT ===\n{user_p}\n\n")
            f.write(f"=== RAW RESPONSE ===\n{resp}\n\n")
            f.write(f"=== PARSED RESPONSE ===\n{json.dumps(parsed, indent=2)}\n")
            if warnings:
                f.write(f"\n=== EXECUTION WARNINGS ===\n")
                for w in warnings:
                    f.write(f"  - {w}\n")


def get_video_range(blocks_dir: Path, start_video: str, end_video: str) -> List[str]:
    """Get list of video IDs within the specified range (inclusive)."""
    all_files = sorted(blocks_dir.glob("P01-*_food_blocks.json"))
    video_ids = [f.stem.replace("_food_blocks", "") for f in all_files]

    start_idx = 0
    end_idx = len(video_ids)

    if start_video and start_video in video_ids:
        start_idx = video_ids.index(start_video)
    if end_video and end_video in video_ids:
        end_idx = video_ids.index(end_video) + 1

    return video_ids[start_idx:end_idx]


def main():
    parser = argparse.ArgumentParser(description="Build Spatio-Temporal Food Graph")
    parser.add_argument('--video-ids', nargs='+', help='Specific video IDs')
    parser.add_argument('--all-videos', action='store_true', help='Process all videos')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--model', default='qwen', choices=['qwen', 'gpt-4o'], help='VLM model to use')
    parser.add_argument('--no-video', action='store_true', help='Disable video (force text-only)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Local mode
    parser.add_argument('--local', action='store_true', help='Use local inventory discovery')
    parser.add_argument('--start-video', help='Start video ID for range')
    parser.add_argument('--end-video', help='End video ID for range')

    # Two-stage pipeline (use pre-computed state descriptions)
    parser.add_argument('--use-descriptions', action='store_true',
                        help='Use pre-computed state descriptions instead of raw narrations')
    parser.add_argument('--descriptions-dir', default=str(_PROJECT_ROOT / "outputs" / "food_classification" / "state_descriptions"),
                        help='Directory containing state descriptions')

    # Paths
    parser.add_argument('--arrivals-dir', default=str(_PROJECT_ROOT / "outputs" / "inventory_discovery_global"))
    parser.add_argument('--blocks-dir', default=str(_PROJECT_ROOT / "outputs" / "food_classification"))
    parser.add_argument('--output-dir', default=str(_PROJECT_ROOT / "outputs" / "food_graph"))
    parser.add_argument('--clips-dir', default=str(_PROJECT_ROOT / "outputs" / "food_clips"))

    args = parser.parse_args()

    # Handle local mode: override paths based on video range
    if args.local:
        # Allow single video with --video-ids, or range with --start-video/--end-video
        if args.video_ids:
            # Single video or explicit list
            if len(args.video_ids) == 1:
                start_video = end_video = args.video_ids[0]
            else:
                start_video = args.video_ids[0]
                end_video = args.video_ids[-1]
        elif args.start_video and args.end_video:
            start_video = args.start_video
            end_video = args.end_video
        else:
            print("ERROR: --local requires either --video-ids or both --start-video and --end-video")
            return

        range_name = f"{start_video}_to_{end_video}"
        args.arrivals_dir = str(_PROJECT_ROOT / "outputs" / "inventory_discovery_local" / range_name)
        args.output_dir = str(_PROJECT_ROOT / "outputs" / "food_graph_local" / range_name)

        # Set video_ids from range if not already set
        if not args.video_ids:
            args.video_ids = get_video_range(Path(args.blocks_dir), start_video, end_video)
            if not args.video_ids:
                print(f"ERROR: No videos found in range {start_video} to {end_video}")
                return

        print(f"[Local Mode] Range: {start_video} -> {end_video}")
        print(f"[Local Mode] Videos: {len(args.video_ids)}")
        print(f"[Local Mode] Arrivals: {args.arrivals_dir}")
        print(f"[Local Mode] Output: {args.output_dir}")

    runner = PipelineRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
