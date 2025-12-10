#!/usr/bin/env python3
"""
Food Graph Builder from Gemini Pre-Annotations

This script builds a spatio-temporal food graph from pre-annotated Gemini state
inference outputs. Unlike the block-based pipeline (05_food_graph_builder.py),
this processes events ONE-AT-A-TIME in chronological order.

Input:
    outputs/{range}/
    ├── inventory.json      # Food arrivals with narration IDs
    └── state_change.json   # Pre-annotated state change events from Gemini

Output:
    outputs/food_graph_gemini/{range}/
    ├── spatio_temporal_graph.json
    ├── event_log.jsonl
    ├── vlm_logs/
    └── clips/

Usage:
    python 06_food_graph_from_gemini.py --input-dir ../outputs/P01-20240202-161354_to_P01-20240202-161948
    python 06_food_graph_from_gemini.py --input-dir ../outputs/P01-20240202-161354_to_P01-20240202-161948 --verbose
"""

import json
import sys
import argparse
import requests
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Default paths (relative to this script's location in gemini_pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add llm-api to path for OpenAI (optional)
sys.path.insert(0, str(_PROJECT_ROOT.parent / 'llm-api'))
try:
    from openai_api import OpenAIAPI
except ImportError:
    OpenAIAPI = None

# Import food_graph module
from food_graph import (
    SpatioTemporalGraph,
    BlockGraph,
    FoodNode,
    FoodState,
    LineageEdge,
    NodeStatus,
    LocationRegistry,
    generate_instance_id,
    reset_instance_counters,
    apply_transactions_batch,
    snapshot_active_state,
    # VLM prompts
    FOOD_INIT_SYSTEM_PROMPT,
    GEMINI_TRANSACTION_SYSTEM_PROMPT,
    build_food_init_prompt,
    build_gemini_transaction_prompt,
    parse_food_init_response,
    parse_vlm_response,
    # Gemini utils
    NarrationLookup,
    VideoClipExtractor,
)

# Global logger
logger = logging.getLogger(__name__)

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
            self.use_video = False  # GPT-4o is text-only for now

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


class GeminiPipelineRunner:
    """Orchestrates the Gemini-based food graph building pipeline."""

    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_logging()

        # Initialize lookup and extractor
        self.narration_lookup = NarrationLookup(self.csv_path)
        self.clip_extractor = VideoClipExtractor(
            video_dir=self.video_dir,
            cache_dir=self.clips_dir,
            fps=2,
            default_buffer=args.clip_buffer
        )

        # Initialize VLM client
        self.vlm_client = VLMClient(model_name=args.model, use_video=not args.no_video)

        # Initialize graph state
        self.location_registry = LocationRegistry()
        self.stg = SpatioTemporalGraph(metadata={
            "source": "gemini_pre_annotations",
            "model": args.model,
            "input_dir": str(args.input_dir)
        })

        # Current graph state (modified in place)
        self.current_graph = BlockGraph(
            video_id="",
            block_id=0,
            block_start_time=0.0,
            block_end_time=0.0,
            food_nodes={},
            container_nodes={},
            containment_edges=[]
        )

        # Event log
        self.event_log_file = self.output_dir / "event_log.jsonl"
        open(self.event_log_file, 'w').close()  # Clear existing

        # Reset instance counters
        reset_instance_counters()

    def setup_paths(self):
        """Setup input/output paths."""
        self.input_dir = Path(self.args.input_dir)
        self.csv_path = Path(self.args.csv_path)
        self.video_dir = Path(self.args.video_dir)

        # Output directory based on input directory name
        input_name = self.input_dir.name
        self.output_dir = Path(self.args.output_dir) / input_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vlm_log_dir = self.output_dir / "vlm_logs"
        self.vlm_log_dir.mkdir(parents=True, exist_ok=True)

        self.clips_dir = self.output_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.graph_file = self.output_dir / "spatio_temporal_graph.json"

    def setup_logging(self):
        """Setup logging configuration."""
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

        log_file = self.output_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO if self.args.verbose else logging.WARNING,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )

    def load_inventory(self) -> List[Dict]:
        """Load inventory.json."""
        inventory_file = self.input_dir / "inventory.json"
        if not inventory_file.exists():
            print(f"WARNING: inventory.json not found at {inventory_file}")
            return []

        with open(inventory_file, 'r') as f:
            data = json.load(f)

        print(f"[Load] Loaded {len(data)} inventory items")
        return data

    def load_state_changes(self) -> List[Dict]:
        """Load state_change.json."""
        state_file = self.input_dir / "state_change.json"
        if not state_file.exists():
            print(f"ERROR: state_change.json not found at {state_file}")
            return []

        with open(state_file, 'r') as f:
            data = json.load(f)

        print(f"[Load] Loaded {len(data)} state change events")
        return data

    def process_inventory(self, inventory: List[Dict]) -> List[str]:
        """
        Process inventory items to initialize food nodes.

        Returns list of created food instance IDs.
        """
        if not inventory:
            return []

        print(f"\n{'='*60}")
        print(f"[Inventory] Processing {len(inventory)} items")
        print(f"{'='*60}")

        created_ids = []

        for i, item in enumerate(inventory, 1):
            narration_id = item.get('narration ID', '')
            food_name = item.get('food_name', '')
            source_action = item.get('source_action', '')

            if not narration_id or not food_name:
                print(f"  {i}. SKIP: Missing narration ID or food name")
                continue

            print(f"\n  {i}. {food_name} ({narration_id})")

            # Lookup timestamp from CSV
            narr = self.narration_lookup.get_narration(narration_id)
            if not narr:
                print(f"     ERROR: Narration not found in CSV")
                continue

            video_id = narr['video_id']
            timestamp = narr['start_timestamp']

            # Extract short video clip around arrival
            clip_name = f"inventory_{i:03d}_{food_name.replace(' ', '_')}"
            clip_path = self.clip_extractor.extract_clip(
                video_id=video_id,
                start_time=max(0, timestamp - 2),
                end_time=timestamp + 3,
                clip_name=clip_name
            )

            if not clip_path:
                print(f"     WARNING: Could not extract clip, using defaults")

            # Query VLM to estimate quantity/count
            arrivals_for_prompt = [{
                'semantic_name': food_name,
                'trigger_text': source_action,
                'timestamp': timestamp
            }]
            user_prompt = build_food_init_prompt(arrivals_for_prompt, block_start_time=timestamp - 2)

            init_states = [{'state': {'form_state': 'whole', 'quantity': 'full', 'count': None}}]

            if not self.args.dry_run and clip_path:
                response = self.vlm_client.query(
                    FOOD_INIT_SYSTEM_PROMPT,
                    user_prompt,
                    video_path=clip_path
                )
                parsed_states = parse_food_init_response(response)
                if parsed_states:
                    init_states = parsed_states

                # Save VLM log
                self.save_vlm_log(
                    video_id=video_id,
                    event_id=f"inventory_{i:03d}",
                    prompt_type="init",
                    system_prompt=FOOD_INIT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response=response,
                    parsed=init_states
                )

            # Create FoodNode
            init_state = init_states[0] if init_states else {}
            instance_id = generate_instance_id(food_name)

            node = FoodNode(
                instance_id=instance_id,
                food_noun=food_name,
                status=NodeStatus.ACTIVE,
                state=FoodState(
                    form_state=init_state.get('state', {}).get('form_state', 'whole'),
                    quantity=init_state.get('state', {}).get('quantity', 'full'),
                    count=init_state.get('state', {}).get('count')
                ),
                location=None,
                parent_instance=None,
                created_at=timestamp,
                created_in_video=video_id,
                created_in_block=0  # Inventory is created at "block 0"
            )

            self.current_graph.food_nodes[instance_id] = node
            created_ids.append(instance_id)

            # Log event
            self._log_event("food_created", {
                "food_id": instance_id,
                "food_noun": food_name,
                "source_narration_id": narration_id,
                "state": init_state.get('state', {})
            })

            print(f"     Created: {instance_id} ({node.state.form_state}, {node.state.quantity})")

        return created_ids

    def process_state_changes(self, state_changes: List[Dict], initial_food_ids: List[str]):
        """
        Process state change events one at a time.

        Args:
            state_changes: List of state change events from Gemini
            initial_food_ids: IDs of food items created from inventory
        """
        if not state_changes:
            print("\n[StateChanges] No state changes to process")
            return

        # Sort by video_id and timestamp
        def sort_key(event):
            narration_ids = event.get('source_narration_ids', [])
            if narration_ids:
                video_id, start_time, _ = self.narration_lookup.get_timestamp_range(narration_ids)
                return (video_id or "", start_time)
            return ("", event.get('timestamp_start', 0))

        sorted_events = sorted(state_changes, key=sort_key)

        print(f"\n{'='*60}")
        print(f"[StateChanges] Processing {len(sorted_events)} events")
        print(f"{'='*60}")

        for i, event in enumerate(sorted_events, 1):
            self.process_single_event(event, i, len(sorted_events))

    def process_single_event(self, event: Dict, event_num: int, total_events: int):
        """Process a single state change event."""
        event_id = event.get('event_id', event_num)
        narration_ids = event.get('source_narration_ids', [])
        state_desc = event.get('state_description', '')
        primary_action = event.get('primary_action', '')

        print(f"\n  Event {event_num}/{total_events}: {primary_action}")
        print(f"    State: {state_desc[:80]}..." if len(state_desc) > 80 else f"    State: {state_desc}")

        # Resolve timestamps from narration IDs
        video_id, start_time, end_time = self.narration_lookup.get_timestamp_range(narration_ids)
        if not video_id:
            print(f"    ERROR: Could not resolve timestamps for narration IDs")
            return

        print(f"    Video: {video_id} [{start_time:.1f}s - {end_time:.1f}s]")

        # Extract video clip for this event
        clip_name = f"event_{event_num:03d}_{video_id}"
        clip_path = self.clip_extractor.extract_clip_for_event(
            narration_lookup=self.narration_lookup,
            narration_ids=narration_ids,
            clip_name=clip_name
        )

        # Build prompt with current graph state
        user_prompt = build_gemini_transaction_prompt(
            graph=self.current_graph,
            event=event,
            newly_created_food_ids=None  # Could track recent creations if needed
        )

        # Query VLM to infer transaction type
        transactions = []
        response = ""

        if not self.args.dry_run:
            response = self.vlm_client.query(
                GEMINI_TRANSACTION_SYSTEM_PROMPT,
                user_prompt,
                video_path=clip_path
            )
            result = parse_vlm_response(response)
            transactions = result.get('transactions', [])

            # Save VLM log
            self.save_vlm_log(
                video_id=video_id,
                event_id=f"event_{event_num:03d}",
                prompt_type="transaction",
                system_prompt=GEMINI_TRANSACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response=response,
                parsed=result
            )

        print(f"    Transactions: {len(transactions)}")

        # Execute transactions
        if transactions:
            event_block = {
                'video_id': video_id,
                'block_id': event_num,
                'block_idx': len(self.stg.block_graphs),
                'block_start_time': start_time,
                'block_end_time': end_time,
            }

            lineage_edges, warnings = apply_transactions_batch(
                self.current_graph,
                transactions,
                event_block,
                self.location_registry
            )

            if warnings:
                print(f"    WARNINGS: {warnings}")

            # Log transactions
            for txn in transactions:
                self._log_event("transaction", {
                    "event_id": event_id,
                    "source_narration_ids": narration_ids,
                    **txn
                })

            # Add lineage edges
            self.stg.lineage_edges.extend(lineage_edges)

        # Create snapshot after event
        self.current_graph.video_id = video_id
        self.current_graph.block_id = event_num
        self.current_graph.block_start_time = start_time
        self.current_graph.block_end_time = end_time

        snapshot = snapshot_active_state(self.current_graph, event_num)
        self.stg.block_graphs.append(snapshot)

        if self.args.verbose:
            active_count = len(snapshot.food_nodes)
            print(f"    Snapshot: {active_count} active food items")

    def _log_event(self, event_type: str, data: Dict):
        """Log event to JSONL file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data
        }
        with open(self.event_log_file, 'a') as f:
            f.write(json.dumps(event) + "\n")

    def save_vlm_log(
        self,
        video_id: str,
        event_id: str,
        prompt_type: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        parsed: Any
    ):
        """Save VLM interaction log."""
        log_entry = {
            "video_id": video_id,
            "event_id": event_id,
            "prompt_type": prompt_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": response,
            "parsed_response": parsed
        }

        # Save JSON
        log_file = self.vlm_log_dir / f"{event_id}_{prompt_type}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)

        # Save text version
        txt_file = self.vlm_log_dir / f"{event_id}_{prompt_type}.txt"
        with open(txt_file, 'w') as f:
            f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
            f.write(f"=== USER PROMPT ===\n{user_prompt}\n\n")
            f.write(f"=== RAW RESPONSE ===\n{response}\n\n")
            f.write(f"=== PARSED RESPONSE ===\n{json.dumps(parsed, indent=2)}\n")

    def collect_vlm_logs(self, state_changes: List[Dict]) -> Dict[str, Any]:
        """
        Collect VLM logs for all events and return as a dictionary indexed by event_id.
        This enables the visualizer to display debugging info.
        """
        vlm_logs = {}

        # Look for transaction logs in vlm_log_dir
        for i in range(1, len(state_changes) + 1):
            event_key = f"event_{i:03d}"
            log_file = self.vlm_log_dir / f"{event_key}_transaction.json"

            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)

                    # Extract key fields for debugging display
                    vlm_logs[event_key] = {
                        "video_id": log_data.get("video_id"),
                        "user_prompt": log_data.get("user_prompt", ""),
                        "raw_response": log_data.get("raw_response", ""),
                        "parsed_response": log_data.get("parsed_response", {})
                    }
                except Exception as e:
                    print(f"    WARNING: Could not read VLM log {log_file}: {e}")

        return vlm_logs

    def run(self):
        """Main execution flow."""
        print("="*60)
        print("FOOD GRAPH FROM GEMINI PRE-ANNOTATIONS")
        print("="*60)
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Model:  {self.args.model}")
        print(f"Dry run: {self.args.dry_run}")

        # Load inputs
        inventory = self.load_inventory()
        state_changes = self.load_state_changes()

        if not state_changes:
            print("\nERROR: No state changes to process")
            return

        # Process inventory first
        initial_food_ids = self.process_inventory(inventory)

        # Process state changes
        self.process_state_changes(state_changes, initial_food_ids)

        # Update metadata
        self.stg.metadata["last_updated"] = datetime.now().isoformat()
        self.stg.metadata["total_events"] = len(state_changes)
        self.stg.metadata["total_inventory"] = len(inventory)
        self.stg.metadata["total_snapshots"] = len(self.stg.block_graphs)

        # Save final graph
        self.stg.save(str(self.graph_file))

        # Collect and embed VLM logs into the graph JSON
        print(f"\n[VLM Logs] Collecting VLM reasoning logs...")
        vlm_logs = self.collect_vlm_logs(state_changes)
        print(f"  Collected {len(vlm_logs)} VLM logs")

        # Re-load the graph JSON and add vlm_logs + inventory
        with open(self.graph_file, 'r') as f:
            graph_data = json.load(f)

        graph_data["vlm_logs"] = vlm_logs
        graph_data["inventory"] = inventory
        graph_data["state_changes"] = state_changes

        with open(self.graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)

        # Summary
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Graph saved: {self.graph_file}")
        print(f"Event log: {self.event_log_file}")
        print(f"VLM logs: {self.vlm_log_dir}")
        print(f"Clips: {self.clips_dir}")
        print(f"\nStats:")
        print(f"  Inventory items: {len(inventory)}")
        print(f"  State changes: {len(state_changes)}")
        print(f"  Snapshots: {len(self.stg.block_graphs)}")
        print(f"  Lineage edges: {len(self.stg.lineage_edges)}")
        print(f"  Embedded VLM logs: {len(vlm_logs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Food Graph from Gemini Pre-Annotations"
    )

    # Input paths
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing inventory.json and state_change.json'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=_PROJECT_ROOT / "P01" / "participant_P01_narrations.csv",
        help='Path to narrations CSV'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        default=_PROJECT_ROOT / "P01",
        help='Directory containing video files'
    )

    # Output paths
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=_PROJECT_ROOT / "gemini_outputs" / "food_graph_gemini",
        help='Output directory for graph and logs'
    )

    # Processing options
    parser.add_argument(
        '--model',
        default='qwen',
        choices=['qwen', 'gpt-4o'],
        help='VLM model for transaction inference (default: qwen)'
    )
    parser.add_argument(
        '--clip-buffer',
        type=float,
        default=2.0,
        help='Buffer seconds before/after event timestamps (default: 2.0)'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Disable video (force text-only VLM queries)'
    )

    # Flags
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse inputs without VLM calls (use default values)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    runner = GeminiPipelineRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
