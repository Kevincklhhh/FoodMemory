Part 1: The Food Knowledge Graph (Text-Only Version)
The core of the system is a graph database designed to represent two primary concepts: the food items themselves (Food Nodes) and the semantic places they can be (Activity Zone Nodes). The goal is to create a digital twin of your kitchen's inventory and its history.

Activity Zone Node
An Activity Zone Node represents a distinct semantic location within the kitchen. Since we are not using video or 3D models, these zones are defined purely by the text used to describe them.

Purpose: To answer "Where is my food?".

Properties:

zone_id (Primary Key): A unique identifier, e.g., zone_fridge_1, zone_counter_main.

name: A human-readable name inferred from narrations, e.g., "fridge", "counter", "pantry", "cupboard", "drawer". Zones are single-level only (no sub-zones like "top shelf of fridge").

type: A categorical label for the zone's function, e.g., Storage, PreparationSurface, Appliance.

Food Node
A Food Node represents a specific instance of a food item, not the general concept. For example, it tracks this specific carton of milk bought on Tuesday, not "milk" in general. This is the central node for tracking inventory and history.

**Scope Definition**: Food Nodes include:
- Consumable/edible items: milk, coffee, oranges, cheese, etc.
- Immediate containers when holding food: milk bottle (with milk), mug (with coffee), coffee capsule

**Exclusions**: Pure tools and appliances (scissors, frothers, machines) are not tracked as Food Nodes.

Purpose: To answer "What food do I have?", "How much is left?", and "Is this still safe to eat?".

Properties:

food_id (Primary Key): A unique identifier, e.g., food_milk_carton_123.

name: The common name of the food, e.g., "milk", "block of cheese", "sweet potato".

state: A dynamic property describing the food's current condition. Examples: raw, chopped, cooked, frozen, opened, half-used, sealed.

first_seen_time: A static timestamp (start_time of the first interaction) indicating when this food item was first added to the knowledge graph.

quantity: A dynamic property describing the amount. This could be a structured object like { "value": 0.5, "unit": "gallon" } or a descriptive string like "approx. 50g".

location (Foreign Key): The zone_id of the Activity Zone Node where the food is currently located. This property directly creates the -[:LOCATED_AT]-> relationship. A value of null or in_hand means it is currently being used.

interaction_history: A chronologically ordered list of all interactions with this food item. This is the core property for use history and safety analysis. Each entry in the list is an object with the following fields:

start_time: The start time of the interaction (from CSV start_timestamp).

end_time: The end time of the interaction (from CSV end_timestamp).

action: A normalized verb describing the interaction (e.g., take, place, chop, cook, open). One narration produces one history entry.

narration_text: The original, full narration text for context and debugging.

location_context: The zone_id where the action took place.

Part 2: The LLM-Powered KG Update Pipeline (from Narration)
This pipeline describes the step-by-step process of turning a single, timestamped narration into a structured update on the Knowledge Graph.

Step 0: Input
The process starts with a new narration from the CSV dataset.

Input CSV Row:
- video_id: 'P01-20240202-110250' (provides date/time context)
- start_timestamp: 21.4 (seconds)
- end_timestamp: 22.0 (seconds)
- narration: "pick up the milk bottle using the handle, which is located in the lower shelf of the door of the fridge"
- nouns: ['milk bottle', 'handle', 'lower shelf of door of fridge'] (pre-extracted)
- verbs: ['pick up'] (pre-extracted)

Step 1: Entity Extraction & Resolution
The program leverages the pre-extracted nouns from the CSV and determines if they refer to existing items in the KG.

Extract Entities: Use the pre-parsed 'nouns' field from CSV to identify potential food and location entities.

Result: food_entity: "milk bottle", location_entity: "fridge" (simplified from "lower shelf of door of fridge").

Check for Food Entity: If no food_entity is found, the process stops. No KG update happens.

Resolve Food Entity (The "Look Up"): This is a critical step. The program queries the KG to find the specific Food Node this narration refers to.

Query: SEARCH FoodNode WHERE name LIKE '%milk%' AND location.name LIKE '%fridge%'

Outcome 1 (Match Found): The query returns food_milk_bottle_456. The program retrieves this node's current properties (its state, quantity, location, etc.).

Outcome 2 (No Match): No existing "milk bottle" is in the "fridge". The system flags this as a potential new item.

Outcome 3 (Ambiguous Match): The query returns two milk bottles in the fridge. The system flags this ambiguity. For now, it might default to the one with the most recent interaction.

Step 2: Context Assembly for LLM
The program constructs a detailed prompt for the main Reasoning LLM. This prompt bundles the new information with the relevant history.

Prompt Components:

System Role: "You are a knowledge graph expert. Your task is to analyze a user's action and generate a JSON command to update a food inventory graph. You must infer the change in state and location."

Current KG Context: (If a match was found) "The user is interacting with food_milk_bottle_456. Its current properties are: { name: 'milk bottle', state: 'sealed', location: 'zone_fridge_1' }."

New Information: "The new user action occurred from 21.4s to 22.0s in video P01-20240202-110250 and was described as: pick up the milk bottle using the handle, which is located in the lower shelf of the door of the fridge."

Instructions & Output Format: "Based on this, generate a JSON update. The possible actions are TAKE, PLACE, MODIFY_STATE, CREATE_NEW. Infer the new location and state. The milk bottle was picked up, so its new location is likely 'in_hand' (null). The new interaction must be added to its history with start_time and end_time."

Step 3: LLM Inference for KG Update
The LLM processes the prompt and generates a structured JSON object representing the required KG update.

LLM JSON Output:

JSON

{
  "update_type": "TAKE",
  "target_food_id": "food_milk_bottle_456",
  "updates": {
    "location": null
  },
  "history_entry": {
    "start_time": 21.4,
    "end_time": 22.0,
    "action": "pick up",
    "narration_text": "pick up the milk bottle using the handle, which is located in the lower shelf of the door of the fridge",
    "location_context": "zone_fridge_1"
  }
}
Step 4: KG Update Execution
The program receives this JSON, validates it, and executes the commands against the graph database.

It finds the Food Node with food_id: "food_milk_bottle_456".

It updates the location property to null (indicating the milk bottle is now in hand).

It appends the new history_entry object to the interaction_history list property of the node.
