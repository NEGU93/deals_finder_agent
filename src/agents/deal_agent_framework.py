import os
import sys
import json
import logging
import chromadb
from typing import List
from src import DB
from src.agents.planning_agent import PlanningAgent
from src.schemas import Opportunity


# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = [
    "red",
    "blue",
    "brown",
    "orange",
    "yellow",
    "green",
    "purple",
    "cyan",
]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        settings = chromadb.Settings(
            allow_reset=True, anonymized_telemetry=False
        )
        client = chromadb.PersistentClient(path=DB, settings=settings)

        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> dict:
        """Read memory as a dictionary keyed by URL"""
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            # Convert list to dict, keyed by URL
            if isinstance(data, list):
                # Handle old format (list)
                memory_dict = {
                    item["deal"]["url"]: Opportunity(**item) for item in data
                }
            else:
                # Already dict format
                memory_dict = {
                    url: Opportunity(**item) for url, item in data.items()
                }
            return memory_dict
        return {}

    def write_memory(self) -> None:
        """Write memory dict to JSON"""
        data = {
            url: opportunity.dict() for url, opportunity in self.memory.items()
        }
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> dict:
        self.init_agents_as_needed()
        self.log("Kicking off Planning Agent")
        # Pass memory as list of opportunities for backward compatibility
        memory_list = list(self.memory.values())
        result = self.planner.plan(memory=memory_list)
        self.log(f"Planning Agent has completed and returned: {result}")
        if result:
            url = result.deal.url
            if url not in self.memory:  # O(1) lookup!
                self.memory[url] = result
                self.write_memory()
                self.log("✅ New deal added to memory")
            else:
                self.log("⚠️ Deal already in memory, skipping duplicate")
        return self.memory


if __name__ == "__main__":
    DealAgentFramework().run()
