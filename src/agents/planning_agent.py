from typing import Optional, List
from src.agents.agent_logger import AgentLogger
from src.schemas import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from src.agents import GPT4MiniRAG


class PlanningAgent(AgentLogger):
    name = "Planning Agent"
    color = AgentLogger.GREEN
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        self.ensemble = GPT4MiniRAG(collection)
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(
            f"Planning Agent has processed a deal with discount ${discount:.2f}"
        )
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        TBD: 3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)
        if selection:
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(
                f"Planning Agent has identified the best deal has discount ${best.discount:.2f}"
            )
            # if best.discount > self.DEAL_THRESHOLD:
            #     self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None
