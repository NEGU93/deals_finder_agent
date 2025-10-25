from pydantic import BaseModel
from typing import List


class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """

    product_description: str
    price: float
    url: str


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal]


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """

    deal: Deal
    estimate: float
    discount: float
