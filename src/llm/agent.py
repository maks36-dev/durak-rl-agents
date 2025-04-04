import asyncio

from dotenv import load_dotenv

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
from llama_index.core.workflow import Context

from src.llm.env import DurakEnv
from src.po_mcts.greedy_player import GreedyAgent
from src.llm.utils import card_id_to_rank_suit

load_dotenv()

system_prompt = """
You are professional player in card game Durak. In this version of game there are 36 cards. In one step you can choose one card and make action
In this game you have id 0:
Every step you get observation in such type: 
obs = (
    trump_suit,
    attacker, if 0 you attack
    defender, if 0 you defend
    table cards,
    size of deck,
    your_hand,
    size of opponent hand
)

Every card is a number in the range 0-36. In card definition you can see the rule how transform card id to rank and suit

Action is a card or None. None in attack position means that you finish attack. None in defend position means that you get all cards from table and next you continue in defend position

#### Cards definition
RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['clubs', 'diamonds', 'hearts', 'spades']

def card_id_to_rank_suit(card_id: int) -> Tuple[str, str]:
    rank_i = card_id // 4
    suit_i = card_id % 4
    return RANKS[rank_i], SUITS[suit_i]

#### INPUT
(
    trump_suit,
    attacker, if 0 you attack
    defender, if 0 you defend
    table_as_tuple,
    size of deck,
    your_hand,
    size of opponent hand
)

#### OUTPUT
Return 1 action and nothing else: Card in range 0-36 which you has in your hand or None in such form: "34" or "None"
"""

async def evaluate(agent, ctx, query):
    query = str(query)
    response = await agent.run(query, ctx=ctx)
    return response

class LLMAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct"):
        self.llm = HuggingFaceInferenceAPI(model_name=model_name)
        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[card_id_to_rank_suit],
            llm=self.llm,
            system_prompt=system_prompt
        )
        self.ctx = Context(self.agent)

    async def act(self, env):
        obs, _, _ = env.get_current_obs()
        result = await evaluate(self.agent, self.ctx, obs)
        try:
            result = str(result)
            if result == "None":
                return None
            else:
                return int(result)
        except:
            return -1


async def play_one_game(env, agent0, agent1):
    obs, payoff, done = env.reset()
    while not done:
        current_player = env.state

        if current_player == 0:
            action = await agent0.act(env)
        else:
            action = agent1.act(env)

        obs, payoff, done = env.step(action)

    return payoff

async def main():
    env = DurakEnv()
    llm_agent = LLMAgent()
    greedy_agent = GreedyAgent()

    await play_one_game(env, llm_agent, greedy_agent)

if __name__ == "__main__":
    asyncio.run(main())