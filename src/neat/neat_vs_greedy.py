from src.neat.env import *
from src.neat.agent import DurakNEAT, NEATPlayer

from tqdm import tqdm


class GreedyAgent:
    @staticmethod
    def act(env):
        if env.done:
            return None

        va = valid_actions(env)

        current_player = env.state
        attacker = env.attacker

        hand = env.hands[current_player]

        if current_player == attacker:
            # Атакующий
            # Отфильтруем те действия, которые != None
            card_actions = [a for a in va if a != 36]
            if not card_actions:
                # только None
                return 36
            # Выберем самую "младшую" по рангу
            best_card = min(card_actions, key=lambda c: (c//4))  # rank = c//4, suit = c%4
            # Но если "None" тоже в va, надо решить, хотим ли мы "пасовать"?
            # Скажем, если стол пуст, и у нас есть карты, мы не будем пасовать.
            # Упростим: атакующий никогда не пасует, если может выложить карту.
            return best_card

        else:
            # Защитник
            # Ищем, бьётся ли какая-то непокрытая атака
            uncovered = []
            for (ac, dc) in env.table_cards:
                if ac is not None and dc is None:
                    uncovered.append(ac)

            if not uncovered:
                # нет непокрытых => мы можем только взять или пас — но если valid_actions позволяет,
                # возможно там всё покрыто. Упростим - тогда None = "бездействие"
                # Но, в классике, если нет uncovered, защищающемуся вообще нечего делать
                # => пусть возвращает None.
                return None

            # Попробуем найти карты, которые бьют хотя бы одну атаку,
            # и выберем самую "дешёвую" (младшую), чтобы не тратить сильные.
            suitable_cards = []
            for a_def in va:
                if a_def is not None and a_def != 36:
                    for c_att in uncovered:
                        if can_beat(c_att, a_def, env.trump_suit):
                            suitable_cards.append(a_def)
                            break

            if suitable_cards:
                # Выбираем "минимальную" (младшую) по рангу
                best_card = min(suitable_cards, key=lambda c: (c//4))
                return best_card
            else:
                # нет подходящих — берём
                return 36

if __name__ == "__main__":
    env = DurakEnv()
    durak_neat = DurakNEAT(env)
    trained_agent = durak_neat.load_agent()

    n = 100
    s = 0

    for _ in tqdm(range(n)):
        obs, payoff, done = env.reset()
        agent0 = GreedyAgent()
        agent1 = NEATPlayer(trained_agent)

        while not done:

            current_player = env.state

            if current_player == 0:
                action = agent0.act(env)
            else:
                action = agent1.act(env)
            obs, payoff, done = env.step(action)

        s += payoff

    print(s, (s+n) // 2, n - (s + n) // 2)
