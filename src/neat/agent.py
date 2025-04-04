import pickle
import neat
import numpy as np

from src.neat.env import DurakEnv, valid_actions, encode_observation


class DurakNEAT:
    def __init__(self, env, generations=100, population_size=50):
        self.env = env
        self.generations = generations
        self.population_size = population_size

        self.config_path = "config/neat_config.ini"
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())

    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0

        for _ in (range(10)):
            env = DurakEnv()
            obs, _, done = env.reset()
            while not done:
                state_vec = encode_observation(obs)

                action = np.argmax(np.exp(net.activate(state_vec)) * valid_actions(env, True))
                obs, reward, done = env.step(action)
                fitness += reward

        return fitness / 10

    def eval_population(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)

    def train(self):
        winner = self.population.run(self.eval_population, self.generations)

        # save best genome
        with open("best_genome/best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)

        return winner

    def load_agent(self, filename="best_genome/best_genome.pkl"):
        with open(filename, "rb") as f:
            best_genome = pickle.load(f)
        return neat.nn.FeedForwardNetwork.create(best_genome, self.config)



class NEATPlayer:
    def __init__(self, net):
        self.net = net

    def act(self, env):
        state, payoff, done = env.get_current_obs()
        state_vec = encode_observation(state)
        return int(np.argmax(np.exp(self.net.activate(state_vec)) * valid_actions(env, True)))


if __name__ == "__main__":
    env = DurakEnv()
    durak_neat = DurakNEAT(env, generations=100)
    best_genome = durak_neat.train()

    trained_agent = durak_neat.load_agent()