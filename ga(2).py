import gym
import numpy as np
import matplotlib.pyplot as plt
import traceback

class NeuralNet:
    """
    Neural network to optimize the gym environment
    """
    def __init__(self, env_name, input_dim, hidden_dim, output_dim, test_run):
        self.env = gym.make(env_name, render_mode="human")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.test_run = test_run

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def init_weights(self):
        input_weight = []
        input_bias = []
        hidden_weight = []
        out_weight = []

        input_nodes = self.env.observation_space.shape[0]

        for i in range(self.test_run):
            inp_w = np.random.rand(self.input_dim, input_nodes)
            input_weight.append(inp_w)
            inp_b = np.random.rand(input_nodes)
            input_bias.append(inp_b)
            hid_w = np.random.rand(input_nodes, self.hidden_dim)
            hidden_weight.append(hid_w)
            out_w = np.random.rand(self.hidden_dim, self.output_dim)
            out_weight.append(out_w)

        return [input_weight, input_bias, hidden_weight, out_weight]

    def forward_prop(self, obs, input_w, input_b, hidden_w, out_w):
        if isinstance(obs, (list, tuple)):
            obs = np.array(obs, dtype=np.float32)
        elif isinstance(obs, np.ndarray):
            obs = obs.flatten().astype(np.float32)

        if len(obs.shape) > 1:
            obs = obs.flatten()

        norm_value = max(np.linalg.norm(obs), 1)
        if norm_value != 0:
            obs = obs / norm_value

        Ain = self.relu(np.dot(obs, input_w) + input_b.T)
        Ahid = self.relu(np.dot(Ain, hidden_w))
        Zout = np.dot(Ahid, out_w)
        A_out = self.relu(Zout)
        output = self.softmax(A_out)

        return np.argmax(output)

    def run_environment(self, input_w, input_b, hidden_w, out_w, render=False):
        obs, info = self.env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            if render:
                self.env.render()
            action = self.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, terminated, truncated, info = self.env.step(action)
            score += reward
            if terminated or truncated:
                break
        return score

    def run_test(self):
        generation = self.init_weights()
        input_w, input_b, hidden_w, out_w = generation
        scores = []
        for ep in range(self.test_run):
            score = self.run_environment(input_w[ep], input_b[ep], hidden_w[ep], out_w[ep])
            scores.append(score)
        return [generation, scores]


class GA:
    """
    Training neural net using genetic algorithm
    """
    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.5):
        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = []
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def crossover(self, DNA_list):
        newDNAs = []
        for _ in range(self.population_size):
            parent1_idx = np.random.randint(len(DNA_list))
            parent2_idx = np.random.randint(len(DNA_list))

            parent1 = DNA_list[parent1_idx]
            parent2 = DNA_list[parent2_idx]

            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            newDNAs.append(child)

        return newDNAs

    def mutation(self, DNA):
        for idx in range(len(DNA)):
            if np.random.rand() < self.mutation_rate:
                DNA[idx] += np.random.normal(0, 0.1, size=DNA[idx].shape)

        return DNA

    def next_generation(self):
        sorted_indices = np.argsort(self.current_fitness)[-2:]
        index_good_fitness = sorted_indices.tolist()

        new_DNA_list = []
        DNA_list = []

        for index in index_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(-1)

            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)

            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(-1)
            dna_w2 = np.append(dna_b1, dna_whid)

            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh.reshape(-1))
            DNA_list.append(dna)

        new_DNA_list += DNA_list

        new_DNA_list += self.crossover(DNA_list)
        new_DNA_list = [self.mutation(dna) for dna in new_DNA_list]

        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:
            newdna_in_w1 = np.array(newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(newdna_in_w1, self.current_generation[0][0].shape)
            new_input_weight.append(new_in_w)

            new_in_b = np.array(newdna[newdna_in_w1.size:newdna_in_w1.size + self.current_generation[1][0].size])
            new_input_bias.append(new_in_b)

            newdna_in_w2 = np.array(newdna[newdna_in_w1.size + self.current_generation[1][0].size:newdna_in_w1.size + self.current_generation[1][0].size + self.current_generation[2][0].size])
            new_hid_w = np.reshape(newdna_in_w2, self.current_generation[2][0].shape)
            new_hidden_weight.append(new_hid_w)

            new_out_w = np.array(newdna[newdna_in_w1.size + self.current_generation[1][0].size + self.current_generation[2][0].size:])
            new_out_w = np.reshape(new_out_w, self.current_generation[3][0].shape)
            new_output_weight.append(new_out_w)

        return [new_input_weight, new_input_bias, new_hidden_weight, new_output_weight]

    def show_fitness_graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_list, marker='o')
        plt.title('Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.show()

    def render_best_individual(self, best_weights):
        input_w, input_b, hidden_w, out_w = best_weights
        obs, info = self.learner.env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            self.learner.env.render()
            action = self.learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, done, truncated, info = self.learner.env.step(action)
            score += reward
            if done or truncated:
                break

    def evolve(self):
        for gen in range(self.number_of_generation):
            try:
                new_gen_weights = self.next_generation()
                self.current_generation = new_gen_weights
                fitness_scores = []

                for weights_idx in range(len(new_gen_weights[0])):
                    input_w = new_gen_weights[0][weights_idx]
                    input_b = new_gen_weights[1][weights_idx]
                    hidden_w = new_gen_weights[2][weights_idx]
                    out_w = new_gen_weights[3][weights_idx]

                    score = self.learner.run_environment(input_w, input_b, hidden_w, out_w)
                    fitness_scores.append(score)

                    if score > self.best_fitness:
                        self.best_gen = [input_w, input_b, hidden_w, out_w]
                        self.best_fitness = score

                self.fitness_list.append(np.max(fitness_scores))
                print(f"Generation {gen+1}, Best Fitness: {self.best_fitness}")

                self.render_best_individual(self.best_gen)

            except Exception as e:
                print(f"Error in generation {gen+1}: {e}")
                print(traceback.format_exc())

        print(f"Best fitness found: {self.best_fitness}")
        self.show_fitness_graph()

        return self.best_gen, self.best_fitness


def trainer(env_name='CartPole-v1', pop_size=15, num_of_generation=200):
    env = gym.make(env_name)
    learner = NeuralNet(env_name, env.observation_space.shape[0], 2, env.action_space.n, pop_size)
    init_weight_list, init_fitness_list = learner.run_test()
    
    ga = GA(init_weight_list, init_fitness_list, num_of_generation, pop_size, learner)
    best_weights, best_fitness = ga.evolve()
    
    return best_weights


def test_run_env(params, env):
    input_w, input_b, hidden_w, out_w = params
    learner = NeuralNet('CartPole-v1', env.observation_space.shape[0], 2, env.action_space.n, 15)
    obs, info = learner.env.reset()
    score = 0
    for t in range(5000):
        learner.env.render()
        action = learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
        obs, reward, terminated, truncated, info = learner.env.step(action)
        score += reward
        print(f"time: {t}, fitness: {score}")
        if terminated or truncated:
            break
    print(f"Final score: {score}")


def main():
    best_weights = trainer()
    env = gym.make('CartPole-v1', render_mode="human")
    test_run_env(best_weights, env)

if __name__ == "__main__":
    main()

