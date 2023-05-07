import random
from tabulate import tabulate
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#%%

'''
lift_head: The baby lifts its head and looks towards the side it wants to roll over.
reach_arm: The baby reaches out with the arm on the side it wants to roll over.
bend_leg: The baby bends the leg opposite to the side it wants to roll over.
kick_leg: The baby kicks with the bent leg, using the momentum to roll over.
'''

class Baby:
    def __init__(self):
        self.states = ["back", "lifted_head", "reached_arm", "bent_leg", "tummy"]
        self.actions = ["lift_head", "reach_arm", "bend_leg", "kick_leg"]

        self.q_table = {
            state: {action: 0 for action in self.actions} for state in self.states
        }
        self.state_values = {"back": 0, "lifted_head": 1, "reached_arm": 2, "bent_leg": 3, "tummy": 4}


    def _initialize_q_table(self):
        q_table = {}
        for state in self.states:
            q_table[state] = {action: 0 for action in self.actions}
        return q_table

    def learn(self, num_episodes, learning_rate, discount_factor):
        for episode in range(num_episodes):
            state = "back"
            done = False
            epsilon = self._calculate_epsilon(episode)

            while not done:
                action = self._choose_action(state, epsilon)
                next_state, reward, done = self._perform_action(state, action)

                # Q-learning update rule
                current_q_value = self.q_table[state][action]
                max_next_q_value = max(self.q_table[next_state].values())
                self.q_table[state][action] = current_q_value + learning_rate * (reward + discount_factor * max_next_q_value - current_q_value)

                state = next_state
                
    def _calculate_epsilon(self, episode):
        # Epsilon decay: start with 1 and decay towards a minimum value over time
        epsilon_min = 0.01
        epsilon_decay = 0.995
        epsilon = max(epsilon_min, epsilon_decay ** episode)
        return epsilon

    def _choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def _perform_action(self, state, action):
        if state == "back" and action == "lift_head":
            return "lifted_head", self._calculate_reward(state, "lifted_head"), False
        elif state == "lifted_head" and action == "reach_arm":
            return "reached_arm", self._calculate_reward(state, "reached_arm"), False
        elif state == "reached_arm" and action == "bend_leg":
            return "bent_leg", self._calculate_reward(state, "bent_leg"), False
        elif state == "bent_leg" and action == "kick_leg":
            return "tummy", self._calculate_reward(state, "tummy"), True
        else:
            return state, self._calculate_reward(state, state), False
    
    def _calculate_reward(self, current_state, next_state):
        progress = self.state_values[next_state] - self.state_values[current_state]
        if progress > 0:
            return progress
        else:
            return -0.1


       
    def print_q_table(self):
        table = []
        for state in self.states:
            row = [state] + [self.q_table[state][action] for action in self.actions]
            table.append(row)
    
        headers = ["State"] + self.actions
        print(tabulate(table, headers=headers))


    def best_rolling_strategy(self):
        return {state: max(self.q_table[state], key=self.q_table[state].get) for state in self.states}
    
    def plot_q_table(self):
        df = pd.DataFrame(self.q_table).T
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, ax=ax)
    
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.title('Q-table Heatmap')
        plt.show()

def main():
    baby = Baby()
    num_episodes = 100
    learning_rate = 0.1
    discount_factor = 0.99
    
    print("Starting the baby's learning process...")
    for i in range(num_episodes):
        baby.learn(1, learning_rate, discount_factor)
        if i % 10 == 0:
            print(f"\nQ-table after {i + 1} episodes:")
            baby.print_q_table()
            baby.plot_q_table()

    best_strategy = baby.best_rolling_strategy()
    print(f"\nThe best rolling strategy for the baby is: {best_strategy}")

if __name__ == "__main__":
    main()
