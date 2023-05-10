import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from Baby import Baby
from time import sleep

class StreamlitBaby(Baby):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.iteration_placeholder = st.empty()
        self.state_text_placeholder = st.empty()
        self.q_table_heatmap_placeholder = st.empty()
        st.markdown("""    
        $$
        Q(s_t, a_t) \\leftarrow Q(s_t, a_t) + \\alpha \\cdot [r_{t+1} + \\gamma \\cdot \\max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]
        $$
        
        - $Q(s_t, a_t)$: The Q-value of the current state ($s_t$) and action ($a_t$).
        - $\\alpha$: The learning rate
        - $r_{t+1}$: The immediate reward received after taking action $a_t$ in state $s_t$.
        - $\\gamma$: The discount factor, which determines how much future rewards are valued 
        - $\\max_{a} Q(s_{t+1}, a)$: Max Q-value for any action in the next state ($s_{t+1}$).
        """)

    def learn_and_display(self, num_episodes, learning_rate, discount_factor):
        self.q_table = self._initialize_q_table()
        for episode in range(num_episodes):
            self.iteration_placeholder.text(f'EPISODE: {episode + 1}')
            state = "back"
            done = False
            epsilon = self._calculate_epsilon(episode)
            while not done:
                action, eps = self._choose_action(state, epsilon)
                rnd = '*' if eps < epsilon else ''
                next_state, reward, done = self._perform_action(state, action)
                self.state_text_placeholder.text(f'STATE: {state:15} | ACTION: {action:15}{rnd:1} | NEXT STATE: {next_state:15}\n(* = random pick; percentage of random picks should decrease with episodes)')
                current_q_value = self.q_table[state][action]
                max_next_q_value = max(self.q_table[next_state].values())
                self.q_table[state][action] = current_q_value * (1-learning_rate) +\
                    learning_rate * (reward + discount_factor * max_next_q_value)
                state = next_state
                self.plot_q_table()
                sleep(1)
            

        st.write(f"\nThe best rolling strategy for the baby is: {self.best_rolling_strategy()}")
        self.plot_q_table()

    def plot_q_table(self):
        self.ax.clear()
        df = pd.DataFrame(self.q_table).T
        sns.heatmap(df, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, ax=self.ax)
        self.ax.set_xlabel('Actions')
        self.ax.set_ylabel('States')
        self.ax.set_title('Q-table')
        self.q_table_heatmap_placeholder.pyplot(self.fig)
