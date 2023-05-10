from matplotlib import pyplot as plt
from Baby import Baby
from animation import BabyFigure
#%%


def main():
    baby = Baby()
    num_episodes = 100
    learning_rate = 0.1
    discount_factor = 0.99
    
    
    baby.learn(num_episodes, learning_rate, discount_factor)
    
    # frames = []
    # for i in range(num_episodes):
    #     baby.learn(1, learning_rate, discount_factor)
    #     if i % 10 == 0:
    #         print(f"\nQ-table after {i + 1} episodes:")
    #         # baby.print_q_table()
    #         baby.plot_q_table()
    #         best_strategy = baby.best_rolling_strategy()
    #         # Append the iteration number to the frames
    #         frames.extend([(state, action, i + 1) for state, action in best_strategy.items()])


    best_strategy = baby.best_rolling_strategy()
    print(f"\nThe best rolling strategy for the baby is: {best_strategy}")

    # fig, ax = plt.subplots()
    # baby_figure = BabyFigure(fig, ax)
    
    # ani = baby_figure.create_animation(frames)

    # ani.save('animation.mp4',
    #          writer='ffmpeg',
    #          fps=1,
    #          dpi=100)
    # plt.close(fig)
   
    
if __name__ == "__main__":
    main()
