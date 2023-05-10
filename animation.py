# -*- coding: utf-8 -*-
"""
Created on Tue May  9 20:42:20 2023

@author: mbelic
"""

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
#%%

class BabyFigure:
    def __init__(self, figure, ax):
        """
        Initialize a BabyFigure object with the given figure and axis.
        
        Args:
            figure (Figure): The matplotlib figure object.
            ax (Axes): The matplotlib axes object.
        """
        self.figure = figure
        self.ax = ax
        self.torso_circle = plt.Circle((0.5, 0.5), 0.1, color='blue')
        self.head_circle = plt.Circle((0.5, 0.6), 0.07, color='blue')
        self.left_arm = plt.Line2D([0.4, 0.45], [0.5, 0.55], linewidth=3, color='blue')
        self.right_arm = plt.Line2D([0.6, 0.55], [0.5, 0.55], linewidth=3, color='blue')
        self.left_leg = plt.Line2D([0.45, 0.45], [0.4, 0.35], linewidth=3, color='blue')
        self.right_leg = plt.Line2D([0.55, 0.55], [0.4, 0.35], linewidth=3, color='blue')
        self.state_text = None
        self.action_text = None
        self.iteration_text = None

        self.ax.add_artist(self.torso_circle)
        self.ax.add_artist(self.head_circle)
        self.ax.add_artist(self.left_arm)
        self.ax.add_artist(self.right_arm)
        self.ax.add_artist(self.left_leg)
        self.ax.add_artist(self.right_leg)

    def update(self, state, action, iteration_num):
        """
        Update the BabyFigure with the given state, action, and iteration number.
        
        Args:
            state (str): The current state of the baby.
            action (str): The action performed by the baby.
            iteration_num (int): The current iteration number of the learning process.
        """
        if action == "lift_head":
            self.left_arm.set_ydata([0.55, 0.6])
            self.right_arm.set_ydata([0.55, 0.6])
            self.head_circle.center = (0.5, 0.63)
            # Lift the head by changing its center y-coordinate
            
        elif action == "reach_arm":
            self.right_arm.set_xdata([0.6, 0.65])
            self.right_arm.set_ydata([0.55, 0.6])
        elif action == "bend_leg":
            self.left_leg.set_ydata([0.4, 0.45])
        elif action == "kick_leg":
            self.left_leg.set_ydata([0.4, 0.35])

        if self.state_text:
            self.state_text.remove()
        self.state_text = self.ax.text(0.3, 0.9,
                                       f"State: {state}",
                                       ha='center',
                                       va='center',
                                       fontsize=12)

        if self.action_text:
            self.action_text.remove()
        self.action_text = self.ax.text(0.3, 0.8,
                                        f"Action: {action}",
                                        ha='center',
                                        va='center',
                                        fontsize=12)
        
        if self.iteration_text:
          self.iteration_text.remove()
        self.iteration_text = self.ax.text(0.3, 0.7,
                                       f"Iteration: {iteration_num}",
                                       ha='center',
                                       va='center',
                                       fontsize=12)


    def get_artists(self):
        """
         Get a list of all artists (matplotlib objects) representing the BabyFigure.
         
         Returns:
             list: A list of matplotlib objects representing the BabyFigure.
         """

        return [self.torso_circle,
                self.head_circle,
                self.left_arm,
                self.right_arm,
                self.left_leg,
                self.right_leg,
                self.state_text,
                self.action_text,
                self.iteration_text]

    
    def draw_baby(self, state, action, iteration_num):
        """
        Draw the baby in the given state and action at the specified iteration number.
        
        Args:
            state (str): The current state of the baby.
            action (str): The action performed by the baby.
            iteration_num (int): The current iteration number of the learning process.
        
        Returns:
            list: A list of matplotlib objects representing the updated BabyFigure.
        """  
        self.update(state, action, iteration_num)
        return self.get_artists()
    
    def init_animation(self):
        self.draw_baby(0, '', 0)
        return self.get_artists()

    def animate(self, j, frames):
        """
        Animate the BabyFigure for the given frame index.
        
        Args:
            j (int): The index of the frame to animate.
            frames (list): A list of frames representing the baby's states and actions.
        
        Returns:
            list: A list of matplotlib objects representing the updated BabyFigure.
        """
        state, action, iteration_num = frames[j]
        self.draw_baby(state, action, iteration_num)
        return self.get_artists()

    def create_animation(self, frames):
        ani = FuncAnimation(self.figure, self.animate,
                            init_func=self.init_animation,
                            frames=len(frames),
                            fargs=(frames,),
                            interval=1000,
                            blit=True,
                            repeat=False)
        return ani

