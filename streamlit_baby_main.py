import streamlit as st
from StreamlitBaby import StreamlitBaby


def main():
    st.title("Baby Learns To Roll")

    baby = StreamlitBaby()

    # Define the input parameters using Streamlit widgets
    num_episodes = st.sidebar.slider("Number of episodes", 1, 20, 10)
    learning_rate = st.sidebar.slider("Learning rate", 0.0, 1.0, 0.1)
    discount_factor = st.sidebar.slider("Discount factor", 0.0, 1.0, 0.9)

    if st.button("Learn"):
        st.write("Look at that baby learning!")
        baby.learn_and_display(num_episodes, learning_rate, discount_factor)


if __name__ == "__main__":
    main()
