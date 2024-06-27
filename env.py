import gym

# Create the environment
env = gym.make('CartPole-v1')

# Reset the environment
observation = env.reset()

# Main loop
for t in range(100):
    # Perform a random action
    action = env.action_space.sample()
    step_result = env.step(action)
    
    # Print to debug
    print(f"Step: {t}, Action: {action}, Step Result: {step_result}")

    # Unpack the step result
    if len(step_result) == 4:
        observation, reward, done, info = step_result
    elif len(step_result) == 5:  # Handle the case where there's an extra element
        observation, reward, done, _, info = step_result

    # Render the environment
    env.render()

    if done:
        break

# Close the environment at the end
env.close()

