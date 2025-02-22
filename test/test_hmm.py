import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    # Step 1: Load the mini weather HMM model
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    # Step 2: Extract parameters from the HMM file
    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    # Step 3: Create an instance of HiddenMarkovModel
    hmm_model = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Step 4: Load observation sequences and expected outputs
    observation_sequence = mini_input['observation_state_sequence']
    expected_viterbi_path = mini_input['best_hidden_state_sequence']

    # Step 5: Run the Forward algorithm and check output
    forward_prob = hmm_model.forward(observation_sequence)
    assert isinstance(forward_prob, float), "Forward algorithm should return a float probability."

    # Step 6: Run the Viterbi algorithm and check output
    viterbi_path = hmm_model.viterbi(observation_sequence)
    assert viterbi_path == expected_viterbi_path.tolist(), "Viterbi algorithm returned incorrect state sequence."

    # Step 7: Edge cases (e.g., empty sequence, all same observation)
    assert hmm_model.forward([]) == 0, "Forward algorithm should return 0 probability for an empty sequence."
    assert hmm_model.viterbi([]) == [], "Viterbi algorithm should return an empty list for an empty sequence."



def test_full_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    # Step 1: Load the full weather HMM model
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    # Step 2: Extract parameters from the HMM file
    observation_states = full_hmm['observation_states']
    hidden_states = full_hmm['hidden_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']

    # Step 3: Create an instance of HiddenMarkovModel
    hmm_model = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Step 4: Load observation sequences and expected outputs
    observation_sequence = full_input['observation_state_sequence']
    expected_viterbi_path = full_input['best_hidden_state_sequence']

    # Step 5: Run the Forward algorithm and check output
    forward_prob = hmm_model.forward(observation_sequence)
    assert isinstance(forward_prob, float), "Forward algorithm should return a float probability."
    print("Forward Probability (Mini Weather):", hmm_model.forward(observation_sequence))

    # Step 6: Run the Viterbi algorithm and check output
    viterbi_path = hmm_model.viterbi(observation_sequence)
    assert viterbi_path == expected_viterbi_path.tolist(), "Viterbi algorithm returned incorrect state sequence."
    print("Viterbi Path (Mini Weather):", hmm_model.viterbi(observation_sequence))

    # Step 7: Edge cases (e.g., empty sequence, all same observation)
    assert hmm_model.forward([]) == 0, "Forward algorithm should return 0 probability for an empty sequence."
    assert hmm_model.viterbi([]) == [], "Viterbi algorithm should return an empty list for an empty sequence."












