import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        num_obs = len(input_observation_states)  # number of observations
        num_states = len(self.hidden_states)  # number of hidden states

        # edge case: if number of observations = 0, then return 0
        if num_obs == 0:
            return 0.0  

        # converting observation sequence into indices for easier lookup
        obs_indices = [self.observation_states_dict[obs] for obs in input_observation_states]

        # Step 2. Calculate probabilities
        # creates a table to store probability of each state at each time step 
        forward_probability = np.zeros((num_obs, num_states)) # num_obs = nrow, num_states = ncols

        for i in range(num_states): # looping over hidden states 
          forward_probability[0, i] = self.prior_p[i] * self.emission_p[i, obs_indices[0]] # multiplying the prior probability by the emission probability

        # using previous probabilities and transition/emission probabilities to update the table
        for t in range(1, num_obs): 
            for j in range(num_states): 
                forward_probability[t, j] = sum(
                    forward_probability[t - 1, i] * self.transition_p[i, j] * self.emission_p[j, obs_indices[t]]
                    for i in range(num_states)
                )

        # Step 3. Return final probability 
        return np.sum(forward_probability[num_obs - 1]) # total probability of the observation sequence
        

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence of observed states
        """        
        
        # Step 1. Initialize variables
        
        num_obs = len(decode_observation_states)  # number of observations
        num_states = len(self.hidden_states)  # number of hidden states

        # edge case: if number of observations = 0, then return an empty sequence
        if num_obs == 0:
            return []

        # store probabilities of best path for each state at each step
        viterbi_table = np.zeros((num_obs, num_states))  

        # store best previous state for traceback
        backpointer = np.zeros((num_obs, num_states), dtype=int)

        # convert observations to indices
        obs_indices = [self.observation_states_dict[obs] for obs in decode_observation_states]  

        # Step 2. Calculate Probabilities (Initialization for t=0)
        for i in range(num_states):
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, obs_indices[0]]
            backpointer[0, i] = 0  # No previous state at t=0
        
        # recursion for t > 0; finding the best previous state at each step
        for t in range(1, num_obs):
            for j in range(num_states):
                max_prob, best_prev_state = max(
                    (viterbi_table[t - 1, i] * self.transition_p[i, j], i) 
                    for i in range(num_states)
                )
                viterbi_table[t, j] = max_prob * self.emission_p[j, obs_indices[t]]
                backpointer[t, j] = best_prev_state  # store the best previous state
        
        # Step 3. Traceback to find best hidden state sequence
        best_path = np.zeros(num_obs, dtype=int)
        best_path[-1] = np.argmax(viterbi_table[num_obs - 1])  # start from the last time step and trace back

        for t in range(num_obs - 2, -1, -1):  # work backwards to reconstruct best sequence
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # Step 4. Return best hidden state sequence as a list of state names
        best_hidden_state_sequence = [self.hidden_states[i] for i in best_path] # convert indices back to hidden states

        return best_hidden_state_sequence
        