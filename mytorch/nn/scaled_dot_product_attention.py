import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim = -1)
        
    
    def forward(self, Q, K, V, mask=None):
        # Cache inputs for backward
        self.Q = Q
        self.K = K
        self.V = V

        # Calculate attention scores
        d_k = Q.shape[-1]

        scaled_dot_product = (
            Q @ np.swapaxes(K, -2, -1)
        ) / np.sqrt(d_k)

        # Apply mask
        if mask is not None:
            scaled_dot_product = np.where(
                mask, -np.inf, scaled_dot_product
            )

        # Identify fully masked rows
        fully_masked = np.all(
            np.isneginf(scaled_dot_product),
            axis=-1,
            keepdims=True
        )

        # Avoid NaN in softmax
        safe_scores = np.where(
            fully_masked, 0.0, scaled_dot_product
        )

        self.attention_scores = self.softmax.forward(safe_scores)

        # Fully masked rows must have zero attention probability
        self.attention_scores = np.where(
            fully_masked, 0.0, self.attention_scores
        )

        # Calculate output
        output = self.attention_scores @ V

        return output
        #raise NotImplementedError
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V
        d_V = (
            np.swapaxes(self.attention_scores, -2, -1) @ d_output
       )
        
        # Calculate gradients for attention scores
        d_attention_scores = (d_output @ np.swapaxes(self.V, -2, -1))
        d_scaled_dot_product = self.softmax.backward(
            d_attention_scores
        )
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = (d_scaled_dot_product / np.sqrt(self.Q.shape[-1]))
        
        # Calculate gradients for Q and K
        d_Q = d_scaled_dot_product @ self.K
        d_K = np.swapaxes(d_scaled_dot_product, -2, -1) @ self.Q
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V
        #raise NotImplementedError

