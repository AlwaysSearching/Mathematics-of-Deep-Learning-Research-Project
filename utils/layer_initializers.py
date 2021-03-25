import tensorflow as tf

def compute_fan_in_fan_out(shape):
    """
        Computes the number of input and output units for a weight shape.
        Parameters
        ----------
            shape: Integer shape tuple or TF tensor shape.
        Returns:
            A tuple of integer scalars (fan_in, fan_out).
    """
    
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        
        for dim in shape[:-2]:
            receptive_field_size *= dim

        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    
    return int(fan_in), int(fan_out)


def mean_field_init(alpha=1, mode='normal'):
    '''
        Returns a Tensorflow keras initializer which can be passsed to keras.model layers.
        For mode = 'normal', the initializer is a random normal distribution with stddev = (1 / m) * alpha
        For mode = 'uniform', the initializer is a uniform distribution with l_bound = u_bound = (1 / m) * alpha
        In the above, m is the fan_in value as calculated by compute_fan_in_fan_out.
    '''
    
    assert mode in ['uniform', 'normal'] 
    
    if mode == 'normal':    
        init = tf.keras.initializers.random_normal()
    else:
        init = tf.keras.initializers.random_uniform()
        
    def mean_field(shape, dtype):
        fan_in, fan_out = compute_fan_in_fan_out(shape)

        scale = alpha * 1 / fan_in
        
        if mode == 'normal':    
            init = tf.keras.initializers.random_normal(mean=0, stddev=scale)
        else:
            init = tf.keras.initializers.random_uniform(minval=-scale, maxval=scale)
        
        return init(shape, dtype)       
        
    return mean_field


def scaled_He_initializer(alpha=1, mode='uniform'):
    '''
        Returns a modified He initializer with scale adjusted by alpha and a preset seed.   
        The default value is equivalent to the initialization used in Double Deep Descent.
        
        Note that the scale parameter for he_uniform is placed under a square root, so as to appropriatly
        scale the initializer, the alpha value is squared. 
    '''
    
    assert mode in ['uniform', 'normal'] 
    
    if mode == 'normal':    
        init = tf.keras.initializers.he_normal()
    else:
        init = tf.keras.initializers.he_uniform()
        
    init.scale = init.scale * (alpha**2)
    
    return init       