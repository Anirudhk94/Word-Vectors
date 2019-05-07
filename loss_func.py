import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    #A = log(exp({u_o}^T v_c))
    A = tf.reshape(tf.log(tf.exp(tf.reduce_sum(tf.multiply(true_w, inputs), 1))), [inputs.get_shape()[0],1])
    # tf.Print(A,[A])
    

    #B = log(sum{exp({u_w}^T v_c)})
    B = tf.reshape(tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, tf.transpose(true_w))), 1)), [inputs.get_shape()[0],1])
    # print(B.get_shape())

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    k              = len(sample) * 1.0
    batch_size     = inputs.get_shape()[0]
    embedding_size = inputs.get_shape()[1]
    unigram_prob   = tf.convert_to_tensor(unigram_prob)
    delta          = tf.exp(-10.0)

    #Converting labels to vectors using label indices
    
    labels_vector = tf.reshape(tf.nn.embedding_lookup(weights, labels), [batch_size, embedding_size]) 

    #Converting samples to vectors using sample indices
    sample_vector = tf.reshape(tf.nn.embedding_lookup(weights, sample), [len(sample), embedding_size]) 

    #POSITIVE : 
    # Calculating dot product for embeded and labels : s(w_o, w_c) = (u_c u_o) + b_o 
    s_pos = tf.reshape(tf.reduce_sum(tf.multiply(inputs, labels_vector), [1]), [batch_size, 1]) 
    biases_pos = tf.reshape(tf.nn.embedding_lookup(biases, labels), [batch_size ,1])
    s_pos = s_pos + biases_pos

    # Create a probability vector using label indices : sigmoid(s(w_o, w_c) - log(k * P(w_o)))
    prob_labels = tf.reshape(tf.nn.embedding_lookup(unigram_prob, labels), [batch_size, 1]) 
    valid_prob_total = tf.sigmoid(s_pos - tf.log(k * prob_labels + delta))

    #NEGATIVE : Calculating dot product for embeded and labels
    biases_neg = tf.reshape(tf.nn.embedding_lookup(biases, sample), [len(sample) ,1])
    s_neg = tf.matmul(sample_vector, tf.transpose(inputs)) + tf.tile(biases_neg, [1, batch_size])

    #NEGATIVE : Create a probability vector using label indices
    prob_noise = tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample), [len(sample), 1]) 
    prob_noise = tf.log(k * prob_noise + delta)
    s_neg = s_neg - tf.tile(prob_noise, [1, batch_size])
    noise_prob_total = tf.sigmoid(s_neg)

    A = tf.log(valid_prob_total + delta)
    B = tf.transpose(tf.reduce_sum(tf.log(1 - noise_prob_total + delta), axis = 0))
    return tf.negative(A + B)
