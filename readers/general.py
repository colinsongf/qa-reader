"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-15 11:04:03
	* @modify date 2018-05-15 11:04:03
	* @desc [description]
"""
import tensorflow as tf


def get_feed_dict(model, batch, dropout_keep_prob):
    feed_dict = {model.p: batch['passage_token_ids'],
                 model.q: batch['question_token_ids'],
                 model.p_length: batch['passage_length'],
                 model.q_length: batch['question_length'],
                 model.start_label: batch['start_id'],
                 model.end_label: batch['end_id'],
                 model.dropout_keep_prob: dropout_keep_prob}
    return feed_dict


def get_opt(optim_type, learning_rate):
    if optim_type == 'adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optim_type == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif optim_type == 'adamdelta':
        opt = tf.train.AdadeltaOptimizer(learning_rate)
    elif optim_type == 'rprop':
        opt = tf.train.RMSPropOptimizer(learning_rate)
    elif optim_type == 'sgd':
        opt = tf.train.GradientDescentOptimizer(
            learning_rate)
    else:
        raise NotImplementedError(
            'Unsupported optimizer: {}'.format(optim_type))
    return opt


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            assert g is not None, var.name
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
