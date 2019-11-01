import tensorflow as tf

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def discriminator_loss(real, generated):
    random = tf.random.uniform(generated.shape)*tf.keras.backend.epsilon()*10
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated)+random, generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 200

@tf.function
def generator_loss(generated):
    random = tf.random.uniform(generated.shape) * tf.keras.backend.epsilon()*10

    return loss_obj(tf.ones_like(generated)-random, generated)

@tf.function
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1

@tf.function
def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.01 * loss
