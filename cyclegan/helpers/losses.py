import tensorflow as tf

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 2


@tf.function
def generator_loss(generated):
        return loss_obj(tf.ones_like(generated), generated)


@tf.function
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


@tf.function
def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * loss * 0.01
