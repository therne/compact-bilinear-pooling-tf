from count_sketch import count_sketch, bilinear_pool
import tensorflow as tf

# vanila test.
with tf.Session() as sess:
    p1 = tf.constant([[0.83, 0.50, 0.10, 0.82, 0.24, 0.11, 0.21, 0.51, 0.39, 0.10, 0.67, 0.18]])
    p2 = tf.constant([[0.18, 0.59, 0.56, 0.12, 0.33, 0.93, 0.17, 0.58, 0.16]])
    d = 6

    bcp = bilinear_pool(p1, p2, d)
    sp1 = count_sketch(p1, d)

    sess.run(tf.initialize_all_variables())
    pp1, pbcp = sess.run([sp1, bcp])

    # print('pp1=================')
    # print(pp1)
    # print('bcp================')
    # print(pbcp)

    der, num = tf.test.compute_gradient(p1, [1, 12], sp1, [1, 6])
    print('Theorical Gradient : ' + str(der))
    print('Numerical Gradient : ' + str(num))

    err = tf.test.compute_gradient_error(p1, [1, 12], sp1, [1, 6])
    print('Gradient Error : %.2f' % err)

    assert err < 1e-2
