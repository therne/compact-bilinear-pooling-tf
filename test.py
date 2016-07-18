import tensorflow as tf
from count_sketch import count_sketch, bilinear_pool

class CountSketchTest(tf.test.TestCase):
    def testGrad(self):
        with self.test_session():
            p = tf.constant([[0.83, 0.50, 0.10, 0.82, 0.24, 0.11, 0.21, 0.51, 0.39, 0.10, 0.67, 0.18]])
            d = 6
            sk = count_sketch(p, d)
            tf.initialize_all_variables().run()

            # thg, numg = tf.test.compute_gradient(p, [1, 12], sk, [1, 6])
            # print('Theorical Gradient : {}'.format(thg))
            # print('Numerical Gradient : {}'.format(numg))

            err = tf.test.compute_gradient_error(p, [1, 12], sk, [1, 6])
            print('SK Gradient Error : %.2f' % err)
            self.assertTrue(err < 1e-3, 'Gradient must be matched with numerical gradient.')


class BilinearPoolingTest(tf.test.TestCase):
    def testGrad(self):
        with self.test_session():
            p1 = tf.constant([[0.83, 0.50, 0.10, 0.82, 0.24, 0.11, 0.21, 0.51, 0.39, 0.10, 0.67, 0.18]])
            p2 = tf.constant([[0.18, 0.59, 0.56, 0.12, 0.33, 0.93, 0.17, 0.58, 0.16]])
            d = 6
            bp = bilinear_pool(p1, p2, d)

            tf.initialize_all_variables().run()

            err = tf.test.compute_gradient_error(p1, [1, 12], bp, [1, 6])
            print('BP Gradient Error (1) : %.2f' % err)
            self.assertTrue(err < 1e-3, 'Gradient must be matched with numerical gradient.')

            err = tf.test.compute_gradient_error(p2, [1, 9], bp, [1, 6])
            print('BP Gradient Error (2) : %.2f' % err)
            self.assertTrue(err < 1e-3, 'Gradient must be matched with numerical gradient.')

if __name__ == '__main__':
    tf.test.main()
