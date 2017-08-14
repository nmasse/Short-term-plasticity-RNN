"""
Nonfunctional.  Kept for reference.
"""

n = ((800//par['dt']) - 1) - 400//par['dt']
# u_0, u_1, v_0, v_1, cov = [tf.placeholder(tf.float32, shape = n)]*5

# print(self.hidden_state_hist)

# u_0, v_0 = zip(*[tf.nn.moments(h, axes=[0,1]) for h in self.hidden_state_hist[(400//par['dt']):(800//par['dt'])-1]])
# u_1, v_1 = zip(*[tf.nn.moments(h, axes=[0,1]) for h in self.hidden_state_hist[(400//par['dt'])+1:(800//par['dt'])]])
desired_corr = np.zeros((40, 40))
for i,j in itertools.product(range(20), range(20)):
    if i!=j:
        desired_corr[i, j] = 0.25

desired_corr = tf.constant(desired_corr, dtype=tf.float32)

mse = []
for h_0, h_1 in zip(self.hidden_state_hist[(400//par['dt']):(800//par['dt'])-1], self.hidden_state_hist[(400//par['dt'])+1:(800//par['dt'])]):
    u_0, v_0 = tf.nn.moments(h_0, axes=1)
    u_1, v_1 = tf.nn.moments(h_1, axes=1)
    cov = tf.matmul(h_0 - tf.tile(tf.reshape(u_0, (40,1)), (1,100)), tf.transpose(h_1 - tf.tile(tf.reshape(u_1, (40,1)), (1,100))))/(100*100)
    b = tf.matmul(tf.reshape(v_0, (40,1)), tf.reshape(v_1, (1,40)))
    mse.append(tf.pow((cov - (desired_corr * b)), 2))
