import tensorflow as tf
import scipy.io as scipy_io
import pdb
import h5py
import numpy as np
import os 
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './summary',
                            """dir to store trained net""")
beta = 0.01 #coefficients for regularizer penulty  
batch_size = 100


"""loading parameters from pretrained MLP """
full_connect_w = []
full_connect_b = []
file = scipy_io.loadmat('param.mat')
full_connect_mask_w = []
full_connect_mask_b = []
for layer_id_ in range(3):
    with tf.variable_scope("full_connect_{}".format(layer_id_)):
	init_w = file['full_{}_w'.format(layer_id_+1)].T
        init_b = file['full_{}_b'.format(layer_id_+1)].ravel()
        full_connect_w.append(tf.get_variable("matrix",initializer=init_w))
        full_connect_b.append(tf.get_variable("bias",initializer=init_b))

# define a tensorflow session
sess = tf.Session()

# set a placeholder for future input
input_ =tf.placeholder(tf.float32, shape=[None,2048])
label_ =tf.placeholder(tf.float32, shape=[None,2])

""" connect the tensorflow graph"""
preds = input_
for layer_id in range(3):
    preds = tf.matmul(preds,full_connect_w[layer_id])+full_connect_b[layer_id]
    if layer_id == 2:
        preds = tf.nn.softmax(preds)
    else:
        preds = tf.nn.relu(preds)
output = preds

""" define losses """
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output,label_))
loss_summary = [tf.summary.scalar('loss_entropy', loss)]
hist_summary = []
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(label_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summary = tf.summary.merge([tf.summary.scalar("accuaracy",accuracy)])

""" adding summary""" 
for layer_id in range(3):
   regularizer = (tf.nn.l2_loss(full_connect_w[layer_id])+tf.nn.l2_loss(full_connect_b[layer_id]))*beta
   loss_summary.append(tf.summary.scalar("loss_layer_{}".format(layer_id),regularizer))
   loss += regularizer 
   histogram_name = "histogram:"+str(layer_id)
   hist_summary.append(tf.summary.histogram(histogram_name,tf.reshape(full_connect_w[layer_id],[-1])))
	
summary = tf.summary.merge(loss_summary)
hist_sum = tf.summary.merge(hist_summary)

"""optimizer """
temp_op = tf.train.AdamOptimizer(.0001)
variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
gvs = temp_op.compute_gradients(loss,var_list=variable_collection)
train_op = temp_op.apply_gradients(gvs)
""" initialization""" 
init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver(tf.all_variables())
#saver.restore(sess,"/scratch/ys1297/model_pruning/interm/summary/model.ckpt-9000")


""" running training """
f_test = h5py.File('test.h5','r')
f_train = h5py.File('train.h5','r')
graph_def = sess.graph.as_graph_def(add_shapes=True)
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)

counter= 0 # set counter back to start of the file
epoch = 1 # set epoch number
for i in range(10000):
    """ training """
    if (counter+1)*batch_size <=f_train['data'].shape[0]:
    
        feed_dict = {input_:f_train['data'][counter*batch_size:(counter+1)*batch_size].reshape(batch_size,-1),label_:f_train['label'][counter*batch_size:(counter+1)*batch_size].reshape(batch_size,-1)}
        _,loss_,summary_ = sess.run([train_op,loss,summary],feed_dict = feed_dict)
        summary_writer.add_summary(summary_,i)
	counter += 1
    else: 
	counter=0
	feed_dict = {input_:f_train['data'][counter*batch_size:(counter+1)*batch_size].reshape(batch_size,-1),label_:f_train['label'][counter*batch_size:(counter+1)*batch_size].reshape(batch_size,-1)}
        _,loss_,summary_ = sess.run([train_op,loss,summary],feed_dict = feed_dict)
        summary_writer.add_summary(summary_,i)
	print " epoch :"+str(epoch)
	epoch +=1

    if i%100==0:
	"""testing  """
	feed_dict = {input_:f_test['data'][0:batch_size].reshape(batch_size,-1),label_:f_test['label'][0:batch_size].reshape(batch_size,-1)}
	auc,test_summary_,hist_sum_ = sess.run([accuracy,accuracy_summary,hist_sum],feed_dict = feed_dict)
	summary_writer.add_summary(test_summary_,i)
	summary_writer.add_summary(hist_sum_,i)
	print "accuracy: "+str(auc)
    if i %9000==0:
	checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=i)

""" perfom thresholding and generating mask  """
mask_w =[]
mask_b = []
for layer_id in range(3):
    """ masking weight matrix """
    w = sess.run([full_connect_w[layer_id]])[0]
    sort_w = np.sort(w.ravel())
    min_w = sort_w[np.floor(sort_w.size*0.05)]
    max_w = sort_w[np.floor(sort_w.size*0.95)]
    w_ = np.ones(w.shape)
    w_[np.multiply(w<max_w,w>min_w)]=0	
    mask_w.append(w_)
    """ masking bias matrix"""
    b = sess.run([full_connect_b[layer_id]])[0]
    sort_b = np.sort(b.ravel())
    min_b = sort_b[np.floor(sort_b.size*0.25)]
    max_b = sort_b[np.floor(sort_b.size*0.75)]
    b_ = np.ones(b.shape)
    b_[np.multiply(b<max_b,b>min_b)]=0   
    mask_b.append(b_)


scipy_io.savemat('mask.mat',{'mask_1_w':mask_w[0],'mask_2_w':mask_w[1],'mask_3_w':mask_w[2],'mask_1_b':mask_b[0],'mask_2_b':mask_b[1],'mask_3_b':mask_b[2]})
