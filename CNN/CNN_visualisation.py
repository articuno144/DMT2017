import matplotlib
import matplotlib.pyplot as plt
import time

def vis_best_gest():
    for j in range(8):
        plt.subplot(8,1,j+1)
        plt.plot(batch_x[m[j][0],:,3:])
        plt.axis([0,300,-1.2,1.2])
    plt.show()
        

def vis_cover_up():
    activations = np.zeros([8,280],dtype = float)
    for j in range(8):
        totest = np.zeros([1,300,6],dtype = float)
        totest[0,:,:] = batch_x[m[j][0],:,:]
        for cov in range(280):
            covered = totest.copy()
            covered[0,cov:cov+20,:] = np.zeros([1,20,6],dtype = float)
            activations[j,cov] = sess.run(s_pred,feed_dict = {x:covered,keep_prob:1.0})[0][j]
        plt.subplot(8,1,j+1)
        plt.plot(activations[j,:])
        plt.axis([0,300,0,1])
    plt.show()

def vis_best_gest_cov(width):
    activations = np.zeros([8,300-width],dtype = float)
    ctr=0
    for j in range(8):
        plt.subplot(2,8,ctr+1)
        plt.plot(batch_x[m[j][0],:,3:])
        plt.axis([0,300,-1.2,1.2])
        totest = np.zeros([1,300,6],dtype = float)
        totest[0,:,:] = batch_x[m[j][0],:,:]
        for cov in range(300-width):
            covered = totest.copy()
            covered[0,cov:cov+width,:] = np.zeros([1,width,6],dtype = float)
            activations[ctr,cov] = sess.run(pred,feed_dict = {x:covered,keep_prob:1.0})[0][j]
        plt.subplot(2,8,ctr+9)
        plt.plot(activations[ctr,:])
        plt.xlim(0,300)
        ctr+=1
    plt.show()

def test_speed():
    t_sum = 0
    for i in range(100):
        sample = np.zeros([1,300,6],dtype = float)
        sample[0,:,:] = batch_x[random.randint(0,512),:,:]
        t1 = time.time()
        p = sess.run(c_argmax,feed_dict={x:sample,keep_prob:1.0})
        t2 = time.time()
        t_sum+=t2-t1
    return t_sum/100

def vis_best_gest_cov_258(width):
    activations = np.zeros([3,300-width],dtype = float)
    ctr=0
    for j in [1,4,7]:
        plt.subplot(2,3,ctr+1)
        plt.plot(batch_x[m[j][0],:,3:])
        plt.axis([0,300,-1.2,1.2])
        totest = np.zeros([1,300,6],dtype = float)
        totest[0,:,:] = batch_x[m[j][0],:,:]
        for cov in range(300-width):
            covered = totest.copy()
            covered[0,cov:cov+width,:] = np.zeros([1,width,6],dtype = float)
            activations[ctr,cov] = sess.run(pred,feed_dict = {x:covered,keep_prob:1.0})[0][j]
        plt.subplot(2,3,ctr+4)
        plt.plot(activations[ctr,:])
        plt.xlim(0,300)
        ctr+=1
    plt.show()

def tell_diff(arr1, arr2):
    if len(arr1)==len(arr2):
        m = {}
        for i in range(len(arr1)):
            if arr1[i]!=arr2[i]:
                if (arr1[i],arr2[i]) in m.keys():
                    m[(arr1[i],arr2[i])]+=1
                else:
                    m[(arr1[i],arr2[i])]=1 
                print(arr1[i], " => ",arr2[i], i)
        print(m)

tell_diff(gest_d,gest_preds)
tell_diff(noise_d,noise_preds)
print("results look ok?")