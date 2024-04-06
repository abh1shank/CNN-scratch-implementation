import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist

class ConvOp:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

    def image_reg(self, image):
        h, w = image.shape
        self.image=image
        for i in range(0, h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                image_patch = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield image_patch, i, j

    def forward_prop(self, image):
        h, w = image.shape
        conv_out = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for image_patch, i, j in self.image_reg(image):
            conv_out[i, j] = np.sum(image_patch * self.conv_filter, axis=(1, 2))
        return conv_out

    def back_prop(self, dL_dout, learning_rate):
        dL_df_params = np.zeros(self.conv_filter.shape)
        for image_patch, i, j in self.image_reg(self.image):
            for k in range(self.num_filters):
                dL_df_params[k] += image_patch * dL_dout[i, j, k]
        self.conv_filter -= learning_rate * dL_df_params
        return dL_df_params


class maxpool:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def image_reg(self, image):
        new_h = image.shape[0] // self.filter_size
        new_w = image.shape[1] // self.filter_size
        self.image = image

        for i in range(new_h):
            for j in range(new_w):
                image_patch = image[(i * self.filter_size):(i * self.filter_size + self.filter_size),
                              (j * self.filter_size):(j * self.filter_size + self.filter_size)]
                yield image_patch, i, j

    def forward_prop(self, image):
        h, w, num_filters = image.shape
        output = np.zeros((h // self.filter_size, w // self.filter_size, num_filters))
        for image_patch, i, j in self.image_reg(image):
            output[i, j] = np.amax(image_patch, axis=(0, 1))
        return output

    def back_prop(self, dl_do):
        dL_dmaxpool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_reg(self.image):
            h, w, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch, axis=(0, 1))

            for i1 in range(h):
                for j1 in range(w):
                    for k1 in range(num_filters):
                        if image_patch[i1, j1, k1] == maximum_val[k1]:
                            dL_dmaxpool[i * self.filter_size + i1, j * self.filter_size + j1, k1] = dl_do[i, j, k1]
        return dL_dmaxpool


class softmax:
    def __init__(self, ip_node, sm_node):
        self.wt = np.random.randn(ip_node, sm_node) / ip_node
        self.bias = np.zeros(sm_node)

    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        image_mod = image.flatten()
        self.modified_ip = image_mod
        op_val = np.dot(image_mod, self.wt) + self.bias
        self.out = op_val
        exp_out = np.exp(op_val)
        return exp_out / np.sum(exp_out, axis=0)

    def backprop(self, dl_do, learning_rate):
        for i, grad in enumerate(dl_do):
            if grad == 0:
                continue
            trans_eq = np.exp(self.out)
            s_tot = np.sum(trans_eq)
            dy_dz = -trans_eq[i] * trans_eq / (s_tot ** 2)
            dy_dz[i] = trans_eq[i] * (s_tot - trans_eq[i]) / (s_tot ** 2)
            dz_dw = self.modified_ip
            dz_db = 1
            dz_d_ip = self.wt
            dl_dz = grad * dy_dz
            dl_dw = dz_dw[np.newaxis].T @ dl_dz[np.newaxis]
            dl_db = dl_dz * dz_db
            dl_d_ip = dz_d_ip @ dl_dz

        self.wt -= learning_rate * dl_dw
        self.bias -= learning_rate * dl_db

        return dl_d_ip.reshape(self.orig_im_shape)


#img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
#conn = ConvOp(18, 7)
#img.shape
#out = conn.forward_prop(img)
#print(out.shape)
#plt.imshow(out[:, :, 17], cmap='gray')
#plt.show()

#conn2=maxpool(4)
#out2=conn2.forward_prop(out)
#print(out2.shape)
#plt.imshow(out2[:,:,17],cmap='gray')
#plt.show()

#conn3=Softmax(83*84*18,10)
#out3=conn3.forward_prop(out2)
#print(out3)

(x_train,y_train),(x_test,y_test)=mnist.load_data()
train_images=x_train[:1500]
test_images=x_test[:1500]
train_labels=y_train[:1500]
test_labels=y_test[:1500]

conv=ConvOp(8,3)
pool=maxpool(2)
softmax=softmax(13*13*8,10)

def cnn_forward_prop(image,label):
    op=conv.forward_prop((image)/255)-0.5
    op=pool.forward_prop(op)
    op=softmax.forward_prop(op)

    cross_entropy_loss=-np.log(op[label])
    accuracy_eval=1 if np.argmax(op)==label else 0
    return op,cross_entropy_loss,accuracy_eval

def training_cnn(image,label,lr=0.005):
    out,loss,acc=cnn_forward_prop(image,label)
    grad=np.zeros(10)
    grad[label]=-1/out[label]

    grad_back=softmax.backprop(grad,lr)
    grad_back=pool.back_prop(grad_back)
    grad_back=conv.back_prop(grad_back,lr)

    return loss,acc

for epoch in range(4):
    print(f"epoch no. {epoch+1}")
    shuffle_data=np.random.permutation(len(train_images))
    train_images=train_images[shuffle_data]
    train_labels=train_labels[shuffle_data]

    loss=0
    nc=0
    for i,(im,label) in enumerate(zip(train_images,train_labels)):
        if i%100==0:
            print(f"steps: {i}, avg_loss: {loss/100}, acc={nc}")
            loss=0
            nc=0
        l1,accu=training_cnn(im,label)
        loss+=l1
        nc+=accu
print("testing")
loss=0
nc=0
for im,label in zip(test_images,test_labels):
    _,ll,accu=cnn_forward_prop(im,label)
    loss+=ll
    nc+=accu
print("test_loss",loss/len(test_labels))
print("test_accuracy",nc/len(test_labels))