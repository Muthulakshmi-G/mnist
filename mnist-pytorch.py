import torch
from torch import nn
from torch.nn import functional as 
from torch.autograd import Variable
import glob
import matplotlib.pyplot as plt

from PIL import Image  
import numpy as np
import random

class Mnist(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=100):
        super(Mnist, self).__init__()
        self.input_size = input_size           
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.linear0 = nn.Linear(self.input_size, self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size) 

    def forward(self, x):
        x = F.tanh(self.linear0(x))
        x = F.tanh(self.linear1(x))
        return F.log_softmax(self.linear(x))

import sys
if __name__ == '__main__':

    if sys.argv[1] == 'train':

        dataset_rootdir = 'mnist_png'
        trainset_path = '{}/{}'.format(dataset_rootdir, 'training')
        testset_path = '{}/{}'.format(dataset_rootdir, 'testing') 
        print(dataset_rootdir)
        print(trainset_path)
        print(testset_path)
 
        trainset = []
        for d in glob.glob('{}/*'.format(trainset_path)):
            label = d.split('/')[-1] 
            for f in glob.glob('{}/*'.format(d)): 
                im_frame = Image.open(f) 
                np_frame = np.array(im_frame.getdata())  
                trainset.append((np_frame, label))  

        testset = []  
        for d in glob.glob('{}/*'.format(testset_path)):
            label = d.split('/')[-1]
            for f in glob.glob('{}/*'.format(d)):
                im_frame = Image.open(f)
                np_frame = np.array(im_frame.getdata())
                testset.append((np_frame, label))


        random.shuffle(trainset)

        trainset_image = [i[0] for i in trainset]  
        trainset_label = [i[1] for i in trainset]

        testset_image = [i[0] for i in testset]
        testset_label = [i[1] for i in testset]

        model = Mnist(784, 10) 

        loss = nn.NLLLoss()  
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 

        for epoch in range(100):   
            batch_size = 400  
            batch_num = int(len(trainset_image)/batch_size)  
            print('=================================')
            for index in range(batch_num): 

                image = np.array(trainset_image[index*batch_size: (index+1)*batch_size]) 
                label = [ int(l) for l in trainset_label[index*batch_size: (index+1)*batch_size]]

                image = Variable(torch.Tensor(image))  
                label = Variable(torch.LongTensor(label)) 
   
                y_ = model(image)  
                l = loss(y_, label) 


                optimizer.zero_grad() 
                l.backward() 
                optimizer.step()  


                if not index % batch_num/2: # 
                    print('--------------')
                    print('{}. index: {}, loss: {}'.format(epoch, index, l.data[0])) 
                    test_loss = 0  
                    correct_count = 0   
                    for tindex in range(batch_num):   
                        image = np.array(trainset_image[tindex*batch_size: (tindex+1)*batch_size])
                        label = [ int(l) for l in trainset_label[tindex*batch_size: (tindex+1)*batch_size]]
                        image = Variable(torch.Tensor(image))
                        label = Variable(torch.LongTensor(label))

                        y_ = model(image)
                        l = loss(y_, label)
                        test_loss += l.data[0]

                        correct_count += (label == y_.max(dim=1)[1]).sum().data[0]/batch_size #

                    print('{}. tindex: {}, loss: {}, accuracy: {}'.format(epoch, tindex, test_loss/batch_num, correct_count/batch_num)) 

        torch.save(model.state_dict(), 'muthu.pth') 
 panni vaikurom.

    if sys.argv[1] == 'infer':  

        model = Mnist(784, 10)  

        model.load_state_dict(torch.load('muthu.pth'))  

        
        im_frame = Image.open(sys.argv[2])    
        np_frame = np.array(im_frame.getdata()) 
        image = Variable(torch.Tensor(np_frame))
        y_ = model(image)
        print('label: {}'.format( y_.view(1, -1).max(dim=-1)[1].data[0]))  
