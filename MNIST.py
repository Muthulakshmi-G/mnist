import torch
from torch import nn
from torch.nn import functional as #softmax functionku
from torch.autograd import Variable
import glob
import matplotlib.pyplot as plt

from PIL import Image  #image arrayva matha python image L
import numpy as np
import random

class Mnist(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=100):
        super(Mnist, self).__init__()
        self.input_size = input_size           
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.linear0 = nn.Linear(self.input_size, self.hidden_size)#input,outputku lasta accuracy pathu hidden size add pannom
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size) 

    def forward(self, x):#x-gurathu input image nama  kodukurom
        x = F.tanh(self.linear0(x))#for non linearity we change this into tanh for linear0
        x = F.tanh(self.linear1(x))
        return F.log_softmax(self.linear(x))#namaku probabilty venum evlo correcta predicta pannuthunu so athanala inga softmax nu oru function use panrom .inga NLL use panrom

import sys
if __name__ == '__main__':

    if sys.argv[1] == 'train':#ithu train dataset ku

        dataset_rootdir = 'mnist_png'#folder name
        trainset_path = '{}/{}'.format(dataset_rootdir, 'training')#traindataset-a training vachukalam
        testset_path = '{}/{}'.format(dataset_rootdir, 'testing') #test dataseta testing nu vachukalam
#folder ,trainset,testset ku   la ennana erukunu print panni pakkalam
        print(dataset_rootdir)
        print(trainset_path)
        print(testset_path)
#train set oru list nu assume 
        trainset = []
        for d in glob.glob('{}/*'.format(trainset_path)):# trainset kulla ulla image and label ennanu pakkurom
            label = d.split('/')[-1]  #namaku input oda label venum so tuple la erunthu label a matum split panrom
            for f in glob.glob('{}/*'.format(d)): #labels-laye 0 t0 10 varaikum eruku entha image entha label nu theriyanum
                im_frame = Image.open(f) #inga imagea array va convert pannrom so now open the image
                np_frame = np.array(im_frame.getdata())  #image la ulla pixel values array va convert panrom by usin numpy
                trainset.append((np_frame, label))  #trainset la ulla ella imageskum image:label set panrom.   e.g   image0:0

        testset = []  
        for d in glob.glob('{}/*'.format(testset_path)):
            label = d.split('/')[-1]
            for f in glob.glob('{}/*'.format(d)):
                im_frame = Image.open(f)
                np_frame = np.array(im_frame.getdata())
                testset.append((np_frame, label))


        random.shuffle(trainset)# train set la ulla  ella imagesum  shuffle panrom

        trainset_image = [i[0] for i in trainset]  #trainsetla ethana images ethula erunthu start from 0 index  in list
        trainset_label = [i[1] for i in trainset]

        testset_image = [i[0] for i in testset]
        testset_label = [i[1] for i in testset]

        model = Mnist(784, 10)  # modelku kodukka vendiya input  784 and output 10

        loss = nn.NLLLoss()  #loss function NLL
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # optimizer stochastic gradient descent use panrom athodaa learning rate nama fix pannrom.accuracy poruthu adjust panrom

        for epoch in range(100):   #evlo times venum.sonatha epoch number set panrom 
            batch_size = 400  #nama dataset 70000 images eruku so ethula erunthu ethu varaikum 0,illa 1 nu theriyathu.so namaley batch pirikurom nama data ku yetha mari
            batch_num = int(len(trainset_image)/batch_size)  #batch num theriyanum so atha total training inagesla erunthu batch size divide pannauna no.of batches therium
            print('=================================')
            for index in range(batch_num): #  from index upto number of batches 

                image = np.array(trainset_image[index*batch_size: (index+1)*batch_size]) 
                label = [ int(l) for l in trainset_label[index*batch_size: (index+1)*batch_size]]

                image = Variable(torch.Tensor(image))  #tensor la  imagesa variablesa mathurom
                label = Variable(torch.LongTensor(label)) #label stringa eruku atha longa mathurom
   
                y_ = model(image)  #output imagela enna number erukunu kandu pudikanum
                l = loss(y_, label) #loss value


                optimizer.zero_grad() # optimizing
                l.backward() #backward nadakuthu
                optimizer.step()  #step by step


                if not index % batch_num/2: # ipo  e.g index 4/2= 2
                    print('--------------')
                    print('{}. index: {}, loss: {}'.format(epoch, index, l.data[0])) # entha batchla evlo loss nu pakkurom.so entha epochla entha indexla evlo loss 
                    test_loss = 0  #   test set oda loss
                    correct_count = 0   #accuracy kndupudika so ethana correcta predict panji erukunu pakkanum
                    for tindex in range(batch_num):   #no of batches la start index la erunthu
                        image = np.array(trainset_image[tindex*batch_size: (tindex+1)*batch_size])
                        label = [ int(l) for l in trainset_label[tindex*batch_size: (tindex+1)*batch_size]]
                        image = Variable(torch.Tensor(image))
                        label = Variable(torch.LongTensor(label))

                        y_ = model(image)
                        l = loss(y_, label)
                        test_loss += l.data[0]

                        correct_count += (label == y_.max(dim=1)[1]).sum().data[0]/batch_size # find out maximum between all accuracy values

                    print('{}. tindex: {}, loss: {}, accuracy: {}'.format(epoch, tindex, test_loss/batch_num, correct_count/batch_num))  # entha epoch la,indexla evlo correcta predict pannathu nu print panrom

        torch.save(model.state_dict(), 'muthu.pth')  # intha modela save panrom.ithaye vera ethum file ku use pannrathukaga save panni vaikurom.

    if sys.argv[1] == 'infer':  # input kodukiratha inference nu solrom.run pannum pothu file name kodukurom athukaga atha sys import panrom.
        model = Mnist(784, 10)  #same input uotput than kodukkanum inagaum
        model.load_state_dict(torch.load('muthu.pth'))  #antha kodukka pora input file inga load panrom by this syntax
        
        im_frame = Image.open(sys.argv[2])    #athula vendiya image open pannanum by this.
        np_frame = np.array(im_frame.getdata())  # antha imagea array va converta pannanum.

        image = Variable(torch.Tensor(np_frame))
        y_ = model(image)
        print('label: {}'.format( y_.view(1, -1).max(dim=-1)[1].data[0]))  #ipo nama oru image koduthutom athuku evlo probability nu pakkurom entha indexla athu athigam nu.ethu max eruko anthu than antha kodutha imagela ulla number with label oda nama atha prin panrom.



