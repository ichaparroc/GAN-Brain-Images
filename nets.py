import torch.nn as nn

from blocks import *

class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()
    
        self.layer1 = block(1, 64, activation = "LeakyReLU")
        self.layer2 = block(64, 128, normalize = True, activation = "LeakyReLU")
        self.layer3 = block(128, 256, normalize = True, activation = "LeakyReLU")
        self.layer4 = block(256, 512, normalize = True, activation = "LeakyReLU")
        self.layer5 = block(512, 1024, normalize = True, activation = "LeakyReLU")

        self.dlayer5 = block(1024, 512, transposed = True, normalize = True, activation = "ReLU", dropout=True)
        self.sa1 = self_attn(512)
        self.dlayer4 = block(1024, 256, transposed = True, normalize = True, activation = "ReLU", dropout=True)
        self.sa2 = self_attn(256)
        self.dlayer3 = block(512, 128, transposed = True, normalize = True, activation = "ReLU")
        self.sa3 = self_attn(128)
        self.dlayer2 = block(256, 64, transposed = True, normalize = True, activation = "ReLU")
        self.sa4 = self_attn(64)
        self.dlayer1 = block(128, 1, transposed = True, activation = "Tanh")

    #def forward(self, noise, x, label):                       #[(-1, 1, 128, 128), (-1, 16, 128, 128)]
    def forward(self, x):                       #[(-1, 1, 128, 128), (-1, 16, 128, 128)]
         #out1_noise = self.noise_layer1(noise)                 #[(-1, 16, 64, 64)
        #out1_x = self.x_layer1(x)                             #[(-1, 32, 64, 64)
        #out1_label = self.label_layer1(label)                 #[(-1, 16, 64, 64)
        
        #out1 = torch.cat([out1_noise, out1_x, out1_label], 1) #[(-1, 64, 64, 64)
        
        out1 = self.layer1(x)                                 #[(-1, 128, 64, 64)
        out2 = self.layer2(out1)                              #[(-1, 128, 32, 32)
        out3 = self.layer3(out2)                              #[(-1, 256, 16, 16)
        out4 = self.layer4(out3)                              #[(-1, 512, 8, 8)
        out5 = self.layer5(out4)                              #[(-1, 1024, 4, 4)
        
        dout5 = self.dlayer5(out5)                            #[(-1, 512, 8, 8)
        dout5 = self.sa1(dout5)                              #[(-1, 512, 8, 8)
        dout5_out4 = torch.cat([dout5, out4], 1)
        
        dout4 = self.dlayer4(dout5_out4)
        dout4 = self.sa2(dout4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        
        dout3 = self.dlayer3(dout4_out3)
        dout3 = self.sa3(dout3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        
        dout2 = self.dlayer2(dout3_out2)
        dout2 = self.sa4(dout2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        
        dout1 = self.dlayer1(dout2_out1)
    
        return dout1

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator,self).__init__()
      
        self.layer1 = block(1, 64, activation = "LeakyReLU")
        self.sa1 = self_attn(64)
        self.layer2 = block(64, 128, normalize = True, activation = "LeakyReLU")
        self.sa2 = self_attn(128)
        self.layer3 = block(128, 256, normalize = True, activation = "LeakyReLU")
        self.sa3 = self_attn(256)
        self.layer4 = block(256, 512, normalize = True, activation = "LeakyReLU")
        self.sa4 = self_attn(512)
        self.layer5 = block(512, 1024, normalize = True, activation = "LeakyReLU")
        self.sa5 = self_attn(1024)
        
    def forward(self, x): #torch.Size([128, 3, 128, 128])
        batchsize = x.size()[0]
        
        out1 = self.layer1(x) #torch.Size([128, 64, 64, 64])
        out1 = self.sa1(out1)
        
        out2 = self.layer2(out1) #torch.Size([128, 128, 32, 32])
        out2 = self.sa2(out2)
        
        out3 = self.layer3(out2) #torch.Size([128, 256, 16, 16])
        out3 = self.sa3(out3)
                
        out4 = self.layer4(out3) #torch.Size([128, 512, 8, 8])
        out4 = self.sa4(out4)
        
        out5 = self.layer5(out4) #torch.Size([128, 512, 4, 4])
        out5 = self.sa5(out5)
        
        output = torch.cat((1*x.view(batchsize,-1),
                            1*out1.view(batchsize,-1), 
                            2*out2.view(batchsize,-1), 
                            2*out3.view(batchsize,-1), 
                            2*out4.view(batchsize,-1),
                            4*out5.view(batchsize,-1)),1)

        return output