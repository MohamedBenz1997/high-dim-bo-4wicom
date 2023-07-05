import tensorflow as tf
from config import Config
import math
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


class User(Config):
    def __init__(self):
        Config.__init__(self)
        pass
    def call(self,dataobj,batch_ind,user_id):
        self.user_id =user_id
        self.p_los = dataobj.p_LOS[batch_ind,:,user_id].numpy()
        
        #
        self.pl = dataobj.pl[batch_ind,:,user_id].numpy()
        self.shadowing_LOS = dataobj.shadowing_LOS[batch_ind,:,user_id].numpy()
        self.shadowing_NLOS = dataobj.shadowing_NLOS[batch_ind,:,user_id].numpy()
        self.antenna_gain = dataobj.G_Antenna[batch_ind,:,user_id].numpy()
        #
        
        self.D  = dataobj.D[batch_ind,:,user_id].numpy()
        self.LSL = dataobj.LSL[batch_ind,:,user_id].numpy()
        # self.shadowing_NLOS = dataobj.LSGclass.shadowing_NLOS[batch_ind,:,user_id].numpy()
        # self.shadowing_LOS = dataobj.LSGclass.shadowing_LOS[batch_ind,:,user_id].numpy()
        # self.pl = dataobj.LSGclass.pl[batch_ind,:,user_id].numpy()
        # self.antenna_gain = dataobj.G_Antenna[batch_ind,:,user_id]
        return
    def print(self,dataobj):
        AP = [i for i in range (self.D.shape[0])]
        dic = {"AP":AP,"P_los":self.p_los,"Distance":self.D,"Large scale Loss (dB)":self.LSL, "Shadowing LOS":self.shadowing_LOS,
               "Shadowing none LOS":self.shadowing_NLOS, "Antenna gain":self.antenna_gain, "Path Loss":self.pl}  
        df = pd.DataFrame(dic)
        print(tabulate(df, headers='keys', tablefmt='psql',numalign="center",showindex=False))
        # print(df)
        # print("User prop "+str(self.user_id))
        # header = [6*" ", "P_los", "Distance","Large scale gain (dB)", "Shadowing none LO", "Shadowing LOS"]
        # l1 = [len(word) for word in header]
        # h=""
        # for word in header:
        #     h = h+word+" "
        # print(h)
        # space =" "
        # for i in range(dataobj.Nap):
        #     p_los = "{:.2f}".format(self.p_los[i])+
        #     D = "{:.2f}".format(self.D[i])
        #     LSL = "{:.2f}".format(self.LSL[i])
        #     shadowingNLS = "{:.2f}".format(self.shadowing_NLOS[i])
        #     shadowing_LOS = "{:.2f}".format(self.shadowing_LOS[i])
        #
        #     print("AP "+str(i)+7*" "++7*" "+str(self.D[i])+space+str(self.LSL[i])
        #           +space+str(self.shadowing_NLOS[i])+space+str(self.shadowing_LOS[i]))
        pass
# user = User()
# user.call(data)