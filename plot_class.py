"""
Created on Thur Aug 19 2021
@author: nikbakht

    This is the class of plotting the calculated SINR values.
    
"""

import matplotlib.pyplot as plt
import numpy as np
from config import Config
import tensorflow as tf
import pandas as pd
from tabulate import tabulate
import os
class Plot(Config):
    
    def __init__(self,**kwargs):
        super(Plot, self).__init__(**kwargs)
        Config.__init__(self)
        # self.Nap = Nap
        # self.Nuser = Nuser
        
    def cdfplot(self, x,xlabel,lines_label="",save=False,plot_name=""):
        fig, ax = plt.subplots(1, 1)
        cdf_5 =[]
        cdf_50=[]
        cdf_95 =[]
        for i in range(len(x)):
            qe, pe = self.ecdf((x[i].flatten()))
            ax.plot(qe, pe, lw=2, label=str(i))
            cdf_5.append(qe[np.argmin(np.abs( pe-0.05))])
            cdf_50.append(qe[np.argmin(np.abs(pe - 0.5))])
            cdf_95.append(qe[np.argmin(np.abs(pe - 0.95))])
        # line = [i for i in range(len(x))]
        dic ={"line":lines_label,"CDF 5%":cdf_5,"CDF 50%":cdf_50,"CDF 95%":cdf_95}
        df = pd.DataFrame(dic)
        print(tabulate(df, headers='keys', tablefmt='psql', numalign="center", showindex=False))
        # ax.hold(True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('CDF')
        ax.legend(lines_label)
        # ax.legend(fancybox=True, loc='right')

        #    plt.xlim([-10,30])
        plt.ylim([0, 1])
        if save:
            dir = os.getcwd()
            plt.savefig(dir + "/results/"+plot_name+".pdf", bbox_inches='tight')
        plt.show()

        
    def ecdf(self,sample):
        # convert sample to a numpy array, if it isn't already
        sample = np.atleast_1d(sample)
        # find the unique values and their corresponding counts
        quantiles, counts = np.unique(sample, return_counts=True)
    
        # take the cumulative sum of the counts and divide by the sample size to
        # get the cumulative probabilities between 0 and 1
        cumprob = np.cumsum(counts).astype(np.double) / sample.size
        return quantiles, cumprob
    def plot_ap_user(self,dataobj):
        # Xap = tf.gather(Xap,assigned_batch_index,axis=0)
        # Xuser = tf.gather(Xuser,assigned_batch_index,axis=0)
        # Xuser_assigned = tf.expand_dims(Xuser,axis=1)*tf.expand_dims(AP_assign_user,axis=3)
        # Xuser_assigned = tf.reduce_sum(Xuser_assigned,axis=2)
        Xap = dataobj.Xap
        Xuser_assigned = dataobj.Xuser
        fig, ax = plt.subplots(1, 1)
        for i in range(dataobj.Nap):
            if i==0:
                ax.plot(Xap[0, i, 0], Xap[0, i, 1], 'x', color='brown', label="BS")
                # ax.text(Xap[0,i,0],Xap[0,i,1],str(i), color='brown',label="BS")
                ax.plot(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], 'o', color='blue', label="User")
                ax.text(Xuser_assigned[0,i,0], Xuser_assigned[0, i, 1], str(i), color='blue', label="User")
            else:
                ax.plot(Xap[0,i,0],Xap[0,i,1],'x', color='brown')
                # ax.text(Xap[0, i, 0], Xap[0, i, 1], str(i), color='brown')
                ax.plot(Xuser_assigned[0,i,0], Xuser_assigned[0, i, 1], 'o', color='blue')
                ax.text(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], str(i), color='blue')
            ax.plot([Xuser_assigned[0,i,0], Xap[0,i,0]],[Xuser_assigned[0, i, 1],Xap[0,i,1]],color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # plt.xlim([0,self.EX])
        # plt.ylim([0,self.EY])
        ax.legend()

        plt.show()
        self.Xuser_assigned =Xuser_assigned
        return
    # def dist_serving_BS(self,dataobj):
    #     # Xap = tf.gather(Xap,assigned_batch_index,axis=0)
    #     # Xuser = tf.gather(Xuser,assigned_batch_index,axis=0)
    #     # Xuser_assigned = tf.expand_dims(Xuser,axis=1)*tf.expand_dims(AP_assign_user,axis=3)
    #     # Xuser_assigned = tf.reduce_sum(Xuser_assigned,axis=2)
    #     Xap = dataobj.Xap
    #     Xuser = dataobj.Xuser
    #     # if self.DeploymentType =="PPP":
    #     #     D, D_2d, BS_wrapped_Cord = dataobj.Deployment.Deploy.Dist(Xap,Xuser, self.EX, self.EY)
    #     # elif self.DeploymentType =="Hex":
    #     #     D, D_2d, BS_wrapped_Cord = dataobj.Deployment.Deploy.Dist(Xap, Xuser)
    #     self.cdfplot([tf.linalg.diag_part(D_2d).numpy()[:]],xlabel="Distance(m)")
        

