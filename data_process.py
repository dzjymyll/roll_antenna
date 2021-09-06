import os
import glob
import pandas as pd
import numpy as np


def read_data(data_path):
   data = np.empty([0,12])
   for sub_path in os.listdir(data_path):
      for f in glob.glob(os.path.join(data_path,sub_path,"*.xls")):
         dfs = pd.read_excel(f, header=None)
         dfs = np.array(dfs)
         data = np.concatenate((data, dfs), axis=0)
         #print(data.shape)
   #np.savetxt(os.path.join(data_path,"whole.csv"), data, delimiter=",")
   return data

if __name__ == "__main__":
   prj_path = r"C:\Users\yuqixiao2\Desktop\0_Phasecenter_data_Theta_plus-minus_90_degree"
   data = read_data(prj_path)
