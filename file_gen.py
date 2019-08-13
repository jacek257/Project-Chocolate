from random import seed
from random import randint

seed()

path = '/home/asophonsri/Data/nomeclature/'

for i in range(100):
    file_name = 'SH_'+"{0:0=3d}".format(randint(0,999))+'_'+"{0:0=2d}".format(randint(0,12))+"{0:0=2d}".format(randint(0,31))+"{0:0=4d}".format(randint(0,9999))+".txt"
    with open(path+file_name,'w') as f:
        print('Created:', file_name)
