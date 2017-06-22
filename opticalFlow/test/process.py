import os
bin = r'..\build\Release\optflow.exe '

dir = "\"C:/Users/megamusz/Desktop/data_stereo_flow/training\""

for i in range(0, 194):
    cmd = '{0} {1} {2}'.format(bin, dir, i)
    print(cmd)
    os.system(cmd)
    
