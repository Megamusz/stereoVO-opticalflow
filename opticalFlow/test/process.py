import os

def del_files(path):
    for root , dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".png", ".yuv")):
                os.remove(os.path.join(root, name))
                
bin = r'..\build\Release\optflow.exe '

dir = "\"C:/Users/megamusz/Desktop/data_stereo_flow/training\""


                
del_files('./')

for i in range(0, 200):
    cmd = '{0} {1} {2}'.format(bin, dir, i)
    print(cmd)
    os.system(cmd)
    p = '{:06d}_10.png'.format(i)
    p_c = '{:06d}_10_c.png'.format(i)
    os.rename(p, 'pred/{0}'.format(p))
    os.rename(p_c, 'pred_correct/{0}'.format(p))
    

