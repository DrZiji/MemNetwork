import sys
sys.path.append("d:\\MemNetwork")

from SiamFC import SiamOGEM

if __name__ == "__main__":
    
    SiamOGEM_ins = SiamOGEM(-1, 'TRAIN')
    SiamOGEM_ins.init_weights()