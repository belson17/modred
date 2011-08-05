
"""Script which makes plots of scaling. To use this, must copy in the
scaling times from profiling benchmark.py with cProfile."""

import numpy as N
import matplotlib.pyplot as PLT

class Scaling(object):
    def __init__(self):
        pass

def lin_n1p_rainier():
    """Scaling of lin_combine on rainier, one node"""
    lin = [Scaling() for i in range(8)]
    lin[0].total = 359.863
    lin[0].loads = 247.751
    lin[0].addmult = 0.
    lin[0].sendrecvs = 0.
    
    lin[1].total = 192.978
    lin[1].loads = 127.519
    lin[1].addmult = 0.
    lin[1].sendrecvs = 4.464 + 4.164
    
    lin[2].total = 148.069
    lin[2].loads = 93.570
    lin[2].addmult = 0
    lin[2].sendrecvs = 7.566 + 5.773
    
    lin[3].total = 117.497
    lin[3].loads = 71.321
    lin[3].addmult = 0
    lin[3].sendrecvs = 8.933 +6.279
    
    lin[4].total = 98.331
    lin[4].loads = 57.303
    lin[4].addmult = 15.239
    lin[4].sendrecvs = 8.420 +6.194
    
    lin[5].total = 95.954
    lin[5].loads = 51.473
    lin[5].addmult = 0
    lin[5].sendrecvs = 13.398 + 7.144
    
    lin[6].total = 87.699
    lin[6].loads = 45.063
    lin[6].addmult = 0
    lin[6].sendrecvs = 11.610 +8.006
    
    lin[7].total = 89.385
    lin[7].loads = 42.535
    lin[7].addmult = 0
    lin[7].sendrecvs =16.788 + 8.418
    
    PLT.figure()
    width = .4
    numProcs = 8
    PLT.plot(N.arange(1,numProcs+1), lin[0].total/N.arange(1,numProcs+1),'k--')
    PLT.plot(1+N.arange(numProcs),[lin[i].total for i in range(numProcs)],'r--')
    for i in range(numProcs):
        bottom = 0
        PLT.bar(i+1, lin[i].sendrecvs,width=width,bottom=bottom,color='r')
        bottom += lin[i].sendrecvs
        #PLT.bar(i+1, lin[i].addmult,width=width,bottom=bottom,color='b')
        #bottom+=lin[i].addmult
        PLT.bar(i+1, lin[i].loads,width=width,bottom=bottom,color='g')
        bottom+=lin[i].loads
        PLT.bar(i+1, lin[i].total - bottom,width=width,bottom=bottom,color='k')
    PLT.legend(['linear','measured','send/recvs','loads','other'])
    PLT.xlabel('number of processors/node')
    PLT.ylabel('seconds per processor')
    PLT.savefig('scaling_lin_n1p_rainier.eps')
    PLT.show()    
    


def ip_n1p_rainier():
    """Scaling of ip_mat on rainier, one node"""
    ip = [Scaling() for i in range(8)]
    ip[0].total = 284.086
    ip[0].loads = 199.399
    ip[0].ips = 75.034
    ip[0].sendrecvs = 0.
    
    ip[1].total = 158.920
    ip[1].loads = 108.425
    ip[1].ips = 37.661
    ip[1].sendrecvs = 3.264 + 3.121
    
    ip[2].total = 119.216
    ip[2].loads = 75.131
    ip[2].ips = 27.425
    ip[2].sendrecvs = 8.646 + 6.346
    
    ip[3].total = 95.416
    ip[3].loads = 58.081
    ip[3].ips = 19.615
    ip[3].sendrecvs = 7.398 +5.031
    
    ip[4].total = 78.554
    ip[4].loads = 46.082
    ip[4].ips = 15.239
    ip[4].sendrecvs = 7.278 +5.143
    
    ip[5].total = 78.310
    ip[5].loads = 46.695
    ip[5].ips = 13.159
    ip[5].sendrecvs = 7.286 + 4.484
    
    ip[6].total = 69.798
    ip[6].loads = 37.203
    ip[6].ips = 12.181
    ip[6].sendrecvs = 8.518 + 6.627
    
    ip[7].total = 67.698
    ip[7].loads = 34.888
    ip[7].ips = 10.658
    ip[7].sendrecvs =10.069 + 6.906
    
    PLT.figure()
    width = .4
    numProcs = 8
    PLT.plot(N.arange(1,numProcs+1), ip[0].total/N.arange(1,numProcs+1),'k--')
    PLT.plot(1+N.arange(numProcs),[ip[i].total for i in range(numProcs)],'r--')
    for i in range(numProcs):
        bottom = 0
        PLT.bar(i+1, ip[i].sendrecvs,width=width,bottom=bottom,color='r')
        bottom += ip[i].sendrecvs
        PLT.bar(i+1, ip[i].ips,width=width,bottom=bottom,color='b')
        bottom+=ip[i].ips
        PLT.bar(i+1, ip[i].loads,width=width,bottom=bottom,color='g')
        bottom+=ip[i].loads
        PLT.bar(i+1, ip[i].total - bottom,width=width,bottom=bottom,color='k')
    PLT.legend(['linear','measured','send/recvs','IPs','loads','other'])
    PLT.xlabel('number of processors/node')
    PLT.ylabel('seconds per processor')
    PLT.savefig('scaling_ipmat_n1p_rainier.eps')
    PLT.show()
    

def ip_np1_della():
    """
    profiling data for one processor per node, varying # nodes, ip_mat
    """
    numNodes = 12
    ip = [Scaling() for i in xrange(numNodes)]
    ip[0].total = 240.206
    ip[0].loads = 164.500
    ip[0].ips = 65.605
    ip[0].sendrecvs = 0.
    
    ip[1].total = 96.103
    ip[1].loads = 46.685
    ip[1].ips = 34.112
    ip[1].sendrecvs = 2.033+7.991
    
    ip[2].total = 61.961
    ip[2].loads = 24.197
    ip[2].ips = 25.325
    ip[2].sendrecvs = 6.712+1.996
    
    ip[3].total = 44.644
    ip[3].loads = 13.834
    ip[3].ips = 17.896
    ip[3].sendrecvs = 8.471 + 1.647
    
    ip[4].total = 0
    ip[4].loads = 0
    ip[4].ips = 0
    ip[4].sendrecvs = 0.
    
    ip[5].total = 29.408
    ip[5].loads = 6.941
    ip[5].ips = 12.147
    ip[5].sendrecvs = 1.142 + 7.260
    
    ip[6].total = 0
    ip[6].loads = 0
    ip[6].ips = 0
    ip[6].sendrecvs = 0.
    
    ip[7].total = 24.200
    ip[7].loads = 5.062
    ip[7].ips = 8.619
    ip[7].sendrecvs = 8.026+0.934
    
    ip[8].total = 0
    ip[8].loads = 0
    ip[8].ips = 0
    ip[8].sendrecvs = 0.
    
    ip[9].total = 20.427
    ip[9].loads = 3.983
    ip[9].ips = 6.931
    ip[9].sendrecvs =7.376+0.769
    
    ip[10].total = 0
    ip[10].loads = 0
    ip[10].ips = 0
    ip[10].sendrecvs = 0.
    
    ip[11].total = 18.326
    ip[11].loads = 3.180
    ip[11].ips = 6.019
    ip[11].sendrecvs = 7.242+0.668
    
    PLT.figure()
    width = .4
    PLT.plot(N.arange(1,numNodes+1), ip[0].total/N.arange(1,numNodes+1),'k--')
    PLT.plot([1,2,3,4,6,8,10,12],[ip[0].total,ip[1].total,ip[2].total,ip[3].total,\
        ip[5].total,ip[7].total,ip[9].total,ip[11].total],'r--')

    for i in range(numNodes):
        bottom = 0
        PLT.bar(i+1, ip[i].sendrecvs,width=width,bottom=bottom,color='r')
        bottom += ip[i].sendrecvs
        PLT.bar(i+1, ip[i].ips,width=width,bottom=bottom,color='b')
        bottom+=ip[i].ips
        PLT.bar(i+1, ip[i].loads,width=width,bottom=bottom,color='g')
        bottom+=ip[i].loads
        PLT.bar(i+1, ip[i].total - bottom,width=width,bottom=bottom,color='k')
    PLT.legend(['linear','measured','send/recvs','IPs','loads','other'])
    PLT.xlabel('number of nodes (1 processor/node)')
    PLT.ylabel('seconds per node (1 node = 1 processor)')
    PLT.axis([1,13,0,250])
    PLT.savefig('scaling_ipmat_np1_della.eps')
    PLT.show()
    
    
    
    
if __name__=='__main__':
    ip_n1p_rainier()
    #ip_n1p_della()
    ip_np1_della()
    lin_n1p_rainier()