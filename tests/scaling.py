
"""Scrcasest which makes plots of scaling. To use this, must copy in the
scaling times from profiling benchmark.py with cProfile."""

import numpy as N
import matplotlib.pyplot as PLT

class Scaling(object):
    def __init__(self):
        pass

def lin():
    """Scaling of lin_combine on della, max procs/node (12)"""
    cases = []
    
    s = Scaling()
    s.total = 2711.155
    s.loads = 2616.031
    s.addmult = 0.
    s.sends = 0
    s.recvs = 0.
    s.barriers=0
    s.workers = 1
    cases.append(s)
       
    s = Scaling()
    s.total = 492.035
    s.loads = 249.827
    s.addmult = 0
    s.sends = 14.359
    s.recvs = 134.804
    s.barriers = 57.284
    s.workers = 6
    cases.append(s)
    
    s = Scaling()
    s.total = 183.618
    s.loads = 119.532
    s.addmult = 0
    s.sends = 12.215
    s.recvs = 16.438
    s.barriers = 12.973
    s.workers = 12
    cases.append(s)
    
    s = Scaling()
    s.total = 57.529
    s.loads = 24.666
    s.addmult = 0
    s.sends = 5.794
    s.recvs = 5.284
    s.barriers = 7.334
    s.workers = 24
    cases.append(s)
    
    s = Scaling()
    s.total = 30.213
    s.loads = 9.525
    s.addmult = 0
    s.sends = 3.859
    s.recvs = 5.831
    s.barriers = 4.921
    s.workers = 36
    cases.append(s)
    
    s = Scaling()
    s.total = 18.126
    s.loads = 3.888
    s.addmult = 0
    s.sends = 2.201
    s.recvs =3.875
    s.barriers = 3.928
    s.workers = 48
    cases.append(s)
    
    """
    s = Scaling()
    s.total = 18.126
    s.loads = 3.888
    s.addmult = 0
    s.sends = 2.201
    s.recvs =3.875
    s.barriers = 3.928
    s.workers = 60
    cases.append(s)
    
    s = Scaling()
    s.total = 18.126
    s.loads = 3.888
    s.addmult = 0
    s.sends = 2.201
    s.recvs =3.875
    s.barriers = 3.928
    s.workers = 72
    cases.append(s)
    
    s = Scaling()
    s.total = 18.126
    s.loads = 3.888
    s.addmult = 0
    s.sends = 2.201
    s.recvs =3.875
    s.barriers = 3.928
    s.workers = 84
    cases.append(s)
    
    s = Scaling()
    s.total = 18.126
    s.loads = 3.888
    s.addmult = 0
    s.sends = 2.201
    s.recvs =3.875
    s.barriers = 3.928
    s.workers = 96
    cases.append(s)
    """
    
    
    
    workers = N.array([c.workers for c in cases])
    
    # Speedup plot
    PLT.figure()
    width = .4
    PLT.hold(True)
    PLT.plot(workers, workers,'k-')
    PLT.plot(workers, [cases[0].total/c.total for c in cases],'ro-')
    PLT.xlabel('Workers')
    PLT.ylabel('Speedup')
    PLT.legend(['Linear','Measured'])
    PLT.savefig('lin_combine_speedup_n1.eps')
    
    # Time spent breakdown
    """
    PLT.figure()
    PLT.hold(True)
    PLT.plot(workers, cases[0].total/workers,'k-')
    PLT.plot(workers, [c.total for c in cases],'bx-')
    for c in cases:
        bottom = 0
        top = c.sends + c.recvs + c.barriers
        PLT.bar(c.workers-width/2, top-bottom, width=width,bottom=bottom,color='r')
        
        bottom = top
        top += c.loads
        PLT.bar(c.workers-width/2, top-bottom, width=width,bottom=bottom,color='g')
        
        bottom = top
        top = c.total
        PLT.bar(c.workers-width/2, top-bottom, width=width,bottom=bottom,color='k')
    PLT.legend(['Linear','Measured','Send/Recvs','Loads','Other'])
    PLT.xlabel('Workers')
    PLT.ylabel('Time [s]')
    PLT.savefig('lin_combine_time_n1.eps')
    """
    PLT.show()
    


def ips_n1p_rainier():
    """Scaling of cases_mat on rainier, one node"""
    cases = []
    
    s = Scaling()
    s.total = 284.086
    s.loads = 199.399
    s.ips = 75.034
    s.sendrecvs = 0.
    s.workers=1
    cases.append(s)
    
    s = Scaling()
    s.total = 158.920
    s.loads = 108.425
    s.ips = 37.661
    s.sendrecvs = 3.264 + 3.121
    s.workers=2
    cases.append(s)
    
    s = Scaling()
    s.total = 119.216
    s.loads = 75.131
    s.ips = 27.425
    s.sendrecvs = 8.646 + 6.346
    s.workers=3
    cases.append(s)
    
    s = Scaling()
    s.total = 95.416
    s.loads = 58.081
    s.ips = 19.615
    s.sendrecvs = 7.398 +5.031
    s.workers=4
    cases.append(s)
    
    s = Scaling()
    s.total = 78.554
    s.loads = 46.082
    s.ips = 15.239
    s.sendrecvs = 7.278 +5.143
    s.workers=5
    cases.append(s)
    
    s = Scaling()
    s.total = 78.310
    s.loads = 46.695
    s.ips = 13.159
    s.sendrecvs = 7.286 + 4.484
    s.workers=6
    cases.append(s)
    
    s = Scaling()
    s.total = 69.798
    s.loads = 37.203
    s.ips = 12.181
    s.sendrecvs = 8.518 + 6.627
    s.workers=7
    cases.append(s)
    
    s = Scaling()
    s.total = 67.698
    s.loads = 34.888
    s.ips = 10.658
    s.sendrecvs =10.069 + 6.906
    s.workers=8
    cases.append(s)
    
    workers = N.array([c.workers for c in cases])
    
    # Speedup plot
    PLT.figure()
    width = .4
    PLT.hold(True)
    PLT.plot(workers, workers, 'k-')
    PLT.plot(workers, cases[0].total/workers, 'ro-')
    PLT.xlabel('Workers')
    PLT.ylabel('Speedup')
    PLT.legend(['Linear', 'Measured'])
    PLT.savefig('IP_speedup_n1.eps')
    
    # Time spent breakdown
    PLT.figure()
    PLT.hold(True)
    PLT.plot(workers, cases[0].total/workers, 'k-')
    PLT.plot(workers, [c.total for c in cases], 'ro-')
    
    for c in cases:
        bottom = 0
        top = c.sendrecvs
        PLT.bar(c.workers - width/2, top - bottom, width=width,bottom=bottom,color='r')
        
        bottom = top
        top += c.loads
        PLT.bar(c.workers - width/2, top - bottom,width=width,bottom=bottom,color='g')
        
        bottom = top
        top += c.ips
        PLT.bar(c.workers - width/2, top - bottom,width=width,bottom=bottom,color='k')
        
        bottom = top
        top = c.total
        PLT.bar(c.workers - width/2, top - bottom,width=width,bottom=bottom,color='k')

    PLT.legend(['Linear','Measured','Send/Recvs','Loads','IPs','Other'])
    PLT.xlabel('Workers')
    PLT.ylabel('Time [s]')
    PLT.savefig('IP_time_n1.eps')
    PLT.show()
    

def ips_np1_della():
    """
    profiling data for one processor per node, varying # nodes, cases_mat
    """
    cases = []
    
    s = Scaling()
    s.total = 240.206
    s.loads = 164.500
    s.ips = 65.605
    s.sendrecvs = 0.
    s.workers = 1
    cases.append(s)
    
    s = Scaling()
    s.total = 96.103
    s.loads = 46.685
    s.ips = 34.112
    s.sendrecvs = 2.033+7.991
    s.workers = 2
    cases.append(s)
        
    s = Scaling()
    s.total = 61.961
    s.loads = 24.197
    s.ips = 25.325
    s.sendrecvs = 6.712+1.996
    s.workers = 3
    cases.append(s)
    
    s = Scaling()
    s.total = 44.644
    s.loads = 13.834
    s.ips = 17.896
    s.sendrecvs = 8.471 + 1.647
    s.workers = 4
    cases.append(s)
    
    s = Scaling()
    s.total = 29.408
    s.loads = 6.941
    s.ips = 12.147
    s.sendrecvs = 1.142 + 7.260
    s.workers = 6
    cases.append(s)
    
    s = Scaling()
    s.total = 24.200
    s.loads = 5.062
    s.ips = 8.619
    s.sendrecvs = 8.026+0.934
    s.workers = 8
    cases.append(s)
    
    s = Scaling()
    s.total = 20.427
    s.loads = 3.983
    s.ips = 6.931
    s.sendrecvs =7.376+0.769
    s.workers = 10
    cases.append(s)
    
    s = Scaling()
    s.total = 18.326
    s.loads = 3.180
    s.ips = 6.019
    s.sendrecvs = 7.242+0.668
    s.workers = 12
    cases.append(s)
    
    # Speedup plot
    PLT.figure()
    width = .4
    workers = N.array([c.workers for c in cases])
    PLT.hold(True)
    PLT.plot(workers, workers,'k-')
    PLT.plot(workers, [cases[0].total/c.total for c in cases],'ro-')
    PLT.xlabel('Workers')
    PLT.ylabel('Speedup')
    PLT.legend(['Linear','Measured'])
    PLT.savefig('IP_speedup_p1.eps')
    
    # Time spent breakdown
    PLT.figure()
    PLT.hold(True)
    PLT.grid(True)
    PLT.plot(workers, cases[0].total/workers,'k-')
    PLT.plot(workers, [c.total for c in cases],'ro-')
    
    for c in cases:
        bottom = 0
        top = c.sendrecvs
        PLT.bar(c.workers-width/2, top - bottom ,width=width,bottom=bottom,color='r')
        
        bottom = top
        top += c.loads       
        PLT.bar(c.workers-width/2, top-bottom,width=width,bottom=bottom,color='g')
        
        bottom = top
        top += c.ips
        PLT.bar(c.workers-width/2, top-bottom,width=width,bottom=bottom,color='k')
        
        bottom = top
        top = c.total
        PLT.bar(c.workers-width/2, top-bottom,width=width,bottom=bottom,color='k')

    PLT.legend(['Linear','Measured','Send/Recvs','Loads','IPs','Other'])
    PLT.xlabel('Workers')
    PLT.ylabel('Time [s]')
    PLT.savefig('IP_time_p1.eps')
    PLT.show()
    
    
    
    
if __name__=='__main__':
    #ips_n1p_rainier()
    #ips_np1_della()
    lin()
    