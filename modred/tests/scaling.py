"""Makes plots of parallel scaling. 

To use this, must copy in the
scaling times from profiling benchmark.py with cProfile."""
from __future__ import division
from __future__ import print_function
from future.builtins import object

import numpy as np
import matplotlib.pyplot as PLT


class Scaling(object):
    """Struct for each case"""
    def __init__(self):
        self.total = 0
        self.total = 0
        self.loads = 0
        self.addmult = 0
        self.sends = 0
        self.recvs = 0
        self.barriers = 0
        self.workers = 0


def lin():
    """Scaling of lin_combine on lonestar, max procs/node (12)"""
    cases = []
    """
    s = Scaling()
    s.total = 69840.919
    s.loads = 68547.300
    s.addmult = 0.
    s.sends = 0
    s.recvs = 0.
    s.barriers=0
    s.workers = 1
    cases.append(s)
       
    s = Scaling()
    s.total = 71956.340
    s.loads = 54121.326
    s.addmult = 0
    s.sends = 1868.101
    s.recvs = 8130.131
    s.barriers = 2711.582
    s.workers = 6
    cases.append(s)
    """
    s = Scaling()
    s.total = 65101.676
    s.loads = 37823.560
    s.addmult = 0
    s.sends = 3100.815
    s.recvs = 11785.722
    s.barriers = 5549.159
    s.workers = 12
    cases.append(s)
    
    s = Scaling()
    s.total = 38372.495
    s.loads = 17996.821
    s.addmult = 0
    s.sends = 3034.603
    s.recvs = 7725.519
    s.barriers = 5077.610
    s.workers = 24
    cases.append(s)
    
    s = Scaling()
    s.total = 42727.516
    s.loads = 10257.985
    s.addmult = 0
    s.sends = 2856.050
    s.recvs = 5032.965
    s.barriers = 7292.715
    s.workers = 36
    cases.append(s)
    
    s = Scaling()
    s.total = 21455.413
    s.loads = 7398.958
    s.addmult = 0
    s.sends = 2785.200
    s.recvs = 4511.474
    s.barriers = 4088.371
    s.workers = 48
    cases.append(s)
    
    s = Scaling()
    s.total = 18999.739
    s.loads = 5840.044
    s.addmult = 0
    s.sends = 2747.765
    s.recvs = 4150.445
    s.barriers = 3648.425
    s.workers = 60
    cases.append(s)
    
    s = Scaling()
    s.total = 18481.499
    s.loads = 4961.553
    s.addmult = 0
    s.sends = 2705.876
    s.recvs = 4026.794
    s.barriers = 4266.667
    s.workers = 72
    cases.append(s)
    
    s = Scaling()
    s.total = 19403.340
    s.loads = 4522.620
    s.addmult = 0
    s.sends = 2725.450
    s.recvs = 4188.932
    s.barriers = 5296.319
    s.workers = 84
    cases.append(s)
    
    s = Scaling()
    s.total = 20059.660
    s.loads = 3844.509
    s.addmult = 0
    s.sends = 2749.347
    s.recvs = 4335.871
    s.barriers = 6554.987
    s.workers = 96
    cases.append(s)
    
    s = Scaling()
    s.total = 17326.856
    s.loads = 2933.798
    s.addmult = 0
    s.sends = 2686.343
    s.recvs = 4316.174
    s.barriers = 4922.009
    s.workers = 144
    cases.append(s)
    
    s = Scaling()
    s.total = 17125.556
    s.loads = 2454.333
    s.addmult = 0
    s.sends = 2679.164
    s.recvs = 4312.584
    s.barriers = 5155.768
    s.workers = 192
    cases.append(s)
    
    s = Scaling()
    s.total = 17159.692
    s.loads = 2229.482
    s.addmult = 0
    s.sends = 2717.322
    s.recvs = 4291.430
    s.barriers = 5294.237
    s.workers = 240
    cases.append(s)

    s = Scaling()
    s.total = 18944.590
    s.loads = 2543.070
    s.addmult = 0
    s.sends = 2729.350
    s.recvs = 4242.178
    s.barriers = 6553.161
    s.workers = 288
    cases.append(s)
    
    workers = np.array([c.workers for c in cases])
    
    # Find the average for each processor, so now in Wall time instead of 
    # CPU time.
    for s in cases:
        s.total /= s.workers
        s.loads /= s.workers
        s.addmult /= s.workers
        s.sends /= s.workers
        s.recvs /= s.workers
        s.barriers /= s.workers
    
    # Speedup plot
    PLT.figure(figsize=(5.5, 4))
    width = .4
    PLT.hold(True)
    #PLT.plot(workers, workers,'k-')
    PLT.plot(workers, [cases[0].total*cases[0].workers/c.total for c in cases], 
        'ro-')
    PLT.xlabel('Workers')
    PLT.ylabel('Speedup')
    PLT.grid(True)
    #PLT.legend(['Linear','Measured'], loc='upper left')
    PLT.savefig('lin_combine_speedup.eps')
    
    # Table of time spent in each operation for diff num of workers
    print('Workers |   Total Wall   |        Loads       |' +\
        '      Send-recvs     |   Barriers')
    for s in cases:
        print('  %d    |  %.1f   | %.1f (%f) | %.1f (%f) | %.1f (%f)'%(
            s.workers, s.total, s.loads, s.loads/s.total, (s.sends+s.recvs), 
            (s.sends+s.recvs)/s.total, s.barriers, s.barriers/s.total))
    
    # Time spent breakdown
    """
    PLT.figure()
    PLT.hold(True)
    PLT.plot(workers, cases[0].total/workers,'k-')
    PLT.plot(workers, [c.total for c in cases],'bx-')
    for c in cases:
        bottom = 0
        top = c.sends + c.recvs + c.barriers
        PLT.bar(c.workers-width/2, top-bottom, width=width, 
            bottom=bottom,color='r')
        
        bottom = top
        top += c.loads
        PLT.bar(c.workers-width/2, top-bottom, width=width, 
            bottom=bottom,color='g')
        
        bottom = top
        top = c.total
        PLT.bar(c.workers-width/2, top-bottom, width=width, 
            bottom=bottom,color='k')
    PLT.legend(['Linear', 'Measured', 'Send/Recvs', 'Loads', 'Other'])
    PLT.xlabel('Workers')
    PLT.ylabel('Time [s]')
    PLT.savefig('lin_combine_time_n1.eps')
    """
    PLT.show()

def ips():
    """
    profiling data for IP mats
    """
    cases = []
    """
    s = Scaling()
    s.total = 15194.852
    s.loads = 12486.497
    s.ips = 2687.022
    s.sendrecvs = 0
    s.barriers = 0
    s.workers = 1
    cases.append(s)
    """
    s = Scaling()
    s.total = 46272.404
    s.loads = 16311.796
    s.ips = 17312.300
    s.sendrecvs = 4993.506 + 3889.390
    s.barriers = 3123.360
    s.workers = 12
    cases.append(s)
    
    s = Scaling()
    s.total = 35854.704
    s.loads = 7995.548
    s.ips = 16565.236
    s.sendrecvs = 4586.795 + 3946.941
    s.barriers = 2382.156
    s.workers = 24
    cases.append(s)
    
    s = Scaling()
    s.total = 32539.035
    s.loads = 5385.914
    s.ips = 16227.939
    s.sendrecvs = 4516.381 + 3813.655
    s.barriers = 2219.718
    s.workers = 36
    cases.append(s)
    
    s = Scaling()
    s.total = 32570.517
    s.loads = 3902.635
    s.ips = 16296.631
    s.sendrecvs = 4915.159 + 4248.279
    s.barriers = 2830.686
    s.workers = 60
    cases.append(s)
    
    s = Scaling()
    s.total = 32557.001
    s.loads = 2548.582
    s.ips = 16123.066
    s.sendrecvs = 5007.735 + 4382.075
    s.barriers = 4054.294
    s.workers = 96
    cases.append(s)
    
    s = Scaling()
    s.total = 32474.814
    s.loads = 1902.123
    s.ips = 16164.761
    s.sendrecvs = 5178.796 + 4453.750
    s.barriers = 4278.013
    s.workers = 144
    cases.append(s)
    
    s = Scaling()
    s.total = 36558.071
    s.loads = 1713.744
    s.ips = 16464.743
    s.sendrecvs = 6015.638 + 4499.427
    s.barriers = 7221.375
    s.workers = 192
    cases.append(s)
    
    s = Scaling()
    s.total = 33431.981
    s.loads = 1267.041
    s.ips = 16070.562
    s.sendrecvs = 5272.887+4476.664
    s.barriers = 5608.543
    s.workers = 288
    cases.append(s)
    
    workers = np.array([c.workers for c in cases])
    
    # Find the average for each processor, so now in Wall time instead of 
    # CPU time.
    for s in cases:
        s.total /= s.workers
        s.loads /= s.workers
        s.ips /= s.workers
        s.sendrecvs /= s.workers
        s.barriers /= s.workers
    
    # Speedup plot
    PLT.figure(figsize=(5.5,4))
    width = .4
    PLT.hold(True)
    #PLT.plot(workers, workers,'k-')
    PLT.plot(
        workers, [cases[0].total*cases[0].workers/c.total for c in cases],'ro-')
    PLT.xlabel('Workers')
    PLT.ylabel('Speedup')
    PLT.grid(True)
    #PLT.legend(['Linear','Measured'], loc='upper left')
    PLT.savefig('IPs_speedup.eps')
    
    # Table of time spent in each operation for diff num of workers
    print('Workers |   Total Wall   |        Loads       |' +\
        '     IPs     |      sends+recvs     |   barriers')
    for s in cases:
        print(('  %d    |  %.1f   | %.1f (%.2f) | %.1f (%.2f) | %.1f (%.2f) |'+
            '%.1f (%.2f)')%(
            s.workers, s.total, s.loads, s.loads/s.total, s.ips, s.ips/s.total,
            s.sendrecvs, s.sendrecvs/s.total, s.barriers, s.barriers/s.total))
    
    # Time spent breakdown
    """
    PLT.figure()
    PLT.hold(True)
    PLT.plot(workers, cases[0].total/workers,'k-')
    PLT.plot(workers, [c.total for c in cases],'bx-')
    for c in cases:
        bottom = 0
        top = c.sends + c.recvs + c.barriers
        PLT.bar(
            c.workers-width/2, top-bottom, width=width,bottom=bottom,color='r')
        
        bottom = top
        top += c.loads
        PLT.bar(
            c.workers-width/2, top-bottom, width=width,bottom=bottom,color='g')
        
        bottom = top
        top = c.total
        PLT.bar(
            c.workers-width/2, top-bottom, width=width,bottom=bottom,color='k')
    PLT.legend(['Linear','Measured','Send/Recvs','Loads','Other'])
    PLT.xlabel('Workers')
    PLT.ylabel('Time [s]')
    PLT.savefig('lin_combine_time_n1.eps')
    """
    PLT.show()
    
    
if __name__ == '__main__':
    #ips_n1p_rainier()
    #ips_np1_della()
    #lin()
    ips()
