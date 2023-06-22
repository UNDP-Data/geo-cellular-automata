import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
import time


def flush_cache(passwd):
    """
    Flush Linux VM caches. Useful for doing meaningful tmei measurements for
    NetCDF or similar libs.
    Needs sudo password
    :return: bool, True if success, False otherwise
    """
    logger.debug('Clearing the OS cache using sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches')
    #ret = os.system('echo %s | sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"' % passwd)
    ret = os.popen('sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"', 'w').write(passwd)
    return not bool(ret)

def timeit(func=None,loops=1,verbose=False, clear_cache=False, sudo_passwd=None):
    #print 0, func, loops, verbose, clear_cache, sudo_passwd
    if func != None:
        if clear_cache:
            assert sudo_passwd, 'sudo_password argument is needed to clear the kernel cache'

        def inner(*args,**kwargs):
            sums = 0.0
            mins = 1.7976931348623157e+308
            maxs = 0.0
            logger.debug('====%s Timing====' % func.__name__)
            for i in range(0,loops):
                if clear_cache:
                    flush_cache(sudo_passwd)
                t0 = time.time()
                result = func(*args,**kwargs)
                dt = time.time() - t0
                mins = dt if dt < mins else mins
                maxs = dt if dt > maxs else maxs
                sums += dt
                if verbose == True:
                    logger.debug('\t%r ran in %2.9f sec on run %s' %(func.__name__,dt,i))
            logger.debug('%r min run time was %2.9f sec' % (func.__name__,mins))
            logger.debug('%r max run time was %2.9f sec' % (func.__name__,maxs))
            logger.info('%r avg run time was %2.9f sec in %s runs' % (func.__name__,sums/loops,loops))
            logger.debug('==== end ====')
            return result

        return inner
    else:
        def partial_inner(func):
            return timeit(func,loops,verbose, clear_cache, sudo_passwd)
        return partial_inner


def cdist(n=3):
    y, x = np.indices((n,n), dtype='u2')
    m = n//2
    t = (m-y)**2 + (m-x)**2
    return t**.5

def distances(xy1, xy2):
    d0 = np.subtract.outer(xy1[:,0], xy2[:,0])
    d1 = np.subtract.outer(xy1[:,1], xy2[:,1])
    return np.hypot(d0, d1)

def distances( c):
    n = ((c.size/2)**.5//2)
    d0 = np.subtract.outer(n, c[:,0])
    d1 = np.subtract.outer(n, c[:,1])
    return np.hypot(d0, d1)



def distances1( n):
    c= np.arange(n)
    x = np.lib.stride_tricks.as_strided(c, (n,n), strides=(0,c.dtype.itemsize)).ravel()
    y = np.lib.stride_tricks.as_strided(c, (n,n), strides=(c.dtype.itemsize, 0)).ravel()
    m = n//2
    d0 = np.subtract.outer(m, y)
    d1 = np.subtract.outer(m, x)
    return np.hypot(d0, d1)




if __name__ == '__main__':
    n =5


    print(distances1(3))