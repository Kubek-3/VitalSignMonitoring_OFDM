import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tracemalloc

n = 1000   # matrix lenght
a = 10      # scale factor
data = np.array(np.random.rand(100000)) # sample data


def f(num, data):
    M = len(data)
    t = np.arange(M)
    sqrt = np.sqrt(data, dtype=np.float64)
    res = num * (sqrt + t)
    return res

def f_parallel(nums, data, workers=4):
    def worker(num):
        s = np.zeros_like(data, dtype=np.float64)
        for n in num:
            s += f(n, data)
        return s
    
    chunks = np.array_split(nums, workers)
    #print("chunks", chunks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(worker, chunks))
    res = np.sum(results, axis=0)
    print("result from workers", res) 
    print("shape of result from workers", res.shape) 
    return res


def f_vectorized(num, data):
    part_sum = np.vectorize(lambda n: f(n, data), otypes=[np.ndarray])(num)
    summm = np.zeros_like(data, dtype=np.float64)
    summm = np.add.reduce(part_sum, axis=0)
    print("rseult from vectorized", summm)
    print("shape of result from vectorized", summm.shape)
    return summm

# tracemalloc.start()
# sum_parallel = f_parallel(np.arange(n), data)
# print("total mem for  WORKERS", tracemalloc.get_traced_memory())
# tracemalloc.stop()

# tracemalloc.start()
# sum_vectorized = f_vectorized(np.arange(n), data)
# print("total mem for VECTORIZED", tracemalloc.get_traced_memory())
# tracemalloc.stop()
para = f_parallel(np.arange(n), data)
vect = f_vectorized(np.arange(n), data)
testt = para - vect
print("difference between workers and vectorized", testt)
np.testing.assert_almost_equal(para, vect)