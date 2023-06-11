from multiprocessing import Process
import os
import time

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

# def f(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     time.sleep(2)
#     p.start()
#     p.join()

import multiprocessing as mp

def foo():
    info('hello')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # q = mp.Queue()
    p = mp.Process(target=foo, args=())
    p.start()
    # print(q.get())
    p.join()