import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--list', nargs='+', required=True)
parser.add_argument('--op', type=str, default='add')
opts = parser.parse_args()
NUM = 56
print("Am I here?")
print(vars(opts))

def func(op, a=[1.2, 3.4]):
    if op == 'add':
        return sum([float(ai) for ai in a])
    elif op == 'max':
        return max([float(ai) for ai in a])

print("And also here")

if __name__=='__main__':

    print(func('add', opts.list))
    print(func('max'))
