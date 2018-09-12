import pickle
import pprint

def loadall(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return

def printdata(lst, heading):
    print(heading)
    print('---')
    pprint.pprint(lst)
    print('\n')


if __name__ == '__main__':
    psm1_data = list(loadall('psm1_recordings.txt'))
    psm2_data = list(loadall('psm2_recordings.txt'))
    printdata(psm1_data, 'PSM1 DATA')
    printdata(psm2_data, 'PSM2 DATA')
