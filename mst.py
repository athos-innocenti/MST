import random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class UnionFind:
    def __init__(self):
        self.list = []

    def make_set(self, x):
        self.list.append(List(x))

    def find_set(self, x):
        return self.list[x].represent

    def union(self, u, v):
        if len(self.list[self.list[u].represent].list) > len(self.list[self.list[v].represent].list):
            new_represent = self.list[u].represent
            old_represent = self.list[v].represent
            self.list[new_represent].list.extend(self.list[old_represent].list)
            for i in range(len(self.list[old_represent].list)):
                self.list[old_represent].list.pop()
            for i in range(len(self.list)):
                if self.list[i].represent == old_represent:
                    self.list[i].represent = new_represent
        else:
            new_represent = self.list[v].represent
            old_represent = self.list[u].represent
            self.list[new_represent].list.extend(self.list[old_represent].list)
            for i in range(len(self.list[old_represent].list)):
                self.list[old_represent].list.pop()
            for i in range(len(self.list)):
                if self.list[i].represent == old_represent:
                    self.list[i].represent = new_represent


class List:
    def __init__(self, x):
        self.list = []
        self.list.append(x)
        self.represent = x


class Arch:
    def __init__(self, u, v, weigth):
        self.u = u
        self.v = v
        self.weigth = weigth


def connected_components(size, arches):
    union_find = UnionFind()
    for v in range(size):
        union_find.make_set(v)
    for k in range(len(arches)):
        u = arches[k].u
        v = arches[k].v
        if union_find.find_set(u) != union_find.find_set(v):
            union_find.union(union_find.find_set(u), union_find.find_set(v))
    count_cc = 0
    for i in range(len(union_find.list)):
        if len(union_find.list[i].list) != 0:
            count_cc += 1
    print "Numero componenti connesse:", count_cc
    return count_cc


def mst_kruskal(arches, size):
    a = []
    op = UnionFind()
    for v in range(size):
        op.make_set(v)
    arches.sort(key=lambda x: x.weigth)
    for k in range(len(arches)):
        u = arches[k].u
        v = arches[k].v
        if op.find_set(u) != op.find_set(v):
            a.append((u, v))
            op.union(op.find_set(u), op.find_set(v))


def kruskal_test(size, arches):
    start = timer()
    mst_kruskal(arches, size)
    end = timer()
    time_k = end - start
    print "Tempo di esecuzione Kruskal:", time_k
    return time_k


def adiacent_matrix_creation(size, prob):
    arches = []
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i):
            if random.randint(1, 100) <= prob and matrix[i][j] == 0:
                matrix[i][j] = random.randint(1, 20)
                matrix[j][i] = matrix[i][j]
                arches.append(Arch(j, i, matrix[i][j]))
    return arches


def average(values):
    values_sum = 0
    avg = 0
    if len(values) != 0:
        for i in range(len(values)):
            values_sum += values[i]
        avg = values_sum / len(values)
    return avg


def main():
    tries = 20
    max_prob = 101
    prob_vect = np.arange(0, max_prob)

    for size in (25, 100, 500):
        time_mst = []
        cc_count = []
        for prob in range(0, max_prob):
            print "Probabilita:", prob
            mst_tries = []
            cc_tries = []
            for j in range(0, tries):
                arches = adiacent_matrix_creation(size, prob)
                arches_k = arches[:]
                cc_tries.append(connected_components(size, arches))
                if cc_tries[j] == 1:  # mst solo su grafi con una sola cc
                    mst_tries.append(kruskal_test(size, arches_k))
            cc_count.append(average(cc_tries))
            time_mst.append(average(mst_tries))
        plt.plot(prob_vect, cc_count)
        plt.xlabel('Probabilita di avere un arco')
        plt.ylabel('Numero di Componenti connesse')
        plt.show()
        plt.plot(prob_vect, time_mst)
        plt.xlabel('Probabilita di avere un arco')
        plt.ylabel('Tempo di esecuzione Kruskal')
        plt.show()


if __name__ == '__main__':
    main()
