# 生成快速排序
def reserverList(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        return reserverList(less) + [pivot] + reserverList(greater)
#测试一下
print(reserverList([1, 3, 5, 7, 9, 2, 4, 6, 8, 0]))
# 生成冒泡排序
def bubbleSort(array):
    for i in range(len(array)):
        for j in range(len(array) - 1 - i):
            if array[j] > array[j+1]:
                array[j],array[j+1] = array[j+1],array[j]
    return array

#测试一下
print(bubbleSort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0]))


