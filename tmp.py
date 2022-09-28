# my_global = 99

# globals()

import datetime

start = datetime.datetime.now()

def my_func1():
    globals()
    my_global = 77
    print(my_global)


def my_func2():
    global my_global
    my_global = 77
    print(my_global)


end = datetime.datetime.now()
delta = (end - start).microseconds
print(delta)
    

my_func1()
# print(my_global)
# my_func1()
# my_func2()
# print(my_global)
