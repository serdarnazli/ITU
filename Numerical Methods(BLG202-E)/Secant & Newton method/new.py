import math

class Methods:
    def __init__(self):
        pass

    def secant_method(self, x0, x1, max_iter, s, f):
        all_iter = [x0, x1]
        for i in range(max_iter):
            #Formula for secant method
            x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            all_iter.append(x2)

            #tolarence cvalue
            if abs(x2 - x1) < s:
                break

            x0 = x1
            x1 = x2

        return all_iter
    
    def newton_method(self, x0, max_iter, s, f, f_prime):
        all_iter = [x0]
        for i in range(max_iter):
            #Formula for newton method
            x1 = x0 - f(x0) / f_prime(x0)
            all_iter.append(x1)

            #tolarence
            if abs(x1 - x0) < s:
                break
            x0 = x1
        return all_iter


    def convergence_rate(self,all_iter):
        xn = all_iter[-1]
        xn1 = all_iter[-2]
        rate = abs((xn-xn1)/(xn-xn1)) 
        return rate





if __name__ == "__main__":

    #Function
    def f(x):
        return 4*math.log(x) - x

    #Derivative of the function
    def f_prime(x):
        return 4/x - 1
    
    mode = input("1 for newton\n2 for secant:\n")
    methods = Methods()
    if mode == "1":
        x0 = float(input("x0: "))
        max_iter = int(input("max iter: "))
        tolarence = float(input("tolarence: "))
        all_iter = methods.newton_method(x0,max_iter,tolarence, f, f_prime)
        print(all_iter)
        print("convergence rate: ",methods.convergence_rate(all_iter))
    if mode == "2":
        x0 = float(input("x0: "))
        x1 = float(input("x1: "))
        max_iter = int(input("max iter: "))
        tolarence = float(input("tolarence: "))
        all_iter = methods.secant_method(x0,x1,max_iter,tolarence, f)
        print(all_iter)
        print("convergence rate: ",methods.convergence_rate(all_iter))

