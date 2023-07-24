#BLG202E - NUMERICAL METHODS, SPRING 2023 
#HOMEWORK3 - QUESTION 3
#PYTHON IMPLEMENTATION OF LAGRANGE INTERPOLATION

#Author: Muhammet Serdar NAZLI - 150210723

import matplotlib.pyplot as plt 
import numpy as np 

#Main lagrange interpolater function
def lagrange_interpolation(x,y,num=None):
    """
    Args:
    x(list): x values
    y(list): f(x) values
    num(int or float): number represents a x-value, will be used to find P(x) if wanted.
    
    Return:
    str: general formula of the lagrange interpolation as string. 
    -> !!The line consist of --- stand for division operation.!!
    float: result of P(x)
    """
    n = len(x)
    #Initial values
    formula = ""
    final_result = 0

    #f_n(x) = \sigma_{0}^{n} L_i(x)*f(x_i) part.
    for i in range(n):
        #Getting numerator, denominator and result of L_i(x)
        numerator,denom,result = L(x,n,i,num)

        #If a number is given as parameter, calculating P(x) of it.
        if num != None:
            final_result += result* y[i]
        
        #Adding numerator and denominator to the formula in a way that it is readable.
        formula += f"{numerator}({y[i]})\n{'-'*(10*n-5)}\n{denom}\n + \n"
    
    #Termination
    return formula.rstrip('+ \n'),final_result





def L(x,n,order,num=None):
    """
    Args:
    x(list): x values
    n(int): n value in the formula of L_i(x)
    order: i in the L_i(x)
    num(int or float): number represents a x-value, will be used to find P(x) if wanted.
    
    Return:
    numerator(str): numerator result of L_i(x). e.g (x-2)(x-1)
    denom(str): denominator result of L_i(x).   e.g (1-2)(1-4)
    result(int or float): result of (numerator/denominator), where x = num.
    """
    #Initial values
    result = 1
    numerator = ""
    denom = ""
    #\Pi_{j=0}^{n} (x-x_j)/(x_i-x_j) part. In this implementation 'order' stands for 'i', 
    #'i' in the for loop stands for 'j'.
    for i in range(n):
        if i != order:
            numerator += f"(x - {x[i]})".replace('- -','+ ')
            denom += f"({x[order]} - {x[i]})".replace('- -','+ ')
            if num != None:
                result *= (num-x[i])/(x[order]-x[i])

    #Termination.
    return numerator,denom,result


def graph_q3(x,y):

    plt.scatter(x,y,color='red',label='Data points')

    def f(x):
        return (23/30)*x**3 - (97/30)*x**2 + (52/15)*x - 1

    x = np.linspace(-5,5,100)
    y = f(x)

    plt.plot(x,y,label='Lagrange Interpolation $P(x)=x^2+x-6$')
    plt.legend()
    plt.title("Sanity test of Q3")
    plt.show()


#Main
if __name__ == "__main__":

    num=3  #Change this as you wish.

    #x and f(x) values
    x = [-2, 0, 1,3]        #Change these as you wish.
    y = [-27,-1,0,1]

    #Calculating the formula and the result of P(num).
    inter_formula,result = lagrange_interpolation(x,y,num)
    #Printing the formula of the interpolated polynomial and the result of P(num).
    print("FORMULA OF INTERPOLATED POLYNOMIAL P(X):\n",inter_formula,f"\n\nResult of P({num}):",result,sep='')

    #graph_q3(x,y)







