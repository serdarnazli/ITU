#Author: Muhammet Serdar NAZLI - 150210723


def f(a,x):
    """python function to calculate value of the mathematical function in the question.
    Args:
        a: the number to find the fifth root of
        x: x value
    Return:
        a-x**5
    """

    return a-x**5


def fifth_root_bisection(a:float,e:float) -> float:
    """
    main function for bisection which will handle all of the process to find fifth root of given a.
    Args:
        a: the number to find the fifth root of
        e: epsilon value
    Return:
        x: fifth root of a
    """

    #Left point of the interval initialized with 0. Since we are looking for fifth root, it is obvious that it will take place in [0,a] or [a,0](if a is negative)
    left = 0

    #Right point of the interval initialized with a. If a is negative, treat it like positive then at the end add a minus sign to result.
    if a <0:
        a = -a
        a_negative = True
    else:
        a_negative = False
    right = a

    #Middle point
    x = (right + left)/2

    while True:
        # If condition is satisfied, break the loop.
        # Here's something I'm not sure about. The question asks us for |a^(1/5)-x| < e. but I don't understand the logic of using bisection in this case if we already know a^(1/5). 
        # Shouldn't we be after the |a-x^5| < e?
        if abs(a**(1/5)-x) <= e:
            break
        
        #if f(left)*f(middle) <0
        if f(a,left)*f(a,x) < 0:
            right = x
            x = (left+right)/2

        # if f(right)*f(middle) <0
        elif f(a,right)*f(a,x) < 0:
            left = x
            x = (left+right)/2

    if a_negative:
        return -x
    return x

    




if __name__ == "__main__":
    #Taking inputs for a and e. e corresponds to epsilon.
    a = float(input("a: "))
    e = float(input("e: "))

    # bisection function call with a and e, printing.
    print("Result: ", fifth_root_bisection(a,e),sep="")