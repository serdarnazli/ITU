#BLG202E - NUMERICAL METHODS, SPRING 2023 
#HOMEWORK3 - QUESTION 2
#PYTHON IMPLEMENTATION OF DIVIDED DIFFERENCES

#Author: Muhammet Serdar NAZLI - 150210723

import matplotlib.pyplot as plt 
import numpy as np 

def divided_diff(x:list,f:list) -> list:
    """
    Args:
    x: x values
    f: f(x) values

    return:
    table: 2D list that holds divided difference table. (e.g [x,f(x),1st_diffs,2st_diffs,...])
    """

    #Divided difference number. (To understand which divided differences we are finding in the loop.)
    dd_num = 1

    #Divided differences will be added to this variable and will be printed at the end.
    table = [x,f]


    while (len(x) != dd_num):
        #Results of the divided differences process.
        results = []
        #Numbers that will be processed. Last column of the table(e.g. for finding 2nd divided diffs. 'nums' variable will consist of 1st divided diffs.)
        nums = table[-1]

        #Making the divided differences process.
        for i in range(len(nums)-1):
            #Formula
            result = (nums[i+1] - nums[i]) / (x[i+dd_num]-x[i])
            #Add the result to results list
            results.append(result)
        
        #All of the results of a one divided diffs. Add them to the table.
        table.append(results)

        #Increase divided difference number by 1.
        dd_num += 1 
    
    #Termination. 
    return table


#This function is only for getting a proper output. There is no calculation.
def print_table(table):
    #Creating a output table consist of 0's in order to get a proper(readable) table output.
    size = len(table[0])
    output = [[0] * (size+1) for _ in range(size)]


    #First two columns will be x,f(x) respectively
    for j in range(size):
        output[j][0] = str(table[0][j])
        output[j][1] = str(table[1][j])

    #Rest of the columns will be consist of ith divided diffs.
    for i in range(2,size):
        for j in range(size-i+1):
            output[j][i] = str(table[i][j])

    #Last column
    output[0][-1] = str(table[-1][0])

    #!! Make attention that output table is filled with strings.
    #Filtering 0's(integer ones), So that only real table will be in the output.
    output = [[item for item in row if item != 0] for row in output]

    #Printing the column names
    print("Divided Difference Table(ith means ith divided difference):")
    print("x\tf(x)\t",end="")
    for i in range(1,size):
        print(f"{i}th",end="\t")
    print("\n")

    #Printing the output table.
    print('\n'.join(['\t'.join([cell for cell in row]) for row in output]))



def graph_q2(x,f):

    plt.scatter(x,f,color='orange',label='Data Points')

    def f(x):
        return 1+(x-1)+(1/4)*(x-1)*(x-3)+(-7/60)*(x-1)*(x-3)*(x-5)

    x = np.linspace(-1,7,200)
    y = f(x)

    plt.plot(x,y,label='Divided Difference Interpolation($x^3+7x+1$)')

    x1 = 4.2
    y1 = f(x1)

    plt.scatter(x1,y1,color='red',label=f'Subquestion C) $(4.2,{y1:.3f})$')
    plt.legend()
    plt.title("Sanity Test of Q2")
    plt.show()


if __name__ == "__main__":
    #x values.
    x = [1,3,5,6]
    #f(x) values.
    f = [1,3,7,8]

    #Getting the table
    table = divided_diff(x,f)

    #Printing the table
    print_table(table)

    graph_q2(x,f)

            
