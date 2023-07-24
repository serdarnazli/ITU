#@Author: Muhammet Serdar NAZLI - 150210723


def integer_to_binary(number):
    """
    Function to convert integer with base 10 to integer with base 2.
    Args:
        number: integer number with base 10
    Return:
        binary version of number"""
    

    # Initialization of binary
    binary = ""


    # As the number is greater than 0, we will divide it by 2 and update our binary according to it.
    while number > 0:
        # Dividing and remainer part.
        digit = (number % 2)

        # If digit is 1 add 1 to binary else add 0
        if digit == 1:
            binary += "1"
        else:
            binary += "0"

        # Making integer dividing.
        number = number // 2

    # Termination. Since the process creates binary in reverse version, I reversed it when returning it.
    return binary[::-1]




def rational_binary(number):
    """
    Main function for float to binary.
    Args:
        number: float number
    Return:
        binary version of the number.
    """

    # If number is negative, treating it like positive and adding '-' to the head of binary.
    if number < 0:
        binary = "-"
        number = -number
    else:
        binary = ""

    # If float is greater than 1, seperating the decimal and integer parts. 
    if (number > 1):
        #Getting the integer part.
        integer = int(number)

        #Getting the decimal part.
        decimal = abs(number) - abs(integer)

        #Binary = prefix + integer part with base 2 + .  -> decimal part will be handled later.
        binary = binary + integer_to_binary(integer) + "."

    # If float is not greater than 1 then inital binary is 0. ->decimal part will be handled later.  
    else:
        integer = 0
        decimal = number
        binary += "0."
    


    # Getting numerator and denominator representation of decimal part.
    numerator, denominator = decimal.as_integer_ratio()

    # A flag variable. To be able to put 0 if the multiplication by 2 was performed in the previous step and nothing was obtained in the loop.
    is_multiplied_by_2 = False

    # Main loop
    while True:

        # Termination of the loop. 
        if numerator == denominator:
            binary += "1"
            break

        # If numerator is greater than denominator, substracting 1 from numerator.
        if numerator > denominator:
            is_multiplied_by_2 = False
            binary += "1"
            numerator -= denominator

        # Else, multiplying numerator by 2.
        else:
            #If the previous step was also multiplication, then adding 0 to binary.
            if is_multiplied_by_2:
                binary += "0"
            
            numerator *= 2 
            is_multiplied_by_2 = True
    
    # Termination.
    return  binary 



if __name__ == "__main__":
    #Taking input
    number = float(input("Number: "))
    #Main function call and printing the result.
    print("Result: ",rational_binary(number),sep="")




